# Copyright 2026 Individual Contributor: Hanwangsophie
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trainable teacher using verl engine workers, GRPO, and NCCL weight sync."""

import asyncio
from uuid import uuid4

import numpy as np
import ray
import torch
import transfer_queue as tq
from tensordict import TensorDict

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.scout.utils import continuation_fields, pack_teacher_rows, response_for_verifier
from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, create_colocated_worker_cls
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.ray_utils import auto_await
from verl.workers.engine_workers import ActorRolloutRefWorker


class TeacherRuntime(PPOTrainerSync):
    """Reuse V1 compute/update methods without a second dataloader or fit loop.

    The frozen reference is a separate worker loaded from the initial teacher
    checkpoint. It is never refreshed when the trainable teacher changes.
    """

    def __init__(self, config, resource_pool, inference_manager, client, reward_handles):
        super().__init__(config)
        self._init_tokenizer()
        self.ref_in_actor = False
        self.distillation_config = None
        self.global_steps = 0
        self.updates = 0
        self.client = client
        self.reward_handles = reward_handles
        if not reward_handles:
            raise ValueError("SCOUT needs rule-based reward workers")
        classes = {
            "actor": RayClassWithInitArgs(
                cls=ray.remote(ActorRolloutRefWorker), config=config.actor_rollout_ref, role="actor"
            )
        }
        if self.use_reference_policy:
            classes["ref"] = RayClassWithInitArgs(
                cls=ray.remote(ActorRolloutRefWorker), config=config.actor_rollout_ref, role="ref"
            )
        # Keep the owner alive: spawned groups share the colocated worker actors.
        self.worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=create_colocated_worker_cls(classes),
            name_prefix="scout_teacher_train",
            device_name=config.trainer.device,
        )
        groups = self.worker_group.spawn(prefix_set=classes.keys())
        self.actor_rollout_wg = groups["actor"]
        self.actor_rollout_wg.init_model()
        if self.use_reference_policy:
            self.ref_policy_wg = groups["ref"]
            self.ref_policy_wg.init_model()
        self.checkpoint_manager = CheckpointEngineManager(
            config=omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine),
            actor_wg=self.actor_rollout_wg,
            replicas=inference_manager.rollout_replicas,
        )

    def sync(self):
        """Publish the current teacher weights and wait for inference to commit them."""
        # update_weights coordinates request quiescence and KV-cache invalidation.
        # Waiting for commit makes the next student batch see a single teacher version.
        self.checkpoint_manager.update_weights(self.updates)
        self.checkpoint_manager.wait_for_weight_update()

    @auto_await
    async def _generate_and_score(self, contexts, student_step):
        limit = asyncio.Semaphore(self.config.scout.max_concurrent_requests)
        k = self.config.scout.teacher_rollouts_per_prefix

        async def one(context, index):
            async with limit:
                output = await self.client.generate(
                    request_id=f"scout-{student_step}-{index}-{uuid4().hex}",
                    prompt_ids=context.teacher_input,
                    sampling_params={
                        "max_tokens": self.config.scout.teacher_max_response_length,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "logprobs": False,
                        "seed": (self.config.scout.seed + student_step * 100000 + index) % (2**31 - 1),
                    },
                )
                if output.stop_reason == "aborted" or not output.token_ids:
                    raise RuntimeError("Incomplete teacher rollout; refusing to train on a partial GRPO group")
                version = output.extra_fields.get("global_steps")
                if version is not None and version != self.updates:
                    raise RuntimeError(f"Stale teacher rollout: expected {self.updates}, received {version}")
                # The verifier sees the full answer, although optimization sees only
                # the continuation. Avoid adding another assistant header or EOS.
                answer = response_for_verifier(context, output.token_ids)
                prompt = torch.tensor([context.prompt_ids], dtype=torch.long)
                response = torch.tensor([answer], dtype=torch.long)
                reward_batch = DataProto(
                    batch=TensorDict(
                        {
                            "prompts": prompt,
                            "responses": response,
                            "input_ids": torch.cat([prompt, response], dim=1),
                            "attention_mask": torch.ones(1, prompt.size(1) + response.size(1), dtype=torch.long),
                        },
                        batch_size=1,
                    ),
                    non_tensor_batch={
                        "data_source": np.array([context.data_source], dtype=object),
                        "reward_model": np.array([context.reward_model], dtype=object),
                        "extra_info": np.array([dict(context.extra_info)], dtype=object),
                    },
                )
                handle = self.reward_handles[index % len(self.reward_handles)]
                result = await handle.compute_score.remote(reward_batch)
                return continuation_fields(context, output.token_ids, float(result["reward_score"]))

        tasks = [asyncio.create_task(one(c, j * k + i)) for j, c in enumerate(contexts) for i in range(k)]
        try:
            return await asyncio.gather(*tasks)
        finally:
            # A failed request must not leave teacher scoring/generation running
            # while a caller handles the error or attempts a weight update.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def update_from_prefixes(self, contexts, student_step):
        """Generate complete GRPO groups, update the teacher, and refresh inference."""
        rows = self._generate_and_score(contexts, student_step)
        group_size = self.config.scout.teacher_rollouts_per_prefix
        scores = torch.stack([row["rm_scores"].sum() for row in rows])
        # GRPO is sequence-level: calculate one scalar advantage per continuation,
        # then broadcast only over its teacher-generated tokens.
        advantages, _ = compute_grpo_outcome_advantage(
            scores[:, None],
            torch.ones(len(rows), 1),
            np.array([row["uid"] for row in rows]),
            norm_adv_by_std_in_grpo=True,
        )
        for row, advantage in zip(rows, advantages[:, 0], strict=True):
            row["advantages"] = torch.ones_like(row["rm_scores"]) * advantage
            row["returns"] = row["advantages"].clone()
        keys = [f"{row['uid']}_{i % group_size}_0" for i, row in enumerate(rows)]
        tags = [
            {
                "seq_len": len(row["input_ids"]),
                "prompt_len": len(row["prompts"]),
                "response_len": len(row["responses"]),
                "status": "success",
            }
            for row in rows
        ]
        partition = f"scout-teacher-{student_step}"
        batch = tq.kv_batch_put(keys=keys, partition_id=partition, fields=pack_teacher_rows(rows), tags=tags)
        self.global_steps = self.updates + 1
        metrics = {}
        try:
            batch = self._balance_batch(batch, metrics)
            batch = self._compute_old_log_prob(batch, metrics)
            if self.use_reference_policy:
                batch = self._compute_ref_log_prob(batch, metrics)
            self._update_actor(batch, metrics)
            self.updates += 1
            self.sync()
        finally:
            tq.kv_clear(partition_id=partition)
        group_rewards = scores.reshape(-1, group_size)
        metrics.update(
            {
                "reward_mean": scores.mean().item(),
                "nonconstant_group_fraction": (group_rewards.std(dim=1) > 0).float().mean().item(),
                "updates": self.updates,
                "continuation_tokens": sum(len(row["responses"]) for row in rows),
                "prefix_count": len(contexts),
            }
        )
        return {f"scout/teacher/{key}": value for key, value in metrics.items()}
