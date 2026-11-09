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
"""Synchronous SCOUT controller. No changes to verl's default PPO trainer."""

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import transfer_queue as tq

from verl.experimental.scout.config import make_teacher_config, validate_scout
from verl.experimental.scout.teacher import TeacherRuntime
from verl.experimental.scout.utils import PrefixContext, prefix_ratio, select_prefix, validate_tokenizers
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


def _unwrap(value):
    # TransferQueue may return a NonTensorData or an ordinary Python object.
    return value.data if hasattr(value, "data") and not hasattr(value, "shape") else value


class SCOUTTrainer(PPOTrainerSync):
    """Add periodic teacher GRPO updates to the synchronous student OPD loop."""

    def __init__(self, config):
        validate_scout(config)
        super().__init__(config)
        self.teacher = None

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        # Reserve all three pools together, so insufficient GPU capacity fails
        # before starting either model. The teacher scoring pool already exists.
        self.resource_pool_manager.resource_pool_spec["scout_teacher_train"] = [
            self.config.scout.teacher_gpus_per_node
        ] * self.config.scout.teacher_nnodes

    def _setup(self):
        super()._setup()
        teacher_config = make_teacher_config(self.config, self.total_training_steps)
        managers = self.teacher_model_manager.teacher_model_managers
        teacher_key = next(iter(managers))
        self.teacher = TeacherRuntime(
            teacher_config,
            self.resource_pool_manager.resource_pool_dict["scout_teacher_train"],
            managers[teacher_key],
            self.teacher_model_manager.get_client()[teacher_key],
            self.get_reward_handles(),
        )
        validate_tokenizers(self.tokenizer, self.teacher.tokenizer)
        if self.processor is not None or self.teacher.processor is not None:
            raise ValueError("This SCOUT implementation supports text-only models")
        if self.global_steps:
            folder = (
                Path(self.config.trainer.resume_from_path)
                if self.config.trainer.resume_mode == "resume_path"
                else self._step_folder()
            )
            state = json.loads((folder / "scout_state.json").read_text())
            self._validate_resume_state(state)
            self.teacher.actor_rollout_wg.load_checkpoint(
                local_path=str(folder / "teacher"), del_local_after_load=False
            )
            self.teacher.updates = state["teacher_updates"]

    def on_init_end(self):
        super().on_init_end()
        # This is also mandatory on resume: inference initially loads the
        # released model, not the updated teacher checkpoint.
        self.teacher.sync()

    def _step_folder(self):
        return Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"

    def _state(self):
        return {
            "schema_version": 1,
            "student_step": self.global_steps,
            "teacher_updates": self.teacher.updates,
            "teacher_model": self.teacher.config.actor_rollout_ref.model.path,
            "student_model": self.config.actor_rollout_ref.model.path,
            "curriculum_total_steps": self.total_training_steps,
            "interval": self.config.scout.teacher_update_interval,
            "prefix_ratio_start": self.config.scout.prefix_ratio_start,
            "prefix_ratio_end": self.config.scout.prefix_ratio_end,
            "teacher_rollouts_per_prefix": self.config.scout.teacher_rollouts_per_prefix,
            "teacher_lr": self.config.scout.teacher_lr,
            "teacher_kl_coef": self.config.scout.teacher_kl_coef,
            "teacher_max_response_length": self.config.scout.teacher_max_response_length,
            "seed": self.config.scout.seed,
        }

    def _validate_resume_state(self, state):
        expected = self._state()
        for key, value in expected.items():
            if key != "teacher_updates" and state.get(key) != value:
                raise ValueError(f"SCOUT checkpoint mismatch for {key}: {state.get(key)!r} != {value!r}")
        if state["teacher_updates"] != self.global_steps // self.config.scout.teacher_update_interval:
            raise ValueError("Checkpoint teacher update count does not match student progress")

    def _save_checkpoint(self):
        folder = self._step_folder()
        folder.mkdir(parents=True, exist_ok=True)
        # Save the teacher and its optimizer before the parent publishes the
        # latest-step pointer. No checkpoint can silently resume a frozen teacher.
        self.teacher.actor_rollout_wg.save_checkpoint(
            str(folder / "teacher"),
            None,
            self.global_steps,
            max_ckpt_to_keep=self.config.trainer.max_actor_ckpt_to_keep,
        )
        temporary = folder / "scout_state.json.tmp"
        temporary.write_text(json.dumps(self._state(), indent=2) + "\n")
        temporary.replace(folder / "scout_state.json")
        super()._save_checkpoint()

    def _step_once(self, metrics, timing_raw, sample_batch_size):
        batch = super()._step_once(metrics, timing_raw, sample_batch_size)
        if self.global_steps % self.config.scout.teacher_update_interval:
            return batch
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["prompts", "responses", "response_mask", "data_source", "reward_model", "extra_info"],
        )
        ratio = prefix_ratio(
            self.global_steps,
            self.total_training_steps,
            self.config.scout.prefix_ratio_start,
            self.config.scout.prefix_ratio_end,
        )
        eos = self.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        contexts = []
        for i, (key, tag) in enumerate(zip(batch.keys, batch.tags, strict=True)):
            if tag.get("is_padding", False):
                continue
            prompt = data["prompts"][i][: tag["prompt_len"]].tolist()
            response = data["responses"][i][: tag["response_len"]].tolist()
            mask = data["response_mask"][i][: tag["response_len"]]
            if not mask.bool().all():
                raise ValueError("SCOUT expects a single uninterrupted student response, without tool observations")
            contexts.append(
                PrefixContext(
                    uid=uuid5(NAMESPACE_URL, f"scout:{self.global_steps}:{key}").hex,
                    prompt_ids=prompt,
                    prefix_ids=select_prefix(response, ratio, eos_ids),
                    data_source=_unwrap(data["data_source"][i]),
                    reward_model=_unwrap(data["reward_model"][i]),
                    extra_info=(_unwrap(data["extra_info"][i]) or {}) if "extra_info" in data else {},
                )
            )
        if len(contexts) != sample_batch_size:
            raise RuntimeError("Unexpected SCOUT prefix batch size; refusing to change GRPO weighting")
        metrics["scout/prefix_ratio"] = ratio
        metrics.update(self.teacher.update_from_prefixes(contexts, self.global_steps))
        return batch

    def fit(self, agent_loop_manager):
        try:
            return super().fit(agent_loop_manager)
        finally:
            if self.teacher is not None:
                self.teacher.checkpoint_manager.close()
