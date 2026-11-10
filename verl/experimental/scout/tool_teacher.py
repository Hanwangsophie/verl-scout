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
"""Teacher GRPO continuations from independently restored student tool states."""

import asyncio
from uuid import uuid4

from omegaconf import OmegaConf

from verl.experimental.scout.teacher import TeacherRuntime
from verl.experimental.scout.tool_env import json_copy, make_environment
from verl.experimental.scout.tool_rollout import run_tool_episode, tool_continuation_fields
from verl.utils.ray_utils import auto_await
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder


class ToolTeacherRuntime(TeacherRuntime):
    """Reuse SCOUT's teacher optimizer, frozen reference, GRPO grouping and sync."""

    def set_student_tokenizer(self, tokenizer):
        """Use the student's chat format for both policies' continuation contexts."""
        self.tool_tokenizer = tokenizer
        self.tool_builder = create_continuous_token_builder(
            tokenizer,
            hf_model_type=self.config.scout.tool.hf_model_type,
            chat_template_kwargs=OmegaConf.to_container(self.config.data.apply_chat_template_kwargs, resolve=True),
        )

    @auto_await
    async def _generate_and_score(self, contexts, student_step):
        tool = self.config.scout.tool
        limit = asyncio.Semaphore(self.config.scout.max_concurrent_requests)
        k = self.config.scout.teacher_rollouts_per_prefix

        async def one(context, index):
            async with limit:
                env = make_environment(
                    tool.environment_class, OmegaConf.to_container(tool.environment_config, resolve=True)
                )
                try:
                    await asyncio.wait_for(
                        env.restore(json_copy(context.boundary["snapshot"], tool.max_snapshot_bytes)),
                        tool.step_timeout_s,
                    )
                    episode = await run_tool_episode(
                        env=env,
                        generate=self.client.generate,
                        tokenizer=self.tool_tokenizer,
                        builder=self.tool_builder,
                        input_ids=context.teacher_input,
                        messages=context.boundary["messages"],
                        sampling_params={
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "logprobs": False,
                            "seed": (self.config.scout.seed + student_step * 100000 + index * (tool.max_steps + 1))
                            % (2**31 - 1),
                        },
                        request_id=f"scout-teacher-{student_step}-{index}-{uuid4().hex}",
                        max_steps=tool.max_steps,
                        max_action_tokens=tool.max_action_tokens,
                        max_response_tokens=self.config.scout.teacher_max_response_length,
                        completed_steps=context.boundary["completed_steps"],
                        expected_version=self.updates,
                        step_timeout_s=tool.step_timeout_s,
                    )
                    return tool_continuation_fields(context, episode)
                finally:
                    await asyncio.wait_for(env.close(), tool.step_timeout_s)

        tasks = [asyncio.create_task(one(context, j * k + i)) for j, context in enumerate(contexts) for i in range(k)]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
