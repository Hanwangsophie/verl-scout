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
"""Student tool trajectories with execution snapshots, registered through Hydra."""

import asyncio
from uuid import uuid4

from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.scout.tool_env import make_environment
from verl.experimental.scout.tool_rollout import ACTION_INSTRUCTION, observation_message, run_tool_episode


class ScoutToolAgentLoop(AgentLoopBase):
    """Run an action-only policy in one isolated environment per student trajectory."""

    merges_context_messages = True

    async def run(self, sampling_params, **kwargs):
        """Store public history and opaque snapshots separately in TransferQueue."""
        tool = self.config.scout.tool
        if self.processor is not None:
            raise ValueError("SCOUT tool mode supports text-only models")
        extra_info = kwargs.get("extra_info") or {}
        if "scout_tool_task" not in extra_info:
            raise ValueError("Tool dataset rows require extra_info.scout_tool_task")
        messages = list(kwargs["raw_prompt"])
        if any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("SCOUT tool prompts must contain text messages only")
        env = make_environment(tool.environment_class, OmegaConf.to_container(tool.environment_config, resolve=True))
        try:
            observation = await asyncio.wait_for(env.reset(extra_info["scout_tool_task"]), tool.step_timeout_s)
            messages = messages + [{"role": "user", "content": ACTION_INSTRUCTION}]
            messages.append(observation_message(observation, tool.max_steps))
            # Do not use the base helper's left truncation: a snapshot must describe
            # exactly the environment and history the policy actually observed.
            prompt_ids = self.continuous_token_builder.build_initial_tokens(messages)
            if len(prompt_ids) > self.rollout_config.prompt_length:
                raise ValueError("Tool prompt plus initial observation exceeds max_prompt_length")
            episode = await run_tool_episode(
                env=env,
                generate=self.server_manager.generate,
                tokenizer=self.tokenizer,
                builder=self.continuous_token_builder,
                input_ids=prompt_ids,
                messages=messages,
                sampling_params=sampling_params,
                request_id=f"scout-student-{uuid4().hex}",
                max_steps=tool.max_steps,
                max_action_tokens=tool.max_action_tokens,
                max_response_tokens=self.rollout_config.response_length,
                record_boundaries=True,
                max_snapshot_bytes=tool.max_snapshot_bytes,
                step_timeout_s=tool.step_timeout_s,
            )
        finally:
            await asyncio.wait_for(env.close(), tool.step_timeout_s)
        return AgentLoopOutput(
            prompt_ids=episode.prompt_ids,
            response_ids=episode.response_ids,
            response_mask=episode.response_mask,
            reward_score=episode.reward,
            num_turns=episode.num_turns,
            metrics={},
            extra_fields={
                **episode.extra_fields,
                "scout_tool_boundaries": episode.boundaries,
                "scout_tool_end_reason": episode.end_reason,
                "reward_extra_info": {
                    "invalid_actions": episode.invalid_actions,
                    "tool_steps": episode.steps,
                    "terminated": int(episode.end_reason == "done"),
                },
            },
        )
