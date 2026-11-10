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
"""Tool-level SCOUT controller; the single-turn reproduction remains separate."""

from uuid import NAMESPACE_URL, uuid5

import transfer_queue as tq
from omegaconf import OmegaConf

from verl.experimental.scout.tool_rollout import ToolPrefixContext, select_tool_boundary
from verl.experimental.scout.tool_teacher import ToolTeacherRuntime
from verl.experimental.scout.trainer import SCOUTTrainer, _unwrap


class ToolSCOUTTrainer(SCOUTTrainer):
    """Adapt the teacher on executable student states and distill action tokens."""

    teacher_runtime_class = ToolTeacherRuntime

    def _configure_teacher(self):
        self.teacher.set_student_tokenizer(self.tokenizer)

    def _state(self):
        state = super()._state()
        state["tool_config"] = OmegaConf.to_container(self.config.scout.tool, resolve=True)
        state["tool_chat_template_kwargs"] = OmegaConf.to_container(
            self.config.data.apply_chat_template_kwargs, resolve=True
        )
        return state

    def _get_prefix_contexts(self, batch, ratio):
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=[
                "prompts",
                "responses",
                "response_mask",
                "data_source",
                "reward_model",
                "extra_info",
                "extra_fields",
            ],
        )
        contexts = []
        for i, (key, tag) in enumerate(zip(batch.keys, batch.tags, strict=True)):
            if tag.get("is_padding", False):
                continue
            prompt = data["prompts"][i][: tag["prompt_len"]].tolist()
            response = data["responses"][i][: tag["response_len"]].tolist()
            mask = data["response_mask"][i][: tag["response_len"]].tolist()
            extra = _unwrap(data["extra_fields"][i])
            point = select_tool_boundary(
                extra.get("scout_tool_boundaries"), response, mask, ratio, self.config.scout.tool.prefix_source
            )
            contexts.append(
                ToolPrefixContext(
                    uid=uuid5(NAMESPACE_URL, f"scout-tool:{self.global_steps}:{key}:{point['offset']}").hex,
                    prompt_ids=prompt,
                    prefix_ids=response[: point["offset"]],
                    data_source=_unwrap(data["data_source"][i]),
                    reward_model=_unwrap(data["reward_model"][i]),
                    extra_info=_unwrap(data["extra_info"][i]) or {},
                    boundary=point,
                )
            )
        return contexts
