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
"""Pure data transformations shared by the runtime and CPU tests."""

import math
from dataclasses import dataclass

import torch


def prefix_ratio(step: int, total_steps: int, start: float, end: float) -> float:
    """Linear curriculum; endpoints are explicit reproduction choices."""
    if total_steps < 1 or not 0 <= start <= end < 1:
        raise ValueError("Require total_steps >= 1 and 0 <= start <= end < 1")
    progress = min(1.0, max(0.0, (step - 1) / max(total_steps - 1, 1)))
    return start + progress * (end - start)


def select_prefix(response: list[int], ratio: float, eos_ids: set[int]) -> list[int]:
    """Cut before EOS and leave at least one student token outside the prefix.

    Input must already be unpadded. Never decode/re-tokenize: this preserves
    the exact student-generated context, including whitespace and reasoning tags.
    """
    if not 0 <= ratio < 1:
        raise ValueError("prefix ratio must be in [0, 1)")
    length = next((i for i, token in enumerate(response) if token in eos_ids), len(response))
    return response[: min(math.floor(length * ratio), max(length - 1, 0))]


@dataclass
class PrefixContext:
    """Original verifier question plus an exact, unretokenized student prefix."""

    uid: str
    prompt_ids: list[int]
    prefix_ids: list[int]
    data_source: str
    reward_model: dict
    extra_info: dict

    @property
    def teacher_input(self) -> list[int]:
        """Concatenate question and prefix for teacher conditioning."""
        return self.prompt_ids + self.prefix_ids


def continuation_fields(context: PrefixContext, continuation: list[int], score: float) -> dict:
    """Prefix is conditioning only. The response/loss covers continuation only."""
    if not continuation:
        raise ValueError("Teacher produced an empty continuation; refusing an empty loss mask")
    if not math.isfinite(score):
        raise ValueError("Verifier returned a non-finite reward")
    prompt = torch.tensor(context.teacher_input, dtype=torch.long)
    response = torch.tensor(continuation, dtype=torch.long)
    scores = torch.zeros(len(continuation), dtype=torch.float32)
    scores[-1] = score
    return {
        "prompts": prompt,
        "responses": response,
        "input_ids": torch.cat([prompt, response]),
        "position_ids": torch.arange(len(prompt) + len(response)),
        "response_mask": torch.ones(len(response), dtype=torch.long),
        "loss_mask": torch.ones(len(response), dtype=torch.long),
        "rm_scores": scores,
        "uid": context.uid,
        "data_source": context.data_source,
        "reward_model": context.reward_model,
        "extra_info": context.extra_info,
        "num_turns": 2,
    }


def validate_tokenizers(student, teacher) -> None:
    """Sampled-token OPD requires identical token-to-ID maps, not just vocab size."""
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("SCOUT currently requires identical student/teacher token-to-ID vocabularies")
    for key in ("eos_token_id", "pad_token_id", "bos_token_id"):
        if getattr(student, key, None) != getattr(teacher, key, None):
            raise ValueError(f"Student and teacher {key} differ")


def pack_teacher_rows(rows):
    """V1 expects jagged tensors even when every continuation has equal length."""
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    data = list_of_dict_to_tensordict(rows)
    for key in rows[0]:
        if isinstance(rows[0][key], torch.Tensor):
            data[key] = torch.nested.as_nested_tensor([row[key] for row in rows], layout=torch.jagged)
    return data


def response_for_verifier(context: PrefixContext, continuation: list[int]) -> list[int]:
    """Reward sees the entire answer, including the student prefix."""
    return context.prefix_ids + continuation
