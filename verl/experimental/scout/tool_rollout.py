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
"""Shared student/teacher executor with exact tokens and decision-boundary snapshots."""

import asyncio
import json
import math
from dataclasses import dataclass, field

from verl.experimental.scout.tool_env import json_copy
from verl.experimental.scout.utils import PrefixContext, continuation_fields

ACTION_INSTRUCTION = (
    'At each turn output exactly one JSON object: {"name": "tool_name", "arguments": {...}}. '
    "Do not output reasoning, Markdown, or text outside the JSON object. "
    "Execute finish to end the task. Environment observations will follow your actions."
)


def parse_action(text):
    """Accept only one action object; malformed samples remain in the training data."""
    try:
        action = json.loads(text)
    except (ValueError, TypeError):
        return None
    if (
        not isinstance(action, dict)
        or set(action) != {"name", "arguments"}
        or not isinstance(action["name"], str)
        or not isinstance(action["arguments"], dict)
    ):
        return None
    try:
        return json_copy(action)
    except ValueError:
        return None


def observation_message(observation, steps_left):
    """Render public feedback and the remaining decision budget as context only."""
    return {"role": "user", "content": f"Environment observation:\n{observation}\nActions remaining: {steps_left}"}


@dataclass
class ToolEpisode:
    """One execution trace; observations stay in context with a zero loss mask."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    boundaries: list[dict]
    reward: float
    steps: int
    invalid_actions: int
    end_reason: str
    num_turns: int
    extra_fields: dict = field(default_factory=dict)


@dataclass
class ToolPrefixContext(PrefixContext):
    """Student-visible exact prefix plus an opaque environment-only checkpoint."""

    boundary: dict


async def run_tool_episode(
    *,
    env,
    generate,
    tokenizer,
    builder,
    input_ids,
    messages,
    sampling_params,
    request_id,
    max_steps,
    max_action_tokens,
    max_response_tokens,
    completed_steps=0,
    record_boundaries=False,
    max_snapshot_bytes=262144,
    expected_version=None,
    step_timeout_s=60.0,
):
    """Execute fresh actions/observations in an already reset or restored environment.

    The caller owns reset/restore/close. Model input contains only visible history.
    A snapshot is recorded immediately BEFORE a decision, after the preceding tool
    observation has been appended in full. Length-limited actions are not executed.
    """
    if not 0 <= completed_steps < max_steps or min(max_action_tokens, max_response_tokens) < 1:
        raise ValueError("Tool continuation needs at least one remaining action and token")
    initial = list(input_ids)
    tokens, mask = list(initial), []
    messages = json_copy(messages, max_bytes=2**31)
    boundaries, versions = [], []
    reward, invalid_actions, steps = 0.0, 0, completed_steps
    num_turns = 1  # Treat the saved prefix as one conditioning turn.
    reason = "max_steps"
    while steps < max_steps:
        remaining = max_response_tokens - len(mask)
        if remaining < 1:
            reason = "max_tokens"
            break
        if record_boundaries:
            boundaries.append(
                {
                    "offset": len(mask),
                    "completed_steps": steps,
                    "messages": json_copy(messages, max_bytes=2**31),
                    "snapshot": json_copy(await asyncio.wait_for(env.snapshot(), step_timeout_s), max_snapshot_bytes),
                }
            )
        params = dict(sampling_params)
        params["max_tokens"] = min(max_action_tokens, remaining)
        if params.get("seed") is not None:
            params["seed"] = (params["seed"] + steps - completed_steps) % (2**31 - 1)
        output = await generate(request_id=request_id, prompt_ids=tokens, sampling_params=params)
        if output.stop_reason == "aborted" or not output.token_ids:
            raise RuntimeError("Incomplete tool generation; refusing a partial rollout group")
        if len(output.token_ids) > params["max_tokens"]:
            raise RuntimeError("Tool generation exceeded its requested token budget")
        version = output.extra_fields.get("global_steps")
        if expected_version is not None and version is not None and version != expected_version:
            raise RuntimeError(f"Stale teacher rollout: expected {expected_version}, received {version}")
        if version is not None:
            versions.append(version)
        merged = builder.merge_assistant_tokens(tokens, output.token_ids)
        if merged.removed_prefix_token_count or merged.token_ids[: len(tokens)] != tokens:
            raise ValueError("SCOUT tool mode requires append-only assistant token boundaries")
        mask, _ = builder.align_response_metadata(merged, mask)
        tokens = merged.token_ids
        if len(mask) > max_response_tokens:
            raise ValueError("Assistant boundary tokens exceed the response budget")
        text = tokenizer.decode(output.token_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text})
        num_turns += 1
        eos = getattr(tokenizer, "eos_token_id", None)
        eos_ids = eos if isinstance(eos, list) else [eos]
        # This verl version maps vLLM's stop AND length finish reasons to
        # 'completed'. At the cap, a missing EOS therefore means truncation.
        hit_token_cap = len(output.token_ids) == params["max_tokens"] and output.token_ids[-1] not in eos_ids
        if output.stop_reason in ("length", "max_tokens") or hit_token_cap:
            reason = "truncated_action"
            break
        action = parse_action(text)
        transition = await asyncio.wait_for(env.step(action), step_timeout_s)
        if not math.isfinite(transition.reward) or not isinstance(transition.observation, str):
            raise ValueError("Environment must return finite rewards and a text observation")
        reward += transition.reward
        invalid_actions += int(action is None or transition.info.get("invalid_action", False))
        steps += 1
        if transition.done:
            reason = "done"
            break
        if steps == max_steps:
            break
        updated = messages + [observation_message(transition.observation, max_steps - steps)]
        merged = builder.merge_context_tokens(messages, updated, tokens)
        # Do not rewrite sampled tokens or move an earlier snapshot boundary.
        if merged.removed_prefix_token_count or merged.token_ids[: len(tokens)] != tokens:
            raise ValueError("SCOUT tool mode requires append-only context token boundaries")
        next_mask, _ = builder.align_response_metadata(merged, mask)
        if len(next_mask) >= max_response_tokens:
            # Keep the executed action, but never retain half a tool observation.
            reason = "observation_overflow"
            break
        tokens, mask, messages = merged.token_ids, next_mask, updated
        num_turns += 1
    if not mask or not any(mask):
        raise ValueError("Tool episode contains no trainable action tokens")
    extra = {}
    if versions:
        if min(versions) != max(versions):
            raise RuntimeError("Policy weights changed within a synchronous tool episode")
        extra = {"min_global_steps": min(versions), "max_global_steps": max(versions)}
    return ToolEpisode(
        initial, tokens[len(initial) :], mask, boundaries, reward, steps, invalid_actions, reason, num_turns, extra
    )


def select_tool_boundary(boundaries, response_ids, response_mask, ratio, source="student"):
    """Select a completed observation boundary, never an arbitrary token prefix."""
    if not boundaries or not 0 <= ratio < 1 or source not in ("student", "initial"):
        raise ValueError("Missing tool snapshots or invalid prefix selection")
    if len(response_ids) != len(response_mask):
        raise ValueError("Tool response and loss mask are misaligned")
    offsets = [point["offset"] for point in boundaries]
    if offsets[0] != 0 or offsets != sorted(set(offsets)):
        raise ValueError("Tool boundaries must start at zero and increase strictly")
    for point in boundaries:
        offset = point["offset"]
        if not 0 <= offset < len(response_mask) or response_mask[offset] != 1:
            raise ValueError("Tool boundary must immediately precede a sampled action")
        if offset and response_mask[offset - 1] != 0:
            raise ValueError("Tool boundary must follow a complete environment observation")
    target = 0 if source == "initial" else math.floor(len(response_ids) * ratio)
    return json_copy(min(boundaries, key=lambda point: abs(point["offset"] - target)), max_bytes=2**31)


def tool_continuation_fields(context, episode):
    """Pack only the teacher continuation; prefix and observation tokens carry no loss."""
    import torch

    if episode.prompt_ids != context.teacher_input:
        raise ValueError("Teacher must continue the exact saved student token prefix")
    row = continuation_fields(context, episode.response_ids, episode.reward)
    mask = torch.tensor(episode.response_mask, dtype=torch.long)
    if mask.numel() != len(episode.response_ids) or not mask.bool().any():
        raise ValueError("Tool continuation has no aligned action mask")
    row["response_mask"] = mask
    row["loss_mask"] = mask.clone()
    row["rm_scores"].zero_()
    row["rm_scores"][mask.nonzero()[-1].item()] = episode.reward
    row["num_turns"] = episode.num_turns
    row["extra_info"] = dict(context.extra_info) | {
        "scout_tool_end_reason": episode.end_reason,
        "scout_tool_invalid_actions": episode.invalid_actions,
    }
    return row
