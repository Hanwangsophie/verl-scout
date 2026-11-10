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
"""Real environment/token/mask/GRPO paths; only model generation and GPU workers are fake."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.scout.config import validate_scout
from verl.experimental.scout.tool_agent_loop import ScoutToolAgentLoop
from verl.experimental.scout.tool_env import CounterEnvironment, json_copy
from verl.experimental.scout.tool_rollout import (
    ToolPrefixContext,
    observation_message,
    parse_action,
    run_tool_episode,
    select_tool_boundary,
    tool_continuation_fields,
)
from verl.experimental.scout.tool_teacher import ToolTeacherRuntime
from verl.experimental.scout.tool_trainer import ToolSCOUTTrainer
from verl.experimental.scout.utils import pack_teacher_rows
from verl.trainer.distillation.losses import distillation_loss
from verl.utils.import_utils import resolve_config_path
from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder
from verl.workers.config import DistillationLossConfig


class ChatTokenizer:
    """Tiny deterministic ChatML tokenizer; does not download or mimic model weights."""

    eos_token_id = 1000

    def encode(self, text, **kwargs):
        chunks = text.split("<|im_end|>")
        result = []
        for i, chunk in enumerate(chunks):
            if i:
                result.append(self.eos_token_id)
            result.extend(map(ord, chunk))
        return result

    def decode(self, ids, skip_special_tokens=False):
        return "".join(("" if skip_special_tokens else "<|im_end|>") if token == 1000 else chr(token) for token in ids)

    def convert_tokens_to_ids(self, token):
        return self.eos_token_id if token == "<|im_end|>" else None

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text


@pytest.fixture
def config():
    directory = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        return compose(config_name="scout_tool_trainer")


def output(name, **arguments):
    text = json.dumps({"name": name, "arguments": arguments})
    return SimpleNamespace(
        token_ids=ChatTokenizer().encode(text) + [1000], stop_reason="completed", extra_fields={"global_steps": 2}
    )


async def student_trace():
    env = CounterEnvironment({})
    observation = await env.reset({"start": 0, "target": 3})
    env.state["private"] = "runtime_only_secret"
    tokenizer = ChatTokenizer()
    builder = QwenContinuousTokenBuilder(tokenizer)
    messages = [observation_message(observation, 6)]
    # Student makes a wrong move, then an unsuccessful recovery. Teacher branches
    # must NOT receive the student's future actions or observations.
    generate = AsyncMock(side_effect=[output("add", amount=-1), output("add", amount=3), output("finish")])
    trace = await run_tool_episode(
        env=env,
        generate=generate,
        tokenizer=tokenizer,
        builder=builder,
        input_ids=builder.build_initial_tokens(messages),
        messages=messages,
        sampling_params={},
        request_id="student",
        max_steps=6,
        max_action_tokens=128,
        max_response_tokens=2048,
        record_boundaries=True,
    )
    await env.close()
    return trace


def middle_context(trace):
    point = trace.boundaries[1]
    return ToolPrefixContext(
        "prefix", trace.prompt_ids, trace.response_ids[: point["offset"]], "counter", {}, {}, point
    )


def teacher(config):
    runtime = object.__new__(ToolTeacherRuntime)
    runtime.config = config
    runtime.updates = 2
    runtime.set_student_tokenizer(ChatTokenizer())
    return runtime


def test_student_snapshots_align_with_exact_tokens_and_context_mask():
    trace = asyncio.run(student_trace())
    assert trace.reward == 0  # finishes at 2, not 3
    assert trace.num_turns == 6
    assert len(trace.boundaries) == 3
    point = trace.boundaries[1]
    assert point["snapshot"]["state"]["value"] == -1
    assert point["completed_steps"] == 1
    assert trace.response_mask[point["offset"] - 1] == 0
    assert trace.response_mask[point["offset"]] == 1
    prefix = ChatTokenizer().decode(trace.prompt_ids + trace.response_ids[: point["offset"]])
    assert '"value": -1' in prefix
    assert '"value": 2' not in prefix
    assert "runtime_only_secret" not in prefix
    # Qwen's synthesized newline after EOS is context, not an action label.
    first_eos = trace.response_ids.index(1000)
    assert trace.response_mask[first_eos] == 1
    assert trace.response_mask[first_eos + 1] == 0


def test_teacher_branches_restore_same_state_but_execute_fresh_actions(config, monkeypatch):
    context = middle_context(asyncio.run(student_trace()))
    config.scout.teacher_rollouts_per_prefix = 2
    runtime = teacher(config)
    seen, closed, counters = {}, [], {}
    original_close = CounterEnvironment.close

    async def close(env):
        closed.append(env.state["value"])
        await original_close(env)

    monkeypatch.setattr(CounterEnvironment, "close", close)

    async def generate(request_id, prompt_ids, sampling_params):
        branch = int(request_id.split("-")[3])
        seen.setdefault(branch, []).append(list(prompt_ids))
        turn = counters.get(branch, 0)
        counters[branch] = turn + 1
        if branch == 0:
            return [output("add", amount=3), output("add", amount=1), output("finish")][turn]
        return [output("add", amount=-2), output("finish")][turn]

    runtime.client = SimpleNamespace(generate=generate)
    rows = runtime._generate_and_score([context], student_step=20)
    assert [row["rm_scores"].sum().item() for row in rows] == [1, 0]
    assert [row["uid"] for row in rows] == ["prefix", "prefix"]
    assert seen[0][0] == seen[1][0] == context.teacher_input
    assert '"value": -3' in ChatTokenizer().decode(seen[1][1])
    assert '"value": 2' not in ChatTokenizer().decode(seen[1][1])
    assert sorted(closed) == [-3, 3]
    assert context.boundary["snapshot"]["state"]["value"] == -1
    for row in rows:
        assert row["prompts"].tolist() == context.teacher_input
        assert (row["loss_mask"] == 0).any()
        assert row["loss_mask"][-1] == 1
        assert row["rm_scores"][:-1].sum() == 0


def test_boundary_selection_never_cuts_action_or_observation():
    trace = asyncio.run(student_trace())
    for ratio in (0, 0.25, 0.5, 0.9):
        selected = select_tool_boundary(trace.boundaries, trace.response_ids, trace.response_mask, ratio)
        assert selected in trace.boundaries
    initial = select_tool_boundary(trace.boundaries, trace.response_ids, trace.response_mask, 0.9, "initial")
    assert initial["offset"] == 0
    broken = json_copy(trace.boundaries)
    broken[1]["offset"] -= 1
    with pytest.raises(ValueError, match="precede a sampled action"):
        select_tool_boundary(broken, trace.response_ids, trace.response_mask, 0.9)


@pytest.mark.parametrize("text", ['{"name":"add","arguments":[]}', "I will use a tool", '{"name":"add"}', "null"])
def test_invalid_action_output_is_rejected_without_retokenizing(text):
    assert parse_action(text) is None


def test_snapshot_rejects_nonfinite_values_and_large_payloads():
    with pytest.raises(ValueError):
        json_copy({"rng": float("nan")})
    with pytest.raises(ValueError, match="max_snapshot_bytes"):
        json_copy({"state": "too large"}, max_bytes=4)


def test_truncated_action_is_not_executed_even_with_completed_stop_reason():
    async def scenario():
        env = CounterEnvironment({})
        await env.reset({"start": 0, "target": 1})
        tokenizer = ChatTokenizer()
        text = json.dumps({"name": "add", "arguments": {"amount": 1}})
        generated = SimpleNamespace(token_ids=tokenizer.encode(text), stop_reason="completed", extra_fields={})
        trace = await run_tool_episode(
            env=env,
            generate=AsyncMock(return_value=generated),
            tokenizer=tokenizer,
            builder=QwenContinuousTokenBuilder(tokenizer),
            input_ids=[10],
            messages=[],
            sampling_params={},
            request_id="limited",
            max_steps=1,
            max_action_tokens=len(text),
            max_response_tokens=1024,
        )
        assert trace.end_reason == "truncated_action"
        assert env.state["calls"] == 0 and trace.reward == 0

    asyncio.run(scenario())


def test_teacher_rejects_stale_version_and_closes_all_branches(config, monkeypatch):
    config.scout.teacher_rollouts_per_prefix = 2
    runtime = teacher(config)
    stale = output("finish")
    stale.extra_fields["global_steps"] = 1
    runtime.client = SimpleNamespace(generate=AsyncMock(return_value=stale))
    closed = AsyncMock()
    monkeypatch.setattr(CounterEnvironment, "close", closed)
    with pytest.raises(RuntimeError, match="Stale teacher"):
        runtime._generate_and_score([middle_context(asyncio.run(student_trace()))], student_step=20)
    # student_trace also closes its environment.
    assert closed.await_count == 3


def test_tool_config_and_hydra_registry(config):
    validate_scout(config)
    registry = OmegaConf.load(resolve_config_path(config.actor_rollout_ref.rollout.agent.agent_loop_config_path))
    assert registry[0]._target_ == "verl.experimental.scout.tool_agent_loop.ScoutToolAgentLoop"
    config.data.apply_chat_template_kwargs.enable_thinking = True
    with pytest.raises(ValueError, match="enable_thinking"):
        validate_scout(config)


def test_student_agent_output_survives_tensordict_roundtrip(config):
    loop = object.__new__(ScoutToolAgentLoop)
    loop.config, loop.rollout_config = config, config.actor_rollout_ref.rollout
    loop.tokenizer, loop.processor = ChatTokenizer(), None
    loop.continuous_token_builder = QwenContinuousTokenBuilder(loop.tokenizer)
    loop.server_manager = SimpleNamespace(generate=AsyncMock(side_effect=[output("add", amount=1), output("finish")]))
    result = asyncio.run(
        loop.run(
            {},
            raw_prompt=[{"role": "user", "content": "Reach 1"}],
            extra_info={"scout_tool_task": {"start": 0, "target": 1}},
        )
    )
    assert result.reward_score == 1
    row = result.as_dict()
    row.update(data_source="counter", reward_model={}, extra_info={"scout_tool_task": {"start": 0, "target": 1}})
    batch = pack_teacher_rows([row])
    assert batch["extra_fields"][0]["scout_tool_boundaries"][1]["snapshot"]["state"]["value"] == 1
    assert batch["rm_scores"][0].sum() == 1


def test_tool_trainer_reads_snapshots_and_validates_checkpoint(config, monkeypatch):
    trace = asyncio.run(student_trace())
    context = middle_context(trace)
    row = tool_continuation_fields(
        ToolPrefixContext("original", trace.prompt_ids, [], "counter", {}, {}, trace.boundaries[0]), trace
    )
    row["extra_fields"] = {"scout_tool_boundaries": trace.boundaries}
    data = pack_teacher_rows([row])
    monkeypatch.setattr("verl.experimental.scout.tool_trainer.tq.kv_batch_get", lambda **kwargs: data)
    controller = object.__new__(ToolSCOUTTrainer)
    controller.config, controller.global_steps, controller.total_training_steps = config, 10, 100
    controller.teacher = SimpleNamespace(updates=1, config=config)
    batch = SimpleNamespace(
        keys=["trajectory"],
        partition_id="train",
        tags=[{"prompt_len": len(trace.prompt_ids), "response_len": len(trace.response_ids)}],
    )
    ratio = context.boundary["offset"] / len(trace.response_ids)
    selected = controller._get_prefix_contexts(batch, ratio)[0]
    assert selected.teacher_input == context.teacher_input
    assert selected.boundary["snapshot"] == context.boundary["snapshot"]
    state = controller._state()
    controller._validate_resume_state(state)
    state["tool_config"]["max_steps"] += 1
    with pytest.raises(ValueError, match="tool_config"):
        controller._validate_resume_state(state)


def test_actual_opd_loss_has_zero_observation_gradient():
    current = torch.tensor([-2.0, -3.0, -2.0, 0.0], requires_grad=True)
    data = TensorDict(
        {
            "prompts": torch.nested.as_nested_tensor([torch.tensor([10])], layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor([torch.tensor([20, 21, 22])], layout=torch.jagged),
            "response_mask": torch.tensor([[1, 0, 1]]),
            "old_log_probs": torch.tensor([[-2.0, -3.0, -2.0]]),
            "teacher_logprobs": torch.nested.as_nested_tensor(
                [torch.tensor([[-1.0], [-9.0], [-3.0], [0.0]])], layout=torch.jagged
            ),
        },
        batch_size=1,
    )
    for key, value in (("dp_size", 1), ("batch_num_tokens", 2), ("global_batch_size", 1)):
        data.set_non_tensor(key, value)
    actor = SimpleNamespace(global_batch_info={}, loss_agg_mode="token-mean", loss_scale_factor=None)
    cfg = SimpleNamespace(distillation_loss=DistillationLossConfig(loss_mode="k1", use_old_log_probs=True))
    loss, _ = distillation_loss(actor, cfg, {"log_probs": current}, data)
    loss.backward()
    assert current.grad[0] < 0 < current.grad[2]
    assert current.grad[1] == 0 and current.grad[3] == 0


def test_teacher_update_masks_grpo_advantages_and_syncs(config, monkeypatch):
    context = middle_context(asyncio.run(student_trace()))
    runtime = teacher(config)
    runtime.config.scout.teacher_rollouts_per_prefix = 2
    runtime.use_reference_policy = True
    # Exercise the actual grouping/update function with tensor rows containing
    # an observation between two actions; the environment execution is tested above.
    trace = SimpleNamespace(
        prompt_ids=context.teacher_input,
        response_ids=[1, 2, 3],
        response_mask=[1, 0, 1],
        reward=0,
        steps=3,
        num_turns=4,
        invalid_actions=0,
        end_reason="done",
    )
    rows = []
    for score in (0, 1):
        trace.reward = score
        rows.append(tool_continuation_fields(context, trace))
    runtime._generate_and_score = Mock(return_value=rows)
    meta = SimpleNamespace(extra_info={})
    put = Mock(return_value=meta)
    clear = Mock()
    monkeypatch.setattr("verl.experimental.scout.teacher.tq.kv_batch_put", put)
    monkeypatch.setattr("verl.experimental.scout.teacher.tq.kv_clear", clear)
    runtime._balance_batch = lambda batch, metrics: batch
    runtime._compute_old_log_prob = Mock(return_value=meta)
    runtime._compute_ref_log_prob = Mock(return_value=meta)
    runtime._update_actor, runtime.sync = Mock(), Mock()
    metrics = runtime.update_from_prefixes([context], 20)
    stored = put.call_args.kwargs["fields"]
    assert stored["advantages"][0][0] < 0 < stored["advantages"][1][0]
    assert all(stored["advantages"][i][1] == 0 for i in range(2))
    assert runtime.updates == 3 and metrics["scout/teacher/action_tokens"] == 4
    runtime._compute_ref_log_prob.assert_called_once()
    runtime._update_actor.assert_called_once()
    runtime.sync.assert_called_once()
    clear.assert_called_once()
