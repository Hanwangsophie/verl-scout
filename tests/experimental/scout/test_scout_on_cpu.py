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
"""SCOUT correctness tests; no model downloads, CUDA, or live Ray cluster."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.scout.config import make_teacher_config, validate_scout
from verl.experimental.scout.utils import (
    PrefixContext,
    continuation_fields,
    pack_teacher_rows,
    prefix_ratio,
    response_for_verifier,
    select_prefix,
    validate_tokenizers,
)
from verl.trainer.distillation.losses import compute_distillation_loss_reverse_kl_estimator
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.workers.config import DistillationLossConfig


@pytest.fixture
def config():
    directory = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        return compose(config_name="scout_trainer")


def context(uid="first-prefix"):
    return PrefixContext(uid, [10, 11], [20, 21], "math", {"ground_truth": "42"}, {})


def test_prefix_excludes_eos_and_leaves_a_suffix():
    assert select_prefix([4, 5, 6, 99, 0, 0], 0.9, {99}) == [4, 5]
    assert select_prefix([99], 0.9, {99}) == []
    assert select_prefix([4], 0.9, {99}) == []
    assert select_prefix([4, 5], 0, {99}) == []
    with pytest.raises(ValueError):
        select_prefix([4, 5], 1.0, {99})


def test_curriculum_has_explicit_bounded_endpoints():
    assert prefix_ratio(1, 101, 0.1, 0.9) == pytest.approx(0.1)
    assert prefix_ratio(51, 101, 0.1, 0.9) == pytest.approx(0.5)
    assert prefix_ratio(101, 101, 0.1, 0.9) == pytest.approx(0.9)
    assert prefix_ratio(999, 101, 0.1, 0.9) == pytest.approx(0.9)


def test_reward_sees_prefix_but_loss_does_not():
    c = context()
    row = continuation_fields(c, [30, 31, 99], 1.0)
    assert row["prompts"].tolist() == [10, 11, 20, 21]
    assert row["responses"].tolist() == [30, 31, 99]
    assert row["loss_mask"].tolist() == [1, 1, 1]
    assert row["rm_scores"].tolist() == [0, 0, 1]
    assert response_for_verifier(c, [30, 31, 99]) == [20, 21, 30, 31, 99]


def test_empty_and_nonfinite_teacher_outputs_fail():
    with pytest.raises(ValueError, match="empty continuation"):
        continuation_fields(context(), [], 1)
    with pytest.raises(ValueError, match="non-finite"):
        continuation_fields(context(), [30], float("nan"))


def test_teacher_batch_remains_jagged_when_lengths_equal():
    rows = [continuation_fields(context(), [30, 99], score) for score in (0, 1)]
    batch = pack_teacher_rows(rows)
    assert batch["prompts"].is_nested
    assert batch["response_mask"].is_nested
    assert batch["input_ids"].offsets().tolist() == [0, 6, 12]


def test_grpo_groups_are_prefix_specific():
    # Prefix A yields both failure and success. Prefix B is all success, so it
    # has zero relative advantage even though the underlying question is shared.
    rewards = torch.tensor([[0.0], [1.0], [1.0], [1.0]])
    mask = torch.ones_like(rewards)
    advantage, _ = compute_grpo_outcome_advantage(rewards, mask, ["A", "A", "B", "B"])
    assert advantage[0] < 0 < advantage[1]
    torch.testing.assert_close(advantage[2:], torch.zeros(2, 1))


def test_tokenizer_equal_size_with_different_ids_is_rejected():
    a = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1}, eos_token_id=1)
    b = SimpleNamespace(get_vocab=lambda: {"b": 0, "a": 1}, eos_token_id=1)
    with pytest.raises(ValueError, match="token-to-ID"):
        validate_tokenizers(a, b)
    validate_tokenizers(a, a)


def _loss_data():
    # One prompt token + two response tokens. The model's first two positions
    # predict those response tokens; its last position is not used.
    return TensorDict(
        {
            "prompts": torch.nested.as_nested_tensor([torch.tensor([1])], layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor([torch.tensor([2, 3])], layout=torch.jagged),
            "response_mask": torch.ones(1, 2),
            "old_log_probs": torch.tensor([[-2.0, -3.0]]),
            "teacher_logprobs": torch.nested.as_nested_tensor(
                [torch.tensor([[-1.0], [-4.0], [0.0]])], layout=torch.jagged
            ),
        },
        batch_size=1,
    )


def test_frozen_advantage_does_not_change_when_current_policy_changes():
    cfg = SimpleNamespace(distillation_loss=DistillationLossConfig(loss_mode="k1", use_old_log_probs=True))
    data = _loss_data()
    first, _ = compute_distillation_loss_reverse_kl_estimator(None, cfg, {"log_probs": None}, data)
    changed, _ = compute_distillation_loss_reverse_kl_estimator(
        None, cfg, {"log_probs": torch.tensor([7.0, 8.0, 9.0])}, data
    )
    torch.testing.assert_close(first, torch.tensor([[-1.0, 1.0]]))
    torch.testing.assert_close(first, changed)
    assert not first.requires_grad
    # The frozen advantage still multiplies a differentiable current-policy ratio.
    current = data["old_log_probs"].clone().requires_grad_()
    loss = -(torch.exp(current - data["old_log_probs"]) * -first).mean()
    loss.backward()
    assert current.grad[0, 0] < 0 < current.grad[0, 1]


def test_existing_opd_default_still_uses_current_policy():
    cfg = SimpleNamespace(distillation_loss=DistillationLossConfig(loss_mode="k1"))
    loss, _ = compute_distillation_loss_reverse_kl_estimator(
        None, cfg, {"log_probs": torch.tensor([-5.0, -6.0, 0.0])}, _loss_data()
    )
    torch.testing.assert_close(loss, torch.tensor([[-4.0, -2.0]]))


def test_old_log_probs_option_rejects_other_objectives():
    with pytest.raises(ValueError, match="policy-gradient K1"):
        DistillationLossConfig(loss_mode="k3", use_old_log_probs=True)


def test_config_and_independent_teacher_state(config):
    validate_scout(config)
    teacher = make_teacher_config(config, total_student_steps=100)
    assert teacher.actor_rollout_ref.model.path == "Qwen/Qwen3-4B-Instruct-2507"
    assert teacher.actor_rollout_ref.model.tokenizer_path == teacher.actor_rollout_ref.model.path
    assert teacher.actor_rollout_ref.actor.optim.total_training_steps == 10
    assert teacher.actor_rollout_ref.rollout.n == 8
    assert teacher.actor_rollout_ref.actor.use_kl_loss
    assert not teacher.distillation.enabled
    assert config.distillation.enabled
    assert config.actor_rollout_ref.model.path == "Qwen/Qwen3-1.7B"


@pytest.mark.parametrize(
    "path,value",
    [
        ("trainer.v1.trainer_mode", "separate_async"),
        ("actor_rollout_ref.rollout.n", 8),
        ("scout.teacher_rollouts_per_prefix", 1),
        ("data.gen_batch_size", 32),
        ("actor_rollout_ref.model.lora_adapter_path", "adapter"),
        ("distillation.teacher_models.teacher_model.inference.max_model_len", 1024),
    ],
)
def test_unsupported_config_fails_before_allocating_gpus(config, path, value):
    OmegaConf.update(config, path, value)
    with pytest.raises(ValueError):
        validate_scout(config)
