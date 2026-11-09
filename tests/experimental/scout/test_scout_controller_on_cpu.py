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
"""Controller contract tests with fake GPU boundaries and real batch transforms."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from verl.experimental.scout.teacher import TeacherRuntime
from verl.experimental.scout.trainer import SCOUTTrainer
from verl.experimental.scout.utils import PrefixContext, continuation_fields, pack_teacher_rows
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


@pytest.fixture
def config(tmp_path):
    directory = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="scout_trainer")
    with open_dict(config):
        config.trainer.default_local_dir = str(tmp_path)
    return config


def controller(config, step=10):
    # Construct the controller without launching models, then test its real hooks.
    obj = object.__new__(SCOUTTrainer)
    obj.config = config
    obj.global_steps = step
    obj.total_training_steps = 100
    obj.tokenizer = SimpleNamespace(eos_token_id=99)
    obj.teacher = SimpleNamespace(
        updates=step // 10,
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(model=SimpleNamespace(path="teacher"))),
        update_from_prefixes=Mock(return_value={"scout/teacher/updates": 1}),
        actor_rollout_wg=SimpleNamespace(save_checkpoint=Mock()),
    )
    return obj


def test_teacher_update_uses_current_batch_and_skips_padding(config, monkeypatch):
    obj = controller(config)
    context = PrefixContext("source", [1, 2], [], "math", {"ground_truth": "42"}, {})
    row = continuation_fields(context, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 99], 1)
    batch = SimpleNamespace(
        keys=["trajectory", "padding"],
        partition_id="train",
        tags=[{"prompt_len": 2, "response_len": 11}, {"is_padding": True}],
    )
    monkeypatch.setattr(PPOTrainerSync, "_step_once", lambda *args: batch)
    monkeypatch.setattr(
        "verl.experimental.scout.trainer.tq.kv_batch_get", lambda **kwargs: pack_teacher_rows([row, row])
    )
    result = obj._step_once({}, {}, 1)
    assert result is batch
    contexts, step = obj.teacher.update_from_prefixes.call_args.args
    assert step == 10 and len(contexts) == 1
    assert contexts[0].teacher_input == [1, 2, 3]
    assert contexts[0].reward_model["ground_truth"] == "42"


def test_intermediate_step_keeps_teacher_frozen(config, monkeypatch):
    obj = controller(config, step=9)
    batch = object()
    monkeypatch.setattr(PPOTrainerSync, "_step_once", lambda *args: batch)
    assert obj._step_once({}, {}, 1) is batch
    obj.teacher.update_from_prefixes.assert_not_called()


def test_checkpoint_contains_teacher_before_parent_publishes_latest(config, monkeypatch):
    obj = controller(config)
    events = []
    obj.teacher.actor_rollout_wg.save_checkpoint.side_effect = lambda *a, **kw: events.append("teacher")

    def parent_save(self):
        state = json.loads((self._step_folder() / "scout_state.json").read_text())
        assert state["teacher_updates"] == 1
        events.append("student_and_latest_pointer")

    monkeypatch.setattr(PPOTrainerSync, "_save_checkpoint", parent_save)
    obj._save_checkpoint()
    assert events == ["teacher", "student_and_latest_pointer"]
    obj._validate_resume_state(obj._state())
    stale = obj._state() | {"teacher_updates": 0}
    with pytest.raises(ValueError, match="update count"):
        obj._validate_resume_state(stale)
    changed_schedule = obj._state() | {"curriculum_total_steps": 200}
    with pytest.raises(ValueError, match="curriculum_total_steps"):
        obj._validate_resume_state(changed_schedule)


def test_initialization_refreshes_teacher_after_student(config, monkeypatch):
    obj = controller(config)
    events = []
    monkeypatch.setattr(PPOTrainerSync, "on_init_end", lambda self: events.append("student_sync"))
    obj.teacher.sync = lambda: events.append("teacher_sync")
    obj.on_init_end()
    assert events == ["student_sync", "teacher_sync"]


def test_sync_waits_for_inference_commit():
    obj = object.__new__(TeacherRuntime)
    obj.updates = 3
    obj.checkpoint_manager = Mock()
    obj.sync()
    calls = obj.checkpoint_manager.mock_calls
    assert calls[0].args == (3,)
    assert calls[0][0] == "update_weights"
    assert calls[1][0] == "wait_for_weight_update"


def test_generation_scores_full_answer_but_trains_continuation(config):
    obj = object.__new__(TeacherRuntime)
    obj.config = config
    obj.updates = 2
    obj.client = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(token_ids=[30, 99], stop_reason="completed", extra_fields={"global_steps": 2})
        )
    )
    reward = AsyncMock(return_value={"reward_score": 1.0})
    obj.reward_handles = [SimpleNamespace(compute_score=SimpleNamespace(remote=reward))]
    context = PrefixContext("prefix", [10, 11], [20, 21], "math", {"ground_truth": "42"}, {})
    rows = obj._generate_and_score([context], student_step=20)
    assert len(rows) == 8
    assert all(row["responses"].tolist() == [30, 99] for row in rows)
    assert all(row["uid"] == "prefix" for row in rows)
    for call in reward.call_args_list:
        assert call.args[0].batch["responses"].tolist() == [[20, 21, 30, 99]]
    seeds = {call.kwargs["sampling_params"]["seed"] for call in obj.client.generate.call_args_list}
    assert len(seeds) == 8


def test_stale_teacher_version_fails_before_training(config):
    obj = object.__new__(TeacherRuntime)
    obj.config = config
    obj.updates = 2
    obj.client = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(token_ids=[30, 99], stop_reason="completed", extra_fields={"global_steps": 1})
        )
    )
    obj.reward_handles = [Mock()]
    context = PrefixContext("prefix", [10], [20], "math", {"ground_truth": "42"}, {})
    with pytest.raises(RuntimeError, match="Stale teacher"):
        obj._generate_and_score([context], student_step=20)
