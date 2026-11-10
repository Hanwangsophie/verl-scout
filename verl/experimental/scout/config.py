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
"""Configuration validation and independent teacher optimizer configuration."""

from omegaconf import OmegaConf, open_dict


def validate_scout(config):
    """Reject options that break the synchronous, single-version SCOUT contract."""
    required = {
        "trainer.use_v1": True,
        "trainer.v1.trainer_mode": "sync",
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "algorithm.use_kl_in_reward": False,
        "algorithm.adv_estimator": "grpo",
        "reward.reward_model.enable": False,
        "distillation.enabled": True,
        "distillation.distillation_loss.loss_mode": "k1",
        "distillation.distillation_loss.use_policy_gradient": True,
        "distillation.distillation_loss.use_old_log_probs": True,
        "distillation.distillation_loss.use_task_rewards": False,
        "actor_rollout_ref.hybrid_engine": True,
        "trainer.critic_warmup": 0,
    }
    for path, expected in required.items():
        if OmegaConf.select(config, path) != expected:
            raise ValueError(f"This synchronous SCOUT recipe requires {path}={expected!r}")
    if config.actor_rollout_ref.actor.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("This SCOUT recipe supports dense FSDP/FSDP2 models")
    if config.actor_rollout_ref.actor.ppo_mini_batch_size != config.data.train_batch_size:
        raise ValueError("Use one student mini-batch per step: ppo_mini_batch_size == train_batch_size")
    if config.data.get("gen_batch_size") not in (None, config.data.train_batch_size):
        raise ValueError("gen_batch_size must equal train_batch_size to avoid carrying stale teacher scores")
    sampler = config.trainer.v1.sampler
    if sampler.get("sync_refill_failed_groups", False) or sampler.custom_sampler.path is not None:
        raise ValueError("SCOUT requires the built-in synchronous sampler without failed-group refill")
    if config.trainer.v1.get("sync", {}).get("parameter_sync_step", 1) != 1:
        raise ValueError("SCOUT requires one student update per synchronous step")
    mode = config.scout.get("mode", "text")
    if mode not in ("text", "tool"):
        raise ValueError("scout.mode must be text or tool")
    agent = "single_turn_agent" if mode == "text" else "scout_tool_agent"
    if config.actor_rollout_ref.rollout.agent.default_agent_loop != agent:
        raise ValueError(f"SCOUT {mode} mode requires {agent}")
    if mode == "tool":
        validate_tool_config(config)
    if OmegaConf.select(config, "algorithm.filter_groups.enable", default=False):
        raise ValueError("Disable group filtering for the SCOUT reproduction")
    if config.trainer.default_hdfs_dir is not None:
        raise ValueError("Use a shared filesystem checkpoint directory; HDFS SCOUT checkpoints are unsupported")
    if config.actor_rollout_ref.actor.checkpoint.get("async_save", False):
        raise ValueError("SCOUT needs synchronous student/teacher checkpoints")
    model = config.actor_rollout_ref.model
    if (model.get("lora") or {}).get("rank", 0) or model.get("lora_rank", 0) or model.get("lora_adapter_path"):
        raise ValueError("SCOUT weight synchronization currently supports full-parameter training only")
    if len(config.distillation.teacher_models) != 1:
        raise ValueError("SCOUT currently supports exactly one teacher")
    teacher = next(iter(config.distillation.teacher_models.values()))
    if teacher.inference.name != "vllm":
        raise ValueError("SCOUT teacher inference requires vLLM")
    scout = config.scout
    if scout.teacher_update_interval < 1 or scout.teacher_rollouts_per_prefix < 2:
        raise ValueError("Teacher interval must be positive and GRPO group size must be >= 2")
    if min(scout.teacher_nnodes, scout.teacher_gpus_per_node, scout.max_concurrent_requests) < 1:
        raise ValueError("Teacher resources and concurrency must be positive")
    if scout.teacher_lr <= 0 or scout.teacher_kl_coef < 0 or scout.teacher_max_response_length < 1:
        raise ValueError("Require a positive teacher LR/response length and nonnegative KL coefficient")
    if not 0 <= scout.prefix_ratio_start <= scout.prefix_ratio_end < 1:
        raise ValueError("Require 0 <= prefix_ratio_start <= prefix_ratio_end < 1")
    if (
        config.data.train_batch_size
        * scout.teacher_rollouts_per_prefix
        % (scout.teacher_nnodes * scout.teacher_gpus_per_node)
    ):
        raise ValueError("Teacher response batch must be divisible by teacher training world size")
    if teacher.inference.checkpoint_engine.backend != "nccl":
        raise ValueError("Teacher inference checkpoint_engine.backend must be nccl")
    max_len = teacher.inference.max_model_len
    required_len = (
        config.data.max_prompt_length
        + (
            config.data.max_response_length
            if mode == "tool"
            else int(config.data.max_response_length * scout.prefix_ratio_end)
        )
        + scout.teacher_max_response_length
    )
    if max_len is None or max_len < required_len:
        raise ValueError(f"Teacher max_model_len must be >= {required_len}; do not truncate student prefixes")
    for mode in ("load_contents", "save_contents"):
        contents = config.actor_rollout_ref.actor.checkpoint[mode]
        if not {"model", "optimizer", "extra"}.issubset(contents):
            raise ValueError(f"SCOUT checkpoint {mode} must include model, optimizer, extra")


def validate_tool_config(config):
    """Reject settings outside the initial action-only, restorable tool contract."""
    tool = config.scout.get("tool")
    if tool is None:
        raise ValueError("Use --config-name scout_tool_trainer for tool mode")
    if not config.actor_rollout_ref.rollout.agent.agent_loop_config_path:
        raise ValueError("Tool mode requires the scout_tool_agent registry config")
    if not config.actor_rollout_ref.rollout.multi_turn.enable:
        raise ValueError("SCOUT tool mode requires multi_turn.enable=true")
    if config.actor_rollout_ref.rollout.calculate_log_probs:
        raise ValueError("SCOUT tool mode recomputes old action log probabilities in training")
    if config.actor_rollout_ref.rollout.get("topk_log_probs", 0):
        raise ValueError("SCOUT tool mode does not store sampler top-k heads")
    if tool.hf_model_type != "qwen3" or config.data.apply_chat_template_kwargs.get("enable_thinking") is not False:
        raise ValueError("The first tool recipe requires Qwen3 text models with enable_thinking=false")
    if min(tool.max_steps, tool.max_action_tokens, tool.max_snapshot_bytes) < 1:
        raise ValueError("Tool step, action-token and snapshot budgets must be positive")
    if tool.max_action_tokens > min(config.data.max_response_length, config.scout.teacher_max_response_length):
        raise ValueError("Action token budget must fit both student and teacher response budgets")
    if tool.prefix_source not in ("student", "initial"):
        raise ValueError("Tool prefix_source must be student or initial")
    if tool.step_timeout_s <= 0:
        raise ValueError("Tool step timeout must be positive")
    if not isinstance(tool.environment_class, str) or "." not in tool.environment_class:
        raise ValueError("Tool mode requires a SnapshotToolEnvironment class path")


def make_teacher_config(config, total_student_steps):
    """Build the independently optimized teacher and frozen-reference worker config."""
    # Resolve the student interpolations before overriding them: do not accidentally
    # reuse the student's optimizer step count, model path, or tokenizer path.
    result = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    teacher = next(iter(config.distillation.teacher_models.values()))
    with open_dict(result):
        result.distillation.enabled = False
        result.actor_rollout_ref.model.path = teacher.model_path
        result.actor_rollout_ref.model.tokenizer_path = teacher.model_path
        result.actor_rollout_ref.actor.optim.lr = config.scout.teacher_lr
        result.actor_rollout_ref.actor.optim.total_training_steps = max(
            1, total_student_steps // config.scout.teacher_update_interval
        )
        result.actor_rollout_ref.actor.optim.lr_scheduler_type = "constant"
        result.actor_rollout_ref.actor.ppo_epochs = 1
        result.actor_rollout_ref.actor.use_kl_loss = config.scout.teacher_kl_coef > 0
        result.actor_rollout_ref.actor.kl_loss_coef = config.scout.teacher_kl_coef
        result.actor_rollout_ref.actor.kl_loss_type = "low_var_kl"
        result.actor_rollout_ref.actor.entropy_coeff = 0.0
        result.actor_rollout_ref.actor.loss_agg_mode = "seq-mean-token-mean"
        result.actor_rollout_ref.rollout.n = config.scout.teacher_rollouts_per_prefix
        result.actor_rollout_ref.rollout.calculate_log_probs = False
        result.actor_rollout_ref.rollout.checkpoint_engine.backend = "nccl"
        result.actor_rollout_ref.ref.strategy = result.actor_rollout_ref.actor.strategy
        result.algorithm.rollout_correction = None
        result.trainer.nnodes = config.scout.teacher_nnodes
        result.trainer.n_gpus_per_node = config.scout.teacher_gpus_per_node
    return result
