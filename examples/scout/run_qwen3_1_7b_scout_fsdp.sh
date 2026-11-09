#!/usr/bin/env bash
# SCOUT | Qwen3 text models | FSDP2 | vLLM | synchronous V1 trainer
set -euo pipefail

: "${TRAIN_FILE:?Set TRAIN_FILE to a verl-format parquet file}"
: "${VAL_FILE:?Set VAL_FILE to a verl-format validation parquet file}"

LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi

"${LAUNCH[@]}" -m verl.experimental.scout.main \
    "data.train_files=['${TRAIN_FILE}']" \
    "data.val_files=['${VAL_FILE}']" \
    "actor_rollout_ref.model.path=${STUDENT_MODEL:-Qwen/Qwen3-1.7B}" \
    "distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL:-Qwen/Qwen3-4B-Instruct-2507}" \
    "${RAY[@]}" \
    "$@"
