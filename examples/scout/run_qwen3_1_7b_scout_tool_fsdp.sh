#!/usr/bin/env bash
# SCOUT tool extension | Qwen3 action-only policies | FSDP2 | vLLM
set -euo pipefail

export TRAIN_FILE="${TRAIN_FILE:-data/scout_counter/train.parquet}"
export VAL_FILE="${VAL_FILE:-data/scout_counter/val.parquet}"
for dataset in "$TRAIN_FILE" "$VAL_FILE"; do
    if [ ! -f "$dataset" ]; then
        echo "Missing $dataset. Run: python examples/scout/prepare_tool_counter.py" >&2
        exit 1
    fi
done

bash examples/scout/run_qwen3_1_7b_scout_fsdp.sh \
    --config-name scout_tool_trainer \
    "$@"
