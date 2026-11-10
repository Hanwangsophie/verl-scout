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
"""Generate a deterministic local tool sandbox dataset; no external downloads."""

import argparse
import random
from pathlib import Path

import pandas as pd


def make_rows(pairs, split):
    """Describe public tools and keep runtime task initialization in extra_info."""
    return [
        {
            "data_source": "scout_counter",
            "agent_name": "scout_tool_agent",
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"Change the counter from {start} to {target}, then finish. "
                        "Available tools: read() returns the current value; "
                        "add(amount) changes it by an integer between -3 and 3; "
                        "finish() ends the task and succeeds only at the target."
                    ),
                }
            ],
            "reward_model": {"style": "rule", "ground_truth": str(target)},
            "extra_info": {"index": i, "split": split, "scout_tool_task": {"start": start, "target": target}},
        }
        for i, (start, target) in enumerate(pairs)
    ]


def main():
    """Write distinct train/validation task pairs in verl parquet format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/scout_counter")
    args = parser.parse_args()
    pairs = [(start, target) for start in range(-20, 21) for target in range(-20, 21) if 0 < abs(start - target) <= 9]
    random.Random(42).shuffle(pairs)
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for split, subset in (("train", pairs[:256]), ("val", pairs[256:320])):
        path = folder / f"{split}.parquet"
        pd.DataFrame(make_rows(subset, split)).to_parquet(path, index=False)
        print(f"{path}: {len(subset)} tasks")


if __name__ == "__main__":
    main()
