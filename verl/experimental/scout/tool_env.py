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
"""Isolated, restorable environments for the experimental tool-level SCOUT path."""

import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


def json_copy(value, max_bytes=262144):
    """Detach a finite JSON payload; never serialize live sessions or pickle handles."""
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > max_bytes:
        raise ValueError("SCOUT environment snapshot exceeds max_snapshot_bytes")
    return json.loads(encoded)


@dataclass
class ToolStep:
    """Observable tool response and reward increment from an actual execution."""

    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


class SnapshotToolEnvironment(ABC):
    """One isolated environment per trajectory, including each teacher branch.

    ``restore`` must restore ALL execution state, including task, RNG, budgets,
    files and session state where applicable. A transcript alone is insufficient.
    State is runtime-only: the runner never inserts snapshots into model prompts.
    Adapters must implement invalid-action feedback and close resources on error.
    External mutations must be confined to independently cloned sandboxes.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    async def reset(self, task: dict) -> str:
        """Initialize one task and return its public initial observation."""

    @abstractmethod
    async def snapshot(self) -> dict:
        """Return a self-contained, JSON-serializable execution checkpoint."""

    @abstractmethod
    async def restore(self, state: dict) -> None:
        """Restore into a fresh instance without modifying the supplied checkpoint."""

    @abstractmethod
    async def step(self, action: dict | None) -> ToolStep:
        """Execute {name, arguments}; None represents malformed action output."""

    @abstractmethod
    async def close(self) -> None:
        """Release this branch's resources, including after cancellation."""


def make_environment(class_path: str, config: dict) -> SnapshotToolEnvironment:
    """Load the configured adapter; never select executable classes from a dataset."""
    module, name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    if not issubclass(cls, SnapshotToolEnvironment):
        raise TypeError("Tool environment must implement SnapshotToolEnvironment")
    return cls(json_copy(config))


class CounterEnvironment(SnapshotToolEnvironment):
    """Small deterministic stateful tool sandbox, NOT an agent benchmark.

    Actions are read(), add(amount: int in [-3, 3]), and finish(). Success requires
    the actual counter to equal the task target when finish() executes. This makes
    a wrong student action change the state from which teacher branches recover.
    """

    async def reset(self, task):
        """Start a new public counter task."""
        if type(task.get("start")) is not int or type(task.get("target")) is not int:
            raise ValueError("Counter task requires integer start and target")
        self.state = {"value": task["start"], "target": task["target"], "calls": 0, "done": False}
        return self._observe()

    def _observe(self):
        return json.dumps({"value": self.state["value"], "target": self.state["target"]})

    async def snapshot(self):
        """Save both execution state and task identity."""
        return {"schema": 1, "state": json_copy(self.state)}

    async def restore(self, state):
        """Clone a snapshot without sharing mutable state with sibling branches."""
        if state.get("schema") != 1:
            raise ValueError("Unsupported counter snapshot version")
        self.state = json_copy(state["state"])
        if self.state["done"]:
            raise ValueError("Cannot continue a terminated counter environment")

    async def step(self, action):
        """Execute one action and return fresh feedback, never cached observations."""
        if self.state["done"]:
            raise RuntimeError("Cannot execute an action after termination")
        self.state["calls"] += 1
        reward = -float(self.config.get("step_cost", 0.0))
        name = action.get("name") if action else None
        args = action.get("arguments", {}) if action else {}
        valid = False
        if name == "read" and args == {}:
            valid = True
        elif name == "add" and set(args) == {"amount"}:
            amount = args["amount"]
            if type(amount) is int and -3 <= amount <= 3:
                self.state["value"] += amount
                valid = True
        elif name == "finish" and args == {}:
            self.state["done"] = True
            reward += float(self.state["value"] == self.state["target"])
            valid = True
        observation = self._observe()
        if not valid:
            reward -= float(self.config.get("invalid_action_cost", 0.0))
            observation = "Invalid action. Use read(), add(amount: integer in [-3, 3]), or finish(). " + observation
        return ToolStep(observation, reward, self.state["done"], {"invalid_action": not valid})

    async def close(self):
        """No external resources are held by this in-memory sandbox."""
