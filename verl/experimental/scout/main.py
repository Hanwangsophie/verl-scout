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
"""Launch SCOUT with ``python -m verl.experimental.scout.main``."""

import hydra
from omegaconf import OmegaConf


class ScoutTaskRunner:
    """Ray driver with the same agent-loop/TransferQueue lifecycle as V1 PPO."""

    def run(self, config):
        """Initialize the synchronous runtime and close shared resources on exit."""
        import transfer_queue as tq

        from verl.experimental.scout.trainer import SCOUTTrainer
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ
        from verl.utils.logging_utils import configure_verl_logging

        configure_verl_logging()
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer = None
        succeeded = False
        try:
            trainer = SCOUTTrainer(config)
            trainer.init()
            agent_loop = AgentLoopManagerTQ.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(agent_loop)
            succeeded = True
        finally:
            try:
                if trainer is not None and getattr(trainer, "logger", None) is not None:
                    trainer.logger.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path="../../trainer/config", config_name="scout_trainer", version_base=None)
def main(config):
    """Validate the recipe before launching the Ray task runner."""
    import ray

    from verl.experimental.scout.config import validate_scout
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    validate_scout(config)
    validate_config(config, use_reference_policy=False, use_critic=False)
    run_ppo(config, ray.remote(ScoutTaskRunner))


if __name__ == "__main__":
    main()
