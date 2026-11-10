# Tool-level SCOUT：代码与运行

这是在本仓库单轮 [SCOUT](SCOUT.md) 基础上新增的实验路径：**教师从学生实际访问的工具环境状态继续执行，学生只蒸馏动作输出。** 当前附带可恢复的本地 counter 工具环境，用于验证链路；它不是 ALFWorld、WebShop，也不构成论文指标复现。

当前只完成 CPU 环境执行、数值与控制器测试、配置及静态检查。模型生成和 GPU worker 在 CPU 测试中使用替身；尚未运行真实 Qwen 模型训练、分布式 NCCL 同步或评估收敛效果。

## 具体改动

| 文件 | 职责 |
| --- | --- |
| `verl/experimental/scout/tool_env.py` | 环境的 `reset/snapshot/restore/step/close` 契约和 counter 示例 |
| `verl/experimental/scout/tool_rollout.py` | 学生、教师共用多轮执行器；原始 token、动作 mask、完整观察边界 |
| `verl/experimental/scout/tool_agent_loop.py` | 将学生工具轨迹及环境快照写入 verl TransferQueue |
| `verl/experimental/scout/tool_trainer.py` | 从当前学生 batch 选择完整决策边界，保存额外 checkpoint 配置 |
| `verl/experimental/scout/tool_teacher.py` | 每个前缀建立 K 个独立环境、恢复同一快照、真实执行教师动作 |
| `verl/experimental/scout/teacher.py` | 复用 GRPO、固定 reference、权重同步；advantage 显式乘动作 mask |
| `verl/experimental/scout/main.py`、`config.py` | 选择 text/tool 路径并验证限制 |
| `verl/trainer/config/scout_tool_trainer.yaml` | 独立工具实验配置；原单轮入口仍可运行 |

执行顺序：

1. 学生在独立环境中交互。每次决策前保存：精确 token 前缀、可见消息、执行状态快照、已用动作数。
2. 当前教师对学生轨迹打 token 分。学生使用现有 sampled K1 / PPO 蒸馏更新；本版本没有额外学生任务奖励 loss。
3. 到教师更新周期时，从**同一个学生 batch**选择决策边界。默认将 SCOUT 的线性前缀比例映射到最近的完整观察边界，不截断 JSON 或工具返回值。
4. 为每个边界创建 K 个独立环境，恢复同一快照。教师的新动作获得新执行结果，不复用学生后续观察。
5. 以环境返回的续跑累计 reward 做每个边界内的 GRPO。只训练教师新生成的动作 token；学生前缀和观察不参与 loss。
6. 发布教师权重并等待同步完成；下一轮学生 rollout 使用更新后的教师打分。

同一个前缀的历史 reward 对 K 条续跑都是常数，故教师 GRPO 使用续跑 reward。所有分支 reward 相同时，任务 GRPO advantage 为零；固定 reference KL 仍可能产生更新。不得将这种情况报告为已学到恢复能力。

## 第一版动作格式

每轮只生成一个 JSON 对象，例如：

```json
{"name": "add", "arguments": {"amount": 2}}
```

counter 支持 `read()`、`add(amount)` 和 `finish()`。工具结果以 `Environment observation` 用户消息加入上下文。它是 action-only 环境协议，**不是 OpenAI function-calling API 的适配器**。Qwen3 设置 `enable_thinking=false`；本版本不处理自由 reasoning 与工具调用混合输出的细粒度 span 提取。

- 模型生成的动作输出（含 EOS）参与 loss；格式错误的动作输出也保留，环境返回错误反馈。
- 初始提示、环境返回值、自动插入的 chat 边界 token：mask 为 0。
- 截断的动作不执行；放不下完整观察时结束轨迹，不保存半个观察作为续跑边界。
- `finish` 是正常可学习的动作；环境决定是否终止及成功。

这里的“tool-level”指**环境状态及续跑边界位于工具决策层，蒸馏监督覆盖动作 token**。损失仍是原有 token 级 sampled K1，不是整段动作概率的 PPO，也没有实现 ToolOPD 的 hindsight skill / credit weighting。这些应分别作为后续实验。

## 运行

在仓库根目录，使用 [SCOUT.md](SCOUT.md) 中相同的 Python 3.12 / CUDA / `uv.lock` 环境。默认依然预留 8 张 GPU：学生 2、教师推理 2、教师训练及 reference 4；这不是测量过的最低显存要求。

```bash
uv sync --frozen --all-packages --extra vllm --extra fsdp
source .venv/bin/activate
python examples/scout/prepare_tool_counter.py

# 两步 smoke run，每步更新一次教师，用于覆盖完整更新链路。
bash examples/scout/run_qwen3_1_7b_scout_tool_fsdp.sh \
  trainer.total_training_steps=2 \
  trainer.resume_mode=disable \
  trainer.save_freq=1 \
  trainer.test_freq=1 \
  trainer.default_local_dir="$PWD/checkpoints/scout-tool-smoke" \
  scout.teacher_update_interval=1 \
  scout.prefix_ratio_start=0.5 \
  scout.prefix_ratio_end=0.9
```

预处理生成 256 个训练任务和 64 个验证任务，两者的 `(start, target)` 对不重叠，固定随机种子。这只是接口自检数据，不能用作 agent benchmark 成绩。

只解析配置、不启动模型：

```bash
bash examples/scout/run_qwen3_1_7b_scout_tool_fsdp.sh --cfg job
```

CPU 验证（无需模型下载和 GPU；须已安装测试依赖）：

```bash
python -m pytest -q tests/experimental/scout \
  tests/workers/test_distillation_topk_symmetry_on_cpu.py
```

应观察到 `scout/teacher/updates` 增长、`action_tokens` 少于包含观察的 `continuation_tokens`，以及教师 `nonconstant_group_fraction`。验证任务的环境 reward、非法动作数和动作数通过现有 reward metrics 输出。CPU 测试验证了两个教师分支从同一错误状态开始，实际执行不同动作后分别成功和失败；这不等于模型学会了该策略。

`scout.tool.prefix_source=initial` 提供“教师只从任务起点训练”的对照。组数和 K 相同，但实际生成 token、环境步数可能不同，比较时必须统计成本，不能直接宣称算力相等。要检验学生状态适配，需要与冻结教师及这个起点对照一起评估。

## 接真实环境

实现 `SnapshotToolEnvironment` 子类并配置：

```yaml
scout:
  tool:
    environment_class: your_package.your_env.YourEnvironment
    environment_config: {}
```

数据保留 verl 的 `prompt/data_source/reward_model/extra_info` 字段，设置 `agent_name: scout_tool_agent`，并将初始化参数放入 `extra_info.scout_tool_task`。`reset` 返回公开初始观察；`step` 返回 `ToolStep(observation, reward, done, info)`，其中 reward 是本次执行的增量。

适配器必须保证：

1. `snapshot` 包含任务、真实可变状态、RNG、预算及必要的会话状态；`restore` 可以在全新实例中恢复。
2. 每个教师分支的文件、浏览器、数据库等可变资源独立；不能把 K 个分支指向同一真实会话。
3. 快照只用于环境恢复，不拼进模型输入；模型只能看到与学生相同的公开历史。
4. 新动作必须重新执行。只有在环境确定、动作重放能验证到达相同状态时，才可用重放实现恢复。
5. 异常与取消会执行 `close`；基础设施错误应抛异常，不能伪装成任务失败混进 GRPO。

ALFWorld、WebShop 的实际恢复/克隆适配器尚未实现。要支持它们，先验证上述契约，再接数据及任务成功率评估。多模态、异步策略更新、外部工具共享会话和混合 thinking 输出不在当前实验范围内。

Checkpoint 同时记录 tool 配置、chat template 参数和教师更新次数。变更环境配置、动作预算、前缀来源后不能无提示地沿用旧训练状态；请使用新的实验目录。
