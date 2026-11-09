# SCOUT on verl：实现与运行说明

这是论文 [On the Off-Policy Teacher in On-Policy Distillation](https://arxiv.org/abs/2609.38360) 的**非官方实验实现**，基于 verl 提交 `482f86fd09959912c99c0acf2b90329f9e061da7`。论文作者的[项目主页](https://shrango.github.io/scout/)在实现时仍标注代码待发布。

**验证范围：CPU 算法与控制器测试、配置解析和静态检查。尚未完成真实多 GPU 训练，不代表已复现论文指标。** GPU 显存、吞吐、分布式 NCCL 同步和收敛需要在训练机器上验证。

## 实现内容

| 模块 | 位置 | 行为 |
| --- | --- | --- |
| SCOUT 入口 | `verl/experimental/scout/main.py` | 复用 V1 同步 trainer 和 agent loop |
| 更新调度 | `verl/experimental/scout/trainer.py` | 从当前学生 batch 取前缀，周期更新教师；联合保存/恢复 |
| 教师运行时 | `verl/experimental/scout/teacher.py` | 独立 FSDP2 optimizer、固定初始 reference、GRPO、NCCL 同步 |
| 前缀与数据转换 | `verl/experimental/scout/utils.py` | 原始 token ID 拼接，排除 EOS/padding，按前缀分组 |
| 学生 K1 advantage | `verl/trainer/distillation/losses.py` | 可选固定到 old policy，保留 current-policy PPO ratio 的梯度 |
| 配置 | `verl/trainer/config/scout_trainer.yaml` | 两个模型、三个资源池、更新周期与前缀课程 |
| 启动脚本 | `examples/scout/run_qwen3_1_7b_scout_fsdp.sh` | Qwen3-4B-Instruct-2507 → Qwen3-1.7B |
| 测试 | `tests/experimental/scout/` | 数值行为、调度、同步屏障、checkpoint 一致性 |

每个学生 step：学生生成完整回答 → 当前教师打 token 分 → 学生更新。每隔 `teacher_update_interval` 个 step，取**这批回答**的前缀；每个前缀生成 K 个教师续写；verifier 检查“学生前缀 + 教师续写”；GRPO 只对教师续写 token 计算 loss。更新完等待教师推理副本完成权重同步，下一步学生才继续。

教师 reference 始终是初始教师，既不追随学生，也不随教师 optimizer 更新。学生没有额外任务奖励 loss。旧学生 log probability 通过 verl 的训练引擎在更新前重算；它与 vLLM 采样 log probability 可能有数值差异，这里保留上游的默认处理。

## 环境和 GPU

使用 Linux、Python 3.12、NVIDIA GPU，以及当前提交的 `uv.lock`。先按上游 [安装说明](https://verl.readthedocs.io/en/latest/start/install.html)配置驱动和 CUDA，再在仓库根目录执行：

```bash
uv sync --frozen --all-packages --extra vllm --extra fsdp
```

默认脚本也会通过相同的 `uv run --frozen` 环境启动本地进程和 Ray workers。在已配置好的容器里可以设 `VERL_USE_UV=0`，此时使用 PATH 中的 `python3`，由你确保包版本兼容。

默认资源总数是 **8 GPU**：学生训练/推理共用 2 张，教师推理 2 张，教师训练及固定 reference 共用另 4 张。**8 张是调度配置，不是经过测量的最小硬件要求**；显存是否够与卡型、序列长度、batch 和 offload 有关。多机需要 Ray 集群、可互通 NCCL 网络及各节点一致可见的 checkpoint 路径。

| 用途 | GPU 配置 | 默认 |
| --- | --- | --- |
| 学生池 | `trainer.n_gpus_per_node` × `trainer.nnodes` | 2 × 1 |
| 教师推理池 | `distillation.n_gpus_per_node` × `distillation.nnodes` | 2 × 1 |
| 教师训练池 | `scout.teacher_gpus_per_node` × `scout.teacher_nnodes` | 4 × 1 |

GPU 数量变化时需要同时调整对应的 tensor parallel size。当前仅支持文本、完整参数训练、FSDP/FSDP2、vLLM 和同步单轮生成；学生/教师的 token→ID 映射及特殊 token 必须相同。不同词表、LoRA、多轮工具调用和异步训练会被拒绝或不在本实现支持范围内。

## 先跑小规模验证

先用 verl 自带 GSM8K 预处理检查链路。**它不是论文的训练数据。**

```bash
uv run --frozen --all-packages --extra vllm --extra fsdp \
  python examples/data_preprocess/gsm8k.py --local_save_dir "$PWD/data/scout-gsm8k"

export TRAIN_FILE="$PWD/data/scout-gsm8k/train.parquet"
export VAL_FILE="$PWD/data/scout-gsm8k/test.parquet"

bash examples/scout/run_qwen3_1_7b_scout_fsdp.sh \
  trainer.total_training_steps=2 \
  trainer.resume_mode=disable \
  trainer.save_freq=1 \
  trainer.default_local_dir="$PWD/checkpoints/scout-smoke" \
  scout.teacher_update_interval=1 \
  scout.teacher_lr=5e-7
```

这里故意每步更新教师，确保两步就覆盖整个链路；用于检查，不用于报告论文结果。应看到 `scout/teacher/updates` 变为 1、2，`scout/teacher/prefix_count=16`，以及教师 continuation token 数和 reward 日志。所有组 reward 相同会得到零 GRPO advantage；需要先检查 verifier 和数据格式。

只解析配置、不加载模型：

```bash
bash examples/scout/run_qwen3_1_7b_scout_fsdp.sh --cfg job
```

## 接近论文数学实验的参数

使用 OpenR1-Math-46K-8192，并转换为 verl parquet 格式，至少保留 `prompt`、`data_source`、`reward_model.ground_truth` 和 `extra_info`。`data_source` 必须对应合适的数学 verifier；不要只改数据文件名却继续使用 GSM8K 专用答案解析。评估集应独立于训练集。

以下命令对齐论文报告的主要训练超参数；数据预处理、prompt/chat template、前缀课程端点、verifier 和评估协议仍需逐项核对作者实现：

```bash
export TRAIN_FILE=/absolute/path/to/openr1_train.parquet
export VAL_FILE=/absolute/path/to/math_validation.parquet

bash examples/scout/run_qwen3_1_7b_scout_fsdp.sh \
  data.train_batch_size=128 \
  data.max_response_length=16384 \
  actor_rollout_ref.rollout.max_model_len=17409 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
  scout.teacher_max_response_length=16384 \
  scout.teacher_update_interval=10 \
  scout.teacher_rollouts_per_prefix=8 \
  scout.teacher_lr=5e-6 \
  scout.teacher_kl_coef=0.001 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="$PWD/checkpoints/scout-math"
```

这个长度需要明显更多显存和时间。教师最大输入长度包含问题和学生前缀，还需要留出教师续写预算；配置验证会检查这一点。教师训练 token budget 同样继承上面的 32768。

默认线性前缀比例从 0.1 到 0.9，**这两个端点是工程选择，论文没有明确给出训练课程端点**；可通过 `scout.prefix_ratio_start/end` 调整。教师采用 verl 的 GRPO 及 `low_var_kl` reference 正则，loss 按每条续写内 token 平均、再按序列平均。当前保留模型默认 chat template；论文前缀诊断使用过 Qwen3 non-thinking，主实验模板未完全指定，需要明确记录你选择的模式。可用 `+data.apply_chat_template_kwargs.enable_thinking=false` 显式关闭 Qwen3 thinking。

## 对照实验

保留本配置的学生目标，设置 `scout.teacher_update_interval` 大于总训练步数，即得到冻结教师的功能对照，但仍分配教师训练池，**不能用来测量节省算力后的基线效率**。将 `scout.prefix_ratio_start=0 scout.prefix_ratio_end=0` 可检查“仅从原问题进行教师 GRPO”的对照。

本实现默认 `distillation.distillation_loss.use_old_log_probs=true`，实现论文 Eq. (2) 的固定 K1 advantage；这个新选项在上游普通 OPD 配置中默认为 false，原有训练行为不变。

## 保存与恢复

`global_step_N/` 下保存原有学生 checkpoint，以及 `teacher/`（教师参数、optimizer、scheduler 等）和 `scout_state.json`。只有教师保存成功，才让父 trainer 发布最新学生 step 指针。恢复后会重新把教师 checkpoint 同步到推理服务，避免使用发布版初始教师继续评分。

同一配置和目录使用 `trainer.resume_mode=auto`；或使用 `trainer.resume_mode=resume_path trainer.resume_from_path=/absolute/path/global_step_N`。恢复会检查模型、总步数、课程、更新次数及教师配置。**改变总步数会改变前缀课程，因此会拒绝直接恢复**；新实验应使用新目录。恢复的数据文件和学生 optimizer 参数也应保持一致。

## 验证和限制

```bash
python -m pytest -q tests/experimental/scout
pre-commit run --files $(git diff --name-only 482f86fd09959912c99c0acf2b90329f9e061da7 HEAD)
```

测试覆盖前缀 loss 排除、完整答案评分、相同长度 batch 的 jagged tensor、每个前缀独立 GRPO 组、K1 固定优势、旧 OPD 默认行为、tokenizer 映射检查、更新间隔、padding 过滤、权重同步等待、过期教师版本拒绝和 checkpoint 顺序。GPU 边界用 fake/mock 隔离；测试通过**不能证明**分布式训练已经跑通。

本次 CPU 检查环境使用 Python 3.12、PyTorch 2.9.0+cpu、TensorDict 0.10.0、TransferQueue 0.1.11；它不是上述锁定 GPU 环境的替代。论文原实验的软件版本也早于此处使用的上游提交。

Megatron 专用回归测试因当前环境未安装 `megatron-core` 无法收集；本实现仅支持 FSDP/FSDP2。

需要在 GPU 服务器完成小规模训练、断点恢复和完整独立评估后，才能报告复现效果。本实现不含代码执行 sandbox verifier，也不提供论文 benchmark 的预计算结果。实现由 OpenAI Codex 协助生成，应审阅后用于正式实验。
