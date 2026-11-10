# 单 A800 完整 GRPO Baseline 训练前报告

日期：2026-11-10（Asia/Shanghai）。本轮只完成 251 Step 标准 GRPO Baseline 的配置、保护、审计、备份和离线验证，没有启动训练、模型加载或 Optimizer Step。

## 最终配置

正式配置为 `configs/a800_baseline_full.yaml`，独立于已经验证的 `a800_baseline_4step.yaml`。训练将从固定 revision `cdbee75f17c01a7cc42f958dc650907174af0554` 的 Qwen3-4B-Instruct-2507 基座开始，使用完整的 4,030 题 train split 和 522 题 val split。核心参数保持为 batch=16、group size=4、LoRA rank/alpha=32、lr=1e-6、FSDP、vLLM、标准 GRPO、temperature=0.6 和一轮训练。

rLLM 的 train dataloader 固定使用 `drop_last=True`，所以 4,030 题会执行 `4030 // 16 = 251` Step，实际优化 4,016 题，最后 14 题不会进入本轮更新；val dataloader 使用 `drop_last=False`，两次验证均覆盖完整 522 题。这里保持官方 rLLM 行为，没有为了处理尾批而改变 batch 或训练算法。

验证调度已经对固定 rLLM 源码和解析后的 Hydra 配置做静态检查：`val_before_train=true` 触发训练前完整验证，`test_freq=1000` 在 251 Step 内不触发周期验证，但正数配置仍会在正常训练结束后触发一次完整验证。因而同一完整 val split 只在训练前后各运行一次。

| 项目 | 最终值 |
| --- | ---: |
| 训练 Step | 251 |
| Train / Val 原始题数 | 4,030 / 522 |
| 实际优化题数 / rollout | 4,016 / 16,064 |
| Train batch / group | 16 / 4 |
| LoRA rank / alpha | 32 / 32 |
| 学习率 | 1e-6 |
| Checkpoint 频率 | 每 25 Step |
| Checkpoint 保留数 | 最近 2 份 |
| 训练前 / 后验证 | 完整 522 / 完整 522 |
| 周期验证 | 无 |
| Fresh run | `resume_mode=disable`，要求目标目录为空 |

## Checkpoint 与恢复

rLLM 会在 Step 25、50、75、100、125、150、175、200、225 和 250 保存周期 Checkpoint，并在保存新版本前删除超出上限的最旧版本。审计层在训练后完整验证成功结束时补存 Step 251，随后验证 `latest_checkpointed_iteration.txt`、LoRA metadata 和最终目录；因此 Step 251 不会被 Step 250 或 Step 225 冒充。每次保存均记录 `checkpoint_start`、`checkpoint_saved` 或明确的 `checkpoint_error` 事件。

正常启动使用 `resume_mode=disable`，并在加载模型前检查完整实验的 Checkpoint 目录为空，防止意外继承四步权重。故障恢复使用同一个配置，只允许显式覆盖：

```text
trainer.resume_mode=auto
finqa.require_empty_checkpoint_dir=false
finqa.min_free_disk_gib=20
```

`resume_mode=auto` 会从 `latest_checkpointed_iteration.txt` 恢复 model、optimizer、RNG、LR scheduler、dataloader 和 global step。四步实验已经实际完成一次从 Step 4 的零更新恢复验证，返回码 0；完整训练沿用同一实现。恢复时降低磁盘启动阈值只用于允许已有两份 Checkpoint 的目录继续运行，不改变训练参数。

## 备份与磁盘

四步成功实验的 11 个 Checkpoint 文件及关键失败、成功、恢复日志已在远端生成 SHA256 清单，并复制到本地忽略目录：

```text
.cache/deploy/baseline64-backup-20261110
```

本地重新计算哈希后，11/11 个 Checkpoint 文件和日志归档全部通过校验；日志归档 SHA256 为 `7686c9015cb7960cbca02ef33dbc9725c8ca189e988e1b3079e0ebb5a29ffd19`。确认校验通过后，只删除远端 `/root/autodl-tmp/finqa/checkpoints/qwen3-4b-a800-baseline64/global_step_4`，成功、失败和恢复日志仍保留。

备份前 `/root/autodl-tmp` 为 90 GiB，总可用 51.25 GiB；系统盘可用 11.42 GiB。四步 Checkpoint 约 18 GiB。校验并删除该远端副本后，数据盘可用 68.43 GiB，能够通过 fresh run 的 65 GiB 启动保护，并容纳两个约 18 GiB 的轮换 Checkpoint。系统盘不再用于保存新训练权重。

## 审计与依赖

训练必须由 `scripts/run_smoke_audit.py --action baseline-full --max-optimizer-steps 251` 启动；直接运行 `scripts/gpu.sh baseline-full` 会被拒绝。启动器冻结 Git commit、Git diff、解析配置、CUDA lock 和依赖版本，并持续记录：

- 每个 rollout 的 response token IDs 与 logprobs、Reward、group advantage、loss、grad norm 和 LoRA 权重摘要；
- Judge SDK 调用、实际 HTTP send、response/error、输入/输出/reasoning/cache token usage；
- 每秒 GPU 显存、利用率、功耗、进程 RSS、cgroup 内存和 OOM 计数；
- Checkpoint 开始、完成、失败、最终 Step 251 校验和作业总耗时/成本。

沿用成功四步实验的依赖，不升级核心组件：rLLM 0.3.0rc0、rLLM Model Gateway 0.1.0、verl 0.8.0、vLLM 0.22.1、PyTorch 2.11.0+cu130、FlashAttention 2.8.3、TensorDict 0.10.0、Transformers 5.22.2、PEFT 0.21.2、Ray 2.60.0 和 NumPy 2.3.5。verl 0.8.0 对 NumPy `<2` 的元数据约束例外继续如实保留，没有修改第三方元数据或隐藏 `uv pip check` 结果。

## 预算

四步实测外推中心值为 39.2 GPU 小时，其中训练约 36.9 小时，两次完整验证约 2.3 小时。按用户确认的 ¥5.59/小时，GPU 作业窗口约 ¥219；建议按网络波动、难题长度和 Checkpoint I/O 预留 35–50 小时，即约 ¥196–¥280。

Judge 预计 17,108 次逻辑评分；按四步实验 1.096875 的实际 HTTP/逻辑请求倍率，约 18,765 次 HTTP 请求。预计输入 12.451M、输出 4.108M、合计 16.559M token。服务商当前公开模型页列出的原价为输入 ¥12/M token、输出 ¥36/M token、缓存命中输入 ¥1.5/M token；四步实验没有 cache token，因此预算按普通输入计价，约 ¥297.29。GPU 与 Judge 中心预算合计约 ¥516.33，35–50 GPU 小时范围约 ¥492.94–¥576.79。

公开单价来源：<https://platform.qianwenai.com/docs/models/qwen3-8-max>。账单查询说明：<https://platform.qianwenai.com/docs/resources/bill-query>。当前浏览器没有登录服务商账户，因此无法核实账户折扣、赠送额度或实际应付账单；正式运行的审计会保存实际 token usage，最终金额应以服务商账单为准。

## 验证与启动状态

本地测试、Ruff、格式检查和四个 GPU 配置的 Hydra 静态解析均通过；远端已用安装的固定 verl/rLLM 完成 full profile 解析，确认 251 Step、batch=16、group=4、LoRA32、训练前后验证和 Checkpoint 参数。数据集实查为 train=4,030、val=522。远端 GPU 当前 0 MiB，完整实验的 Checkpoint 目录不存在。

准备代码提交为 `d6d00ab`，相关提交均已推送到 GitHub。同步最终文档提交后，远端将固定到同一 HEAD。训练应在 `screen` 中启动，以免 SSH 断开终止约 39 小时作业；启动命令会写入独立的实验目录并通过审计器严格限制为 251 个 Optimizer Step。

最终授权后的 fresh run 命令为：

```bash
screen -S finqa-full
source /root/autodl-tmp/finqa/env.sh
cd /root/autodl-tmp/FinQA-agentic-rl
python scripts/run_smoke_audit.py \
  --run-dir /root/autodl-tmp/finqa/experiments/full-baseline-YYYYMMDDTHHMMSSZ \
  --timeout-seconds 180000 \
  --action baseline-full \
  --max-optimizer-steps 251 \
  --gpu-hourly-price 5.59 \
  --judge-input-price-per-million 12 \
  --judge-output-price-per-million 36
```

恢复时复用同一命令和同一实验 Checkpoint 目录，并额外传入三个 `--hydra-override` 参数，对应上文列出的 `resume_mode`、空目录保护和恢复磁盘阈值。每次尝试使用新的审计目录，保留原始失败日志。

当前具备正式 GRPO Baseline 的软件、显存、配置、监控和恢复条件。完成备份校验与空间释放后，只等待用户最终授权；本轮没有调用完整训练入口。
