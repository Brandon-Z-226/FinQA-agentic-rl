# 单 A800 四步 GRPO Baseline 报告

日期：2026-11-09（Asia/Shanghai）。本轮在 AutoDL NVIDIA A800 80GB PCIe
单卡上完成固定 64 题、每题 4 条 rollout、共 4 个 Optimizer Step 的标准
GRPO 小规模 Baseline。训练成功后已停止，没有启动 251 Step 完整训练。

## 结论

标准训练闭环已经通过正式 batch=16 的真实验证：四批训练数据均完成 rollout、
Gateway token/logprob 捕获、GRPO advantage、backward、optimizer、LoRA 权重同步和
checkpoint。64 个 group 中有 20 个出现组内 Reward 差异，四步均得到非零 advantage、
有限 loss/梯度和实际 LoRA 权重变化。没有 CUDA OOM 或 CPU cgroup OOM。

成功运行耗时 46.71 分钟，按 ¥5.59/小时计 GPU 作业窗口约 ¥4.352。一次有价值的
失败运行和一次 checkpoint 恢复验证一并计入时，本阶段三个 GPU 作业窗口约 ¥6.231。
完整一轮按本次吞吐、只做训练前后两次完整验证估算约 39.2 GPU 小时、¥219；Judge
预计约 18,765 次 HTTP 请求和 1,655.9 万 token。由于服务商没有提供可核实的 token
单价，本报告不虚构 API 金额。

## 代码、依赖与实验边界

| 项目 | 实际值 |
| --- | --- |
| 成功运行提交 | `30ebd8f9230204959b3e1bd0d8a8521af3a82509` |
| Judge 异常修复 | `4000be4`：有限重试、最终失败抛出，不再转为 Reward=0 |
| HTTP/usage 审计 | `b884c2f`：在 OpenAI SDK transport 层统计真实发送 |
| Judge 格式兼容 | `30ebd8f`：解析官方 `{True/False, explanation}` 返回并有限重试 |
| 模型 | Qwen3-4B-Instruct-2507，固定 revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| 训练配置 | `configs/a800_baseline_4step.yaml` |
| 核心参数 | batch=16、group=4、LoRA rank/alpha=32、lr=1e-6、标准 GRPO |
| 关键依赖 | torch 2.11.0+cu130、vLLM 0.22.1、verl 0.8.0、FlashAttention 2.8.3 |
| 其余版本 | rLLM 0.3.0rc0、TensorDict 0.10.0、Transformers 5.22.2、PEFT 0.21.2 |

verl 0.8.0 与 NumPy 2.3.5 的已记录元数据例外原样保留。本轮没有升级固定依赖、
修改 Reward/GRPO、改变模型或扩大训练规模。

错误注入测试覆盖了连接失败、超时、非重试错误和持续错误响应：可重试错误只执行
有限次数，耗尽后抛出 `JudgeConnectionError` 或 `JudgeResponseError`；训练链路不会将
API 故障记为错误答案。成功运行中 Judge SDK/HTTP error 均为 0。

## Checkpoint 备份与磁盘

成功单步 smoke 的 checkpoint 和日志先生成 SHA256 清单，再复制到独立系统盘：

```text
/root/finqa-backups/smoke-20261108T185116Z
```

11 个 checkpoint 文件和 62 个日志文件共 73 项逐一校验通过。确认备份后只删除了
数据盘上的原 smoke checkpoint，必要日志仍保留；数据盘可用空间从约 52 GiB 回升到
约 69 GiB。四步 Baseline 产生新 checkpoint 后，当前状态为：

| 位置 | 大小 / 可用空间 |
| --- | ---: |
| 系统盘 smoke 备份 | 18 GiB；系统盘剩余约 12 GiB |
| 数据盘四步 checkpoint | 18 GiB |
| `/root/autodl-tmp` | 90 GiB 总量，剩余约 52 GiB |

没有删除其他数据盘内容。

## 两次训练尝试

首次运行保留在
`/root/autodl-tmp/finqa/experiments/baseline64-20261109T0430Z/`。它完成 Step 1，
随后 Judge 返回官方 `{False, '...not true...'}` 结构；旧的宽松文本判断同时看到
`false` 与说明中的 `true`，明确抛出 `JudgeResponseError`。这不是错误答案、OOM 或
API 中断。运行返回码 1，耗时 1,017.52 秒，132 次 HTTP 请求均成功，日志完整保留。

解析器按官方结构修正并通过测试后，只重试一次真实训练。成功运行位于：

```text
/root/autodl-tmp/finqa/experiments/baseline64-retry1-20261109T0440Z
```

返回码 0，非超时，墙钟 2,802.70 秒。审计硬限制为最多 4 个
`optimizer.step()`；记录到调用/返回各 4 次，没有第五次。

## 数据、轨迹与吞吐

训练集用 `seed=20261109` 从 4,030 题固定抽取 64 个不同索引；不是前 64 题，也不是
连续简单题。验证集用 `seed=20261110` 固定抽取 32 题。选择文件 SHA256 为
`d66cbe973e56614bbf3d0f7b6f633f24e7958149a5d11dea43322f2d8a0c8215`，完整索引保存在
Hydra output 的 `dataset-selection.json` 和本地归档中。

| Step | 题目 / rollout | Agent step | Prompt IDs | Response IDs / logprobs | 完整 Step 墙钟 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 / 64 | 307 | 666,703 | 64,093 / 64,093 | 456.69 s |
| 2 | 16 / 64 | 312 | 732,699 | 70,346 / 70,346 | 574.97 s |
| 3 | 16 / 64 | 342 | 782,222 | 85,671 / 85,671 | 574.84 s |
| 4 | 16 / 64 | 322 | 725,340 | 69,549 / 69,549 | 509.26 s |

所有 1,283 个 Agent step 的 response token IDs 与 rollout logprobs 长度一致。每个完整
训练 Step 平均 528.94 秒，即 8.82 分钟、约 7.26 rollout/分钟。该时间包含 rollout、
advantage、policy update、权重同步；Step 4 还包含 checkpoint。

## Reward、Advantage 与优化器

256 条训练 rollout 中 156 条 Reward=1，总 Reward mean=0.609375。64 个问题 group 中
20 个有组内 Reward 差异，占 31.25%；其余全对或全错，标准 GRPO 对这些组给出零优势。

| Step | Reward mean | 有信息 group / 16 | Advantage std | 零 advantage 比例 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.625000 | 3 | 0.433012 | 81.25% |
| 2 | 0.546875 | 5 | 0.559016 | 68.75% |
| 3 | 0.625000 | 5 | 0.559016 | 68.75% |
| 4 | 0.640625 | 7 | 0.661436 | 56.25% |

没有修改 Reward 或人为制造梯度。四步优化结果为：

| Step | Loss / PG loss | Grad norm | 有非零梯度张量 | 权重变化张量 | 最大绝对变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.112440 / -0.112440 | 0.164341 | 252 | 252 | 9.99997e-7 |
| 2 | -0.193629 / -0.193629 | 0.254140 | 504 | 504 | 1.00134e-6 |
| 3 | -0.230835 / -0.230835 | 0.255560 | 504 | 504 | 1.00358e-6 |
| 4 | 0.009021 / 0.009021 | 0.353378 | 504 | 504 | 1.00666e-6 |

每步梯度均为有限值。LoRA 可训练参数为 66,060,288；每步更新前后权重 SHA256 均不同，
且下一步的 before SHA256 与上一步 after SHA256 连续一致。Step 1 只有 252 个张量出现
非零梯度/变化，是该批稀疏有效 token 的实际结果；后续三步 504 个 LoRA 张量全部变化。

## 训练前后验证

训练前后都使用相同的 32 个固定验证样本；审计通过首轮 prompt token 的 SHA256
逐题确认样本集合相同。训练前 24/32=75.00%，训练后 22/32=68.75%，下降 6.25 个百分点。
配对结果为 20 题保持正确、6 题保持错误、2 题由错变对、4 题由对变错。

这个 32 题验证使用官方保留的 temperature=0.6、每题 1 条 rollout，存在采样方差；
4 步训练的样本量也很小。因此该结果只证明训练前后评测链路可比，不能据此断言模型
性能下降或提升。完整 Baseline 应使用同一 522 题验证集做训练前后比较。

## Judge 请求与 Token Usage

SDK 层与底层 transport 层均记录 351 次 attempt 和 351 次成功 response，错误数均为 0；
因此实际 HTTP 请求数可确认，不再把 SDK 调用数误当成网络发送数。逻辑评估为 320 次：
训练前验证 32、训练 256、训练后验证 32。结构化返回格式触发 31 次有限重试，额外请求
比例 9.69%。每次 response 都记录 usage，未记录 prompt、header 或 API key。

| 阶段 | 实际 HTTP 请求 | Prompt token | Completion token | Total token |
| --- | ---: | ---: | ---: | ---: |
| 训练 Step 1 | 64 | 41,105 | 10,819 | 51,924 |
| 训练 Step 2 | 81 | 53,166 | 21,472 | 74,638 |
| 训练 Step 3 | 66 | 46,823 | 14,642 | 61,465 |
| 训练 Step 4 | 67 | 43,624 | 13,752 | 57,376 |
| 全运行（含前后验证） | 351 | 232,892 | 76,835 | 309,727 |

Completion token 中 reasoning token 为 60,488；cached token 为 0。服务商没有公开可核实
的当前价格，用户目前只确认 GPU 为 ¥5.59/小时。若输入、输出单价分别为
`P_in`、`P_out` 元/百万 token，本次成功运行的 Judge 费用为
`0.232892 × P_in + 0.076835 × P_out` 元。

## GPU、CPU 与 Checkpoint 恢复

| 资源指标 | 实测峰值 |
| --- | ---: |
| nvidia-smi 显存采样 | 60,026 MiB（58.62 GiB） |
| actor CUDA allocated / reserved | 44.64 / 46.33 GiB |
| GPU utilization | 100% |
| cgroup 内存总量 | 81.38 GiB |
| 匿名内存 / 页缓存 | 26.20 / 56.19 GiB |
| 进程树 RSS 和 | 52.36 GiB；共享页可能重复计入 |
| cgroup OOM 计数 | 0 |

作业结束后 GPU 为 0 MiB，没有残留训练进程。Checkpoint 位于：

```text
/root/autodl-tmp/finqa/checkpoints/qwen3-4b-a800-baseline64/global_step_4
```

目录约 18 GiB，模型文件 17,910,768,163 bytes、optimizer 文件 528,916,075 bytes，
LoRA metadata 为 rank/alpha 32。随后执行独立恢复验证：`resume_mode=auto` 从
`global_step_4` 找到并加载 model、optimizer、RNG、LR scheduler 和 dataloader，LoRA metadata
通过检查；运行返回码 0、耗时 192.76 秒、执行 0 个新 Optimizer Step，GPU 再次释放。

## 完整 251 Step 估算与建议配置

正式 train 为 4,030 题，batch=16、drop-last，一轮 251 Step，实际训练 4,016 题、
16,064 条 rollout。成功运行完整 Step 平均 528.94 秒，训练本身线性外推为 36.88 小时。
训练前后各跑一次完整 522 题验证，按本次前后验证吞吐外推约 2.30 小时，合计中心估计：

- **39.2 GPU 小时，约 ¥219 GPU 作业窗口费用**；
- 建议按难题长度、网络抖动和 checkpoint I/O 预留 **35–50 小时、¥196–¥280**；
- 17,108 次逻辑 Judge 评估，按本次 1.096875 重试倍率约 **18,765 次 HTTP 请求**；
- 约 12.451M prompt token、4.108M completion token、合计 **16.559M token**；
- Judge 费用公式为 `12.451 × P_in + 4.108 × P_out` 元，单价单位为元/百万 token。

原 `test_freq=20` 会做 12 次周期完整验证和一次训练后验证；不仅昂贵，也不是获取可信
训练前后对比所必需。完整 Baseline 建议保留核心训练参数，只调整运行与验收参数：

| 参数 | 建议值 | 原因 |
| --- | ---: | --- |
| `trainer.val_before_train` | `true` | 建立同配置训练前基线 |
| `trainer.test_freq` | `1000` | 251 Step 内不做周期验证；正数仍触发训练后验证 |
| `trainer.save_freq` | `50` | 每约 7.3 小时保留恢复点 |
| `trainer.max_actor_ckpt_to_keep` | `2` | 保留最近两个恢复点并控制磁盘 |
| validation split | 完整 522 题、`n_val=1` | 训练前后公平比较 |

模型、数据、batch=16、group=4、LoRA32、lr=1e-6、PPO epochs=1、Reward 和标准 GRPO
全部保持不变。由于 251 不是 50 的倍数，当前 rLLM 只会在 50/100/150/200/250 保存，
不会自动保存 Step 251 的最终权重。正式训练前应加一个最小的“正常结束时保存最终
checkpoint”工程修复并做恢复检查；不能用 Step 250 冒充最终模型。

磁盘也需在正式运行前处理：当前 52 GiB 可用空间理论上能容纳 checkpoint 替换时的
临时双份，但安全余量不足。应先把四步 checkpoint 归档到外部存储并校验后删除，或再
增加至少 20–30 GiB 数据盘空间。远端 smoke 备份所在系统盘只有约 12 GiB 可用，不适合
继续堆放正式 checkpoint。

## 日志与停止状态

远端证据：

```text
/root/autodl-tmp/finqa/experiments/baseline64-20261109T0430Z
/root/autodl-tmp/finqa/experiments/baseline64-retry1-20261109T0440Z
/root/autodl-tmp/finqa/experiments/baseline64-restore-20261109
/root/autodl-tmp/finqa/outputs/qwen3-4b-a800-baseline64/2026-11-09_04-22-59
```

本地忽略归档为 `.cache/deploy/baseline64-artifacts-20261109.tar.gz`，SHA256
`7686c9015cb7960cbca02ef33dbc9725c8ca189e988e1b3079e0ebb5a29ffd19`。它包含失败、成功、
恢复三次运行和成功 Hydra output，不含 `.env`。远端 Git 工作区干净，GPU 空闲。

当前已经具备正式 GRPO 的训练正确性与单卡资源条件。正式启动前仍需解决最终 Step 251
checkpoint、磁盘安全余量，并取得 Judge 输入/输出 token 单价。完整训练未启动，等待
用户确认最终配置和预算。
