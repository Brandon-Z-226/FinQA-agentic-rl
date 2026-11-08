# 单 A800 GRPO 单步闭环成功

日期：2026-11-09（Asia/Shanghai）。本次沿用 `a800_smoke.yaml`，真实执行并完成了一个 Optimizer Step，随后正常退出。没有启动完整 GRPO Baseline。

## 运行来源与边界

| 项目 | 实际值 |
| --- | --- |
| Git 提交 | `a64ad1aede6aa57fec9cbfdfaf2726cb2778060b`，远端启动前工作区干净 |
| 服务器 | `autodl-container-66714bb469-f35807ff`，NVIDIA A800 80GB PCIe × 1 |
| 驱动 / CUDA Toolkit | 590.48.01 / 13.0.88 |
| 实验目录 | `/root/autodl-tmp/finqa/experiments/smoke-20261108T185116Z` |
| 模型 | Qwen3-4B-Instruct-2507，revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| CUDA lock SHA256 | `f0811e1ced035ab22fce25706e60857745e0698ab574b6f0d142a938cf3dcb24` |
| 关键版本 | torch 2.11.0+cu130、vLLM 0.22.1、verl 0.8.0、FlashAttention 2.8.3、TensorDict 0.10.0、NumPy 2.3.5 |

用户接受的 NumPy 元数据例外原样保留：`uv pip check` 仍只报告 verl 声明 `numpy<2.0.0` 与实际 NumPy 2.3.5 的冲突。本次没有升级 verl/vLLM、修改第三方元数据、改变 Reward、GRPO 算法或核心实验参数。

配置仍为 2 个 train 问题、每题 4 条 rollout、Qwen3-4B、LoRA rank=alpha=32、PPO epochs=1、单 GPU、单个总训练 step、step 1 保存 checkpoint、关闭训练前和周期验证。跨进程独占文件将实际 Optimizer Step 硬限制为最多一次。

## 训练闭环结果

作业返回码 0，非超时，总墙钟 **271.554 秒**。TensorDict 0.10.0 修复后，完整路径为：

```text
FSDP/PEFT 加载 → 8 条 rollout → Gateway traces → TensorDict 转换
→ old logprobs → GRPO advantage → backward → optimizer.step
→ rollout 权重同步 → checkpoint → 正常退出
```

两题各 4 条 rollout，8/8 均 `ENV_DONE`、reward=1、table access reward=1。共得到 33 个 Agent step，轨迹平均 4.125 step；本批 Agent 共发起 33 次模型调用。8 条轨迹的训练数据汇总如下：

| 指标 | 实测值 |
| --- | --- |
| Prompt token IDs | 61,660 |
| Response token IDs | 7,366 |
| 总训练 token | 69,026 |
| Response IDs 与 rollout logprobs | 8/8 轨迹、33/33 step 长度一致 |
| old logprob | 成功计算，9.988 秒 |
| rollout / actor 概率相关系数 | 0.99080 |
| rollout / actor log-PPL 差 | 0.00123 |
| actor entropy | 0.099998 |

两组 reward 都是 `[1,1,1,1]`，因此标准 GRPO 实际计算出的 step advantage 全为 0，`advantage mean/std/max/min=0`、`fraction_zero=1.0`。没有修改 reward 或算法来制造梯度。

训练仍然真实进入 backward 与 optimizer：

| 指标 | 实测值与解释 |
| --- | --- |
| LoRA 可训练参数 | 66,060,288，共 504 个 FSDP 参数张量 |
| 梯度有限性 | 全部有限，没有 NaN/Inf |
| 非零梯度张量 | 0，符合本批零优势 |
| loss / pg_loss / grad_norm | 0 / 0 / 0 |
| Optimizer Step | 调用 1 次、正常返回 1 次；没有第二次调用 |
| 权重变化 | 0 个张量变化，更新前后 SHA256 相同；这是零优势结果，不是 optimizer 未运行 |
| lr / MFU | `1e-6` / 0.3828 |

`update_actor` 用时 19.766 秒，优化器调用前后的监测确认真实 `optimizer.step()` 返回。CUDA peak allocated/reserved 分别为 34.31/36.17 GiB；它们是 actor worker 统计，不包含同卡 vLLM 的全部显存。

## Checkpoint

checkpoint 保存成功并由仓库入口自动验证：

```text
/root/autodl-tmp/finqa/checkpoints/qwen3-4b-a800-smoke/global_step_1/
```

目录约 **18 GiB**，包含：

- `actor/model_world_size_1_rank_0.pt`：17,910,768,163 bytes；
- `actor/optim_world_size_1_rank_0.pt`：528,916,075 bytes；
- `actor/extra_state_world_size_1_rank_0.pt`、tokenizer/config、`data.pt`；
- `actor/lora_train_meta.json`：`r=32`、`lora_alpha=32`、`task_type=CAUSAL_LM`。

保存用时 24.003 秒，随后 rollout 权重同步用时 14.349 秒。此次只验证“可保存且结构/元数据正确”，尚未执行完整 checkpoint 恢复训练测试。

## 资源与费用

监测得到 262 个约一秒采样点：

| 指标 | 实测值 |
| --- | --- |
| GPU 显存最高采样 | **60,024 MiB，58.62 GiB** |
| GPU utilization 最高 | 100% |
| 容器内存最高采样 | **63.20 GiB** |
| 匿名内存 / 页缓存最高采样 | 24.61 / 38.58 GiB |
| 进程树 RSS 和最高采样 | 50.04 GiB；共享页可能重复计入 |
| cgroup v1 memory.failcnt | 全程 0 |
| 结束状态 | GPU 0 MiB，无 raylet/gcs_server 或训练进程 |

没有 CUDA OOM 或 CPU cgroup OOM。verl 的 `perf/cpu_memory_used_gb=129.57` 来自主机口径，超过容器的 120 GiB 限额，不能当作容器实占；本报告采用 cgroup v1 的 63.20 GiB 采样峰值。

按用户确认的 **¥5.59/小时**，本次 0.075432 GPU 小时约 **¥0.4217**。三次 smoke 作业窗口合计约 **¥0.8167**，不含安装、诊断、空闲等待及服务器继续开机时间。

## Judge 记录限制

8 条正奖励轨迹证明完成了 **8 次逻辑 judge 评估**；evaluator 平均耗时 3.049 秒。原有底层 HTTP hook 本次仍漏记，因此无法恢复精确 HTTP 重试次数和 judge 输入/输出 token，不将它们写成 0。用户也尚未提供 judge token 单价，所以 API 费用未估造。

失败后的监测修复不属于本次实际运行：审计已增加同步 OpenAI SDK `chat.completions.create` 的逻辑请求与 response usage 记录，不保存 prompt、header 或 API key。mock 回归确认 token usage 可记录且私密 prompt 不会写入日志；没有为验证监测而额外调用付费 API，也没有重跑训练。

## 正式 Baseline 时间与成本估算

正式配置对 4,030 个 train 样本采用 batch=16、drop_last，一轮为 **251 个训练 step**，实际训练 4,016 个问题、16,064 条 rollout。`test_freq=20` 会在 step 20…240 做 12 次验证，训练结束再做一次，共 13 次；val split 为 522 条、`n_val=1`，即另有 6,786 条验证 rollout。预计逻辑 judge 评估合计 **22,850 次**。

本次两题较容易，平均 4.125 个 Agent step，不能代表全数据。仅按本次相同长度和并发做线性外推：

- 初始化约 2.89 分钟；
- 正式 batch 的 64 条 rollout 在并发 8 下约 8 个波次；rollout、old logprob、actor update 按 8 倍数据量外推，并加每步约 14.35 秒权重同步，约 **7.64 分钟/训练 step**；
- 251 个训练 step 及 step 100/200 两次 checkpoint 约 **32.0 小时**；
- 13 次完整 val 约 **6.1 小时**；
- 中心估计约 **38 小时、¥213 GPU 租金**。

这个中心值不包含 judge API 费用，也没有覆盖难题出现更多工具调用、20-turn 上限、失败重试、网络波动或正式 batch 的 CPU/显存变化。合理规划应预留 **30–50 小时、约 ¥168–¥280 GPU 租金**；首次小规模 Baseline 应重新测量多个非零优势 batch 后再收紧估计。若不需要如此频繁的全量验证，改变 `test_freq` 会显著影响成本，但那属于实验配置变更，本轮没有调整。

## 是否具备 Baseline 条件

训练软件闭环已经跑通，具备进入下一阶段小规模标准 GRPO Baseline 的条件：FSDP LoRA、rollout、token/logprob、GRPO advantage、backward、optimizer、权重同步和 checkpoint 均已到达。零权重变化由本批零优势解释，并非链路失败。

完整一轮前还有两个工程条件：

1. 当前数据盘在 smoke checkpoint 后剩余约 52 GiB。正式 checkpoint 每份约 18 GiB，且正式实验目录与 smoke 目录分别保留；step 200 替换 checkpoint 时可能出现临时双份。开始完整训练前应先归档并清理 smoke checkpoint，或扩容至少 20–30 GiB。
2. 新的 SDK 层 judge usage 审计应在小规模 Baseline 中确认实际生效，以便估算 API 账单；这不影响训练正确性。

远端完整日志位于 `/root/autodl-tmp/finqa/experiments/smoke-20261108T185116Z/`，本地忽略目录备份为 `.cache/deploy/smoke-20261108T185116Z/`。其中保存 `console.log`、`summary.json`、`result.json`、`resources.jsonl`、审计事件、rollout token/logprob、冻结配置/lock/Git 状态、Hydra output、checkpoint 路径与凭证扫描结果。扫描 124 个文件，没有匹配到已知凭证；`.env` 未跟踪且权限 600。

本轮已停止，未启动完整训练。
