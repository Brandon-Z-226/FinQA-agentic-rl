# 单 A800 GRPO Smoke Test：第二次授权尝试

日期：2026-11-09（Asia/Shanghai）。用户明确授权“再试一次，仍限 1 步”。**本次成功加载真实 FSDP LoRA 模型并完成 8 条 rollout，随后在 DataProto → TensorDict 转换处失败。真实 Optimizer Step 为 0，任务已停止，没有第三次尝试或正式 Baseline。**

## 实际运行来源

| 项目 | 值 |
| --- | --- |
| 实际提交 | `af1c5591071caffb262ecf7579f1e368a13b23c7`，远端启动前工作区干净 |
| 服务器 / 项目 | `autodl-container-66714bb469-f35807ff` / `/root/autodl-tmp/FinQA-agentic-rl` |
| GPU / 驱动 / Toolkit | NVIDIA A800 80GB PCIe × 1 / 590.48.01 / CUDA 13.0.88 |
| CPU / 容器内存上限 | 14 核 / cgroup v1，120 GiB |
| 启动时间 | 2026-11-09 02:26:53，UTC 2026-11-08 18:26:53 |
| 实验目录 | `/root/autodl-tmp/finqa/experiments/smoke-20261108T182653Z` |
| 模型 | Qwen3-4B-Instruct-2507，revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| 运行时 CUDA lock SHA256 | `358dae28be539e93ef857abcafddc04b28e5dd68d13bf968676ebc1226b3f351` |

部署补丁和 CUDA lock 先保存为 `2457729`，单步监测为 `159ecaa`，首轮 Ray 修复及失败记录为本次使用的 `af1c559`。这些提交已保存于本地与服务器，尚未 push。用户原有本地 README 和 `.vscode` 改动未纳入实验提交。启动前检查跟踪文件无已配置 judge 凭证，`.env` 未跟踪、权限 600；归档前再次检查实验文件及跟踪文件，没有匹配到已知 judge 凭证或 Ray auth token。

实际依赖为 Python 3.11.17、rLLM 0.3.0rc0、gateway 0.1.0（两者同一来源 `4e828c81680e5a85b638ceedeaf09d1e90109ca8`）、PyTorch 2.11.0+cu130、vLLM 0.22.1、verl 0.8.0、FlashAttention 2.8.3、NumPy 2.3.5、Transformers 5.22.2、PEFT 0.21.2、Ray 2.60.0、**TensorDict 0.8.3**。TensorDict 初版 `versions.json` 未列入，但保存在本次冻结的 lock，失败后也核对了实际安装版本；后续版本采集已补此项。

用户接受的 NumPy 元数据例外原样保留：`uv pip check` 返回非零，唯一冲突仍为 verl 声明 `numpy<2.0.0`，实际安装 2.3.5。本次异常与 NumPy 无关，没有升级 verl/vLLM 或修改第三方包元数据。

## 执行与六项验收

复用 `scripts/run_smoke_audit.py`，只启动一次现有 `bash scripts/gpu.sh smoke`。核心 YAML 不变：2 条 train / 2 条 val、每题 4 条 rollout、rank=alpha=32、单 GPU、PPO minibatch=2、PPO epochs=1、total_training_steps=1、rLLM total_batches=1、save_freq=1、训练前及周期验证关闭。诊断保留跨进程独占的一步上限保护；没有修改奖励、算法或模型参数。

| 验收项 | 结果与边界 |
| --- | --- |
| 真实 FSDP LoRA Trainer / 模型加载 | **通过初始化阶段**：Ray、真实 FSDP worker、PEFT 模型及 398 项权重加载完成，训练 vLLM 成功启动。优化器更新尚未验证 |
| 4 条 rollout / question | **两题各 4 条，共 8 条正常结束**，均 `ENV_DONE`、reward=1，共 37 次模型生成调用 |
| Gateway Token IDs / Logprobs | traces 接口返回 HTTP 200；原审计只在优势计算后记录长度，本次提前失败，**没有保存可核对的 token/logprob 数组，尚不能验收对齐** |
| Reward / Advantage / Loss / 梯度 | reward 已产生；两组奖励均 `[1,1,1,1]`，按标准 GRPO 预计组内优势为零，但实际尚未执行优势计算、loss 或 backward，不能将缺失指标写成零或归因于优化器 |
| CUDA OOM / CPU OOM | 未发生；GPU 最大采样 58,048 MiB，cgroup memory.failcnt 全程 0。尚未覆盖反向和保存时的峰值 |
| Optimizer / LoRA 更新 / checkpoint | **0 次**调用和返回，无 optimizer claim，无 `global_step_1` checkpoint；权重更新和 checkpoint 元数据均未验收 |

此前 meta LoRA 检查得到 66,060,288 个可训练参数，占 4,477,484,544 的 1.4754%。本次真实 worker 加载了 PEFT/FSDP 模型，但原审计在 optimizer 入口才采集可训练张量清单，因此不能把之前的 meta 数量冒充本次更新参数实测。

日志中的两组 rollout 如下。时间为上游日志的取整值，各 rollout 并行执行，不能相加当作墙钟时间。

| Question ID 前缀 | rollout 序号 | 模型调用数 | rollout 秒数 | judge 秒数 | reward |
| --- | --- | --- | --- | --- | --- |
| 0435523c | 3 / 2 / 0 / 1 | 5 / 4 / 5 / 5 | 20 / 20 / 26 / 29 | 2 / 3 / 3 / 3 | 全部 1 |
| 841b3a34 | 1 / 3 / 0 / 2 | 4 / 4 / 5 / 5 | 24 / 25 / 31 / 31 | 4 / 4 / 4 / 4 | 全部 1 |

这是两题链路测试，8/8 不代表 FinQA 全集成功率。组内奖励相同是本批的实际观察；没有为了获得非零优势而更换问题、奖励或算法。

## 失败原因与最小修复

首次 smoke 的 Ray uv hook 修复本次已在真实启动中验证通过。本次停在 rollout 完成后的批次转换：

```text
rllm/trainer/verl/verl_backend.py:624, process_backend_batch
    batch_td = batch.to_tensordict()
verl/protocol.py:1108, to_tensordict
    assert parse_version(tensordict.__version__) >= parse_version("0.10")
AssertionError: Convert DataProto to TensorDict at least requires tensordict version 0.10
```

固定 verl 0.8.0 的依赖声明允许 `tensordict>=0.8.0,<=0.10.0,!=0.9.0`，旧 lock 解析为 0.8.3；但当前 EngineWorker 路径调用的转换接口实际要求至少 0.10。依据 [verl 0.8.0 protocol 源码](https://github.com/verl-project/verl/blob/v0.8.0/verl/protocol.py) 和已安装源码，CUDA 环境新增直接固定 `tensordict==0.10.0`。没有改动第三方源码、verl/vLLM、核心 YAML 或 NumPy。

失败后只进行环境修复和小型 CPU 验证，**未重启 Ray、模型、rollout 或 optimizer**。实际 `DataProto.from_dict(...).to_tensordict()` 成功转换 2×2 token 张量和非张量 uid，张量内容保持一致，保存为 `postfix-compatibility.json`。训练入口的 preflight 同时补充 TensorDict 版本和真实转换检查，后续可在昂贵的模型加载前发现这个问题。

新 lock SHA256 为 `f0811e1ced035ab22fce25706e60857745e0698ab574b6f0d142a938cf3dcb24`。实际 lock/sync 变更完整保留：TensorDict 0.8.3 → 0.10.0，增加 pyvers 0.1.0，packaging 26.3 → 25.0，virtualenv 21.15.6 → 21.14.5。pyvers 要求 packaging >=25,<26，解析器相应选择兼容版本。核心 CUDA/训练框架版本不变，修复后 `pip check` 仍仅报告既有 NumPy 例外。运行时旧 lock 和失败后的新 lock 分开存档，不能把新版本倒算为本次实际使用版本。

## Judge 与 token 记录缺口

原 HTTP 审计事件为零，但日志明确存在 **8 次完成的逻辑 judge 评估**，正奖励也不能来自异常 fallback。因此“0 个审计事件”不是“未调用 API”。当前无法恢复准确 HTTP 尝试次数（SDK 重试可能增加次数），也没有得到 judge 输入/输出 token 用量；训练模型 token 总量同样未保存，不填 0 或估造数值。

一个待验证原因是 Ray worker 重建 evaluator 状态时没有对应的 `FINQA_JUDGE_BASE_URL` 环境变量，HTTP hook 因而无法匹配 endpoint。失败后仅向诊断 runtime env 显式传递不含凭证的公开 endpoint，增加 hook 安装状态记录；没有传播 API key、请求 headers 或完整 prompt。该防护通过 mock 测试，**尚未在下一次真实运行中验证是否彻底解决漏计**。

另将 token ID/logprob 审计保存点提前到 `generate_episodes` 返回后、批次转换之前，避免下游失败再次丢失生成证据。保存内容仅供后续新实验，无法补造本次已结束进程的数据。新增单测确认：即使后续批次转换抛错，rollout token/logprob 仍已保存，且不改变方法返回值。

## 资源、用时与费用

总墙钟 **208.983 秒**，返回码 1，非超时。rollout 阶段 **31.166 秒**；此前约 174.47 秒包括 preflight、数据预加载、Ray、FSDP 和训练 vLLM 初始化。初始化事件中的 `trainer_init` 仅包装 Python 构造函数，不能用其短耗时代表整个模型加载阶段。

监测共 202 个约一秒间隔采样点，下表是观测最大值，不是精确瞬时峰值。容器内存包括同容器其他活动和文件缓存，各列最大值不保证发生于同一时刻。

| 阶段 | GPU 显存最大采样 | 容器内存最大采样 | 匿名内存最大采样 |
| --- | --- | --- | --- |
| rollout 前：preflight / 初始化 / 加载 | 57,960 MiB（56.60 GiB） | 44.41 GiB | 23.89 GiB |
| rollout | 58,048 MiB（56.69 GiB） | 29.51 GiB | 8.99 GiB |
| rollout 后：转换失败及退出 | 17,940 MiB（17.52 GiB） | 29.59 GiB | 9.07 GiB |

整体页缓存最高约 20.51 GiB；进程树 RSS 总和最高 48.38 GiB，会重复计入共享页，不能替代容器内存。FSDP 日志另报告加载后 allocated=15.27 GB、reserved=17.55 GB、device used=19.13 GB，这只是该时刻，并非整次最大值。采样中 GPU utilization>0 的区间约 63.36 秒，仅是粗略活跃时间估计，不是 CUDA kernel profiler 测量。全程 memory.failcnt=0，没有 CUDA OOM 堆栈。

按用户确认 **¥5.59/小时**，这次作业窗口为 **0.058051 GPU 小时，约 ¥0.3245**。加上首轮失败的 ¥0.0705，两次作业窗口合计约 ¥0.3950。费用不含安装、调试、等待及服务器继续开机时间；没有关闭租赁实例，平台仍按开机时长计费。Judge 有实际调用，但 token 和单价缺失，无法估算其费用。

目前不能可靠外推正式训练成本：本次未覆盖 backward、optimizer、权重同步和 checkpoint，只有两个样本的生成吞吐。一个 epoch 名义上为 4,030×4=16,120 条 rollout，另需考虑最后 batch、验证和失败处理。后续可用成功步骤的分阶段时间估计总 GPU 小时，再乘 5.59；10 小时约 ¥55.90、24 小时约 ¥134.16。Judge 费用应按实际输入/输出 token 和对应单价另计。

## 修复验证、归档与下一步

修复后本地完整离线测试 **74 passed**（13.83 秒），6,923 条原始 pandas 弃用警告未隐藏；Ruff lint/format、shell 语法、Git diff 检查通过。真实 Hydra 合成及固定 rLLM sync_config 对两个单卡 profile 均通过。新增回归覆盖提前保存 rollout、公开 endpoint 传递与无原 judge 环境变量的用量计数；原有一次 optimizer 限制和参数变化测试继续通过。这些 CPU/mock 测试不替代真实单步训练验收。

远端实验目录包含：

- `console.log`、`result.json`、`summary.json`：完整堆栈、退出状态、阶段与费用摘要。
- `manifest.json`、`versions.json`、`git-*`、`configs/`、`cuda-uv.lock`：本次实际 Git / 依赖 / 配置快照。
- `resources.jsonl`、`memory-before.json`、`memory-after.json`、`audit/events-*.jsonl`：资源和阶段事件。
- `hydra-output/`：来自 `outputs/qwen3-4b-a800-smoke/2026-11-09_02-27-35` 的配置及 preflight。
- `tensordict-fix-lock.log`、`tensordict-fix-sync.log`、`postfix-cuda-uv.lock`、`postfix-compatibility.json`、`postfix-pip-check.log`：失败后修复的独立证据。
- `secret-scan.json`：已知凭证检查结果，不含凭证值。

本地副本为忽略目录 `.cache/deploy/smoke-20261108T182653Z/`。运行进程已经退出，检查 GPU 显存回到 **0 MiB**，没有遗留 Ray 训练进程，没有 checkpoint。

当前具备继续进行受控 smoke 的基础，已修复本次明确的接口版本错误；但完整训练链路仍未通过。下一次仍须单独授权最多一个 optimizer step，重点检查 token/logprob 对齐、实际优势、loss/梯度、优化器调用、LoRA 变化及 checkpoint。相同奖励可能导致零优势，应如实记录，不能为追求非零指标擅改实验。**本轮到此停止，不启动第三次 smoke 或正式 GRPO Baseline。**
