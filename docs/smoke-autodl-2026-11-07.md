# 首次单 A800 GRPO Smoke Test

日期：2026-11-07（Asia/Shanghai）。**唯一一次尝试在 Ray 初始化前置阶段失败，已停止。实际 Optimizer Step 为 0，没有重跑，没有启动正式 GRPO Baseline。** 最小启动修复已通过函数级验证，但没有用第二次训练尝试验证。

## Git 与环境快照

用户已接受 NumPy 2.3.5 的已记录元数据例外。本轮没有升级 verl/vLLM、修改第三方 METADATA、隐藏 pip check 结果或改变模型/算法/实验规模。

| 项目 | 本次值 |
| --- | --- |
| 部署补丁与 CUDA lock 提交 | `24577293f1a3d3f50e5dee965a25b5cd249fb17f` |
| 实验实际提交 | `159ecaab264a66424b000aabbf591c00ed68ccee`，增加单次监测与更新上限保护 |
| 服务器 | `autodl-container-66714bb469-f35807ff` |
| 工作目录 | `/root/autodl-tmp/FinQA-agentic-rl` |
| 实验目录 | `/root/autodl-tmp/finqa/experiments/smoke-20261106T202835Z` |
| GPU / 驱动 / Toolkit | A800 80GB PCIe × 1 / 590.48.01 / CUDA 13.0.88 |
| 模型 | Qwen3-4B-Instruct-2507，revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| CUDA lock SHA256 | `358dae28be539e93ef857abcafddc04b28e5dd68d13bf968676ebc1226b3f351` |

远端在启动前为干净 Git 工作区，所有跟踪文件检查未发现已配置 judge 凭证；`.env` 未跟踪、权限 600。部署提交内容与远端文件逐一核对。原有本地 README 和 `.vscode` 用户改动未混入实验提交。本轮提交保存在本地与服务器，未 push。

实验目录保存 `git-head.txt`、`git-status.txt`、`git-diff.patch`、`cuda-uv.lock`、完整 `configs/`、`versions.json` 和 `manifest.json`。实际模型路径、单步参数与 Hydra 解析结果另存 `hydra-output/effective-config.yaml`。失败后的修复不属于本次运行的 `159ecaa`，不能倒算成本次已使用修复。

| 依赖 | 实际版本 |
| --- | --- |
| Python / rLLM / gateway | 3.11.17 / 0.3.0rc0 / 0.1.0 |
| rLLM 与 gateway 来源 | 同一官方提交 `4e828c81680e5a85b638ceedeaf09d1e90109ca8` |
| torch / vLLM / verl | 2.11.0+cu130 / 0.22.1 / 0.8.0 |
| FlashAttention / NumPy | 2.8.3 / 2.3.5 |
| Transformers / PEFT / Ray | 5.22.2 / 0.21.2 / 2.60.0 |

`pip-check.log` 保留返回码 1 和唯一冲突：verl 声明 `numpy<2.0.0`，环境实际为 2.3.5。它是用户接受的已知例外；本次异常堆栈与 NumPy 无关。

## 执行范围与观测方式

保持 `configs/a800_smoke.yaml` 不变：2 条 train、2 条 val、每题 4 条 rollout、LoRA rank=alpha=32、单 GPU、PPO minibatch=2、PPO epochs=1、total_training_steps=1、rLLM total_batches=1、首步保存 checkpoint、关闭周期验证和训练前验证。

实际只启动一次 `bash scripts/gpu.sh smoke`，由 `scripts/run_smoke_audit.py` 启动并记录输出，超时上限 1,800 秒，没有重试循环。rLLM 固定实现按 `max(1, train_batch_size // ppo_mini_batch_size)` 计算本批更新数，这套参数预期为 1，并非简单将多轮 step 行数当成更新次数。

仅在本次命令树中启用的 `scripts/smoke_audit/sitecustomize.py` 增加观测：

- 用共享独占文件限制 FSDP `optimizer_step` 最多进入一次，防止 worker 重启或重复调用突破限制。
- 用优化器 pre/post hook 区分“进入更新路径”和“真实 optimizer.step 返回”。
- 记录可训练参数及 LoRA 更新前后摘要、变化张量数量、梯度有限性与范数。
- 记录 rollout token/logprob 长度、组奖励/优势、更新指标和 judge HTTP 用量，不记录请求密钥、headers 或 prompt。
- 每约一秒采样 nvidia-smi 和进程 RSS。诊断不改变 reward、advantage、loss、optimizer 参数或实验配置。

这些训练阶段的观测点本次没有被触发，不能将已加入 hook 当作训练验证已通过。LoRA 摘要在未来真正进入更新阶段时会有少量额外 CPU 内存和传输开销，应与纯训练吞吐区分。

## 失败位置与根因

作业开始于 2026-11-07 04:28:35 左右，总耗时 **45.407 秒**，非超时，返回码 **1**。数据预加载、配置校验及小型 CUDA/FlashAttention preflight 已通过，随后在 `trainer.train()` 调用的 Ray 环境准备阶段失败：

```text
rllm/trainer/verl/verl_launcher.py:126
  ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)
ray/_private/runtime_env/uv_runtime_env_hook.py:411
  if _is_path(working_dir):
TypeError: path_or_uri must be a string, got <class 'NoneType'>.
```

固定 rLLM 的 `get_ppo_ray_runtime_env()` 显式返回 `working_dir=None`。Ray 2.60.0 检测到上层 `uv run` 后，自动 uv hook 将该值交给只接受字符串的路径检查函数。因此异常发生在 Ray 集群和真实 FSDP worker 初始化之前。并非模型加载 OOM、NumPy 数值兼容问题、零优势或优化器异常。

## 六项验收结果

| 验收项 | 本次结论 |
| --- | --- |
| 真实 FSDP LoRA Trainer / 模型加载 | 未到达；只构造了外层 AgentTrainer/launcher |
| 每题 4 条 rollout、训练 Gateway token/logprob | 未产生 rollout，无法验收；不能复用此前独立推理的 2/2 结果代替 |
| Reward、Group Advantage、Loss、梯度 | 全部未计算，值为“未产生”，不是数值 0，也不是“零优势” |
| CUDA OOM | 未出现；只执行小型 preflight GPU 算子 |
| Optimizer Step、LoRA 更新、checkpoint | 0 次调用/返回；无更新 claim 文件；没有 `global_step_1` checkpoint |
| Judge 与 token | 0 次 judge HTTP 请求；0 次训练模型生成；因此两者本次生成 token 均为 0 |

`audit/` 只有进程启动和 hook 安装事件，没有训练 phase 事件、optimizer 调用事件或 judge 请求事件。堆栈停点与这些计数一致。不能将“没有梯度记录”解释为优化器收到零梯度。

## 资源与费用

| 指标 | 实测值与限制 |
| --- | --- |
| nvidia-smi 显存最大采样值 | **537 MiB**，属于 preflight；不是模型训练显存峰值 |
| 进程树 RSS 总和最大采样值 | **1,565,745,152 bytes，约 1.46 GiB**；共享页可能重复计数，不等同唯一物理内存 |
| 采样数 | 44；其中 1 个采样点 GPU utilization >0，不能据此精确推算 kernel 活跃秒数 |
| 容器内存峰值 | 本次未完整取得：初版采样器只处理 cgroup v2，服务器实际为 v1 |
| 内存后验检查 | v1 `memory.failcnt=0`；后验匿名内存约 0.95 GiB，页缓存约 21.98 GiB，这些不是本次阶段峰值 |
| 结束状态 | GPU 0 MiB，无本次遗留 raylet/gcs_server 或训练进程 |

失败后修正了 cgroup v1/v2 识别，补充容器总用量、匿名内存、页缓存及 OOM 计数；实际 v1 读取和离线测试均通过。没有借修复之机重跑作业，因此本次缺失的完整内存时间序列无法补造。

按用户确认的 **¥5.59/小时**：作业窗口为 **0.012613 GPU 小时，约 ¥0.0705**。这是单卡在这 45.41 秒内的租用成本估计，不是 CUDA 活跃时间账单，也不包括准备、诊断、等待或服务器继续开机的费用。未关闭租赁实例，平台仍会按开机时间计费。Judge 本次未调用，没有新增 judge 请求费用。

本次没有完成 rollout 或一次更新，**无法可靠外推正式训练总时长与总价**。可用费用公式为 `GPU 费用 = 5.59 × 实际租用小时`，即约 ¥0.0932/分钟；1 / 10 / 24 小时分别 ¥5.59 / ¥55.90 / ¥134.16。一个 epoch 的名义数据量为 4,030 × 4 = 16,120 条 rollout，实际还受最后 batch 处理、验证和失败重试影响。需要成功的单步数据才能估计生成、反向、权重同步、checkpoint 与 judge 的分项耗时。Judge 单价尚未提供，后续应按实测输入/输出 token 分别计费，不能用本次零调用推断正式训练 API 成本为零。

## 已修复但未重跑的内容

`scripts/gpu.sh` 增加 `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`，使用本机已锁定的 uv Python 环境，避免 Ray 再自动重写 uv runtime。该开关是 [Ray 官方提供的兼容方式](https://github.com/ray-project/ray/issues/54344)，已安装版本源码也明确支持它。没有升级依赖、修改第三方源码或改动 `working_dir` 去上传整个项目。

在服务器仅调用实际 `worker._maybe_modify_runtime_env()`，确认关闭开关后 `working_dir=None` 可正常通过，保存 `compatibility-check.json`。检查没有启动 Ray 集群、加载模型、调用 judge 或再次运行训练。它证明当前报错点已修复，不证明之后的 FSDP、rollout、权重同步或 optimizer 路径一定成功。

首次运行前本地回归为 70 项通过。失败后增加 cgroup v1/v2 测试，完整离线回归 **71 项通过**（13.06 秒），Ruff lint/format、shell 语法和 Git diff 检查通过。单步保护、参数变化、judge 计数与凭证隔离、cgroup 测试均通过。核心 YAML 和 CUDA lock 未修改。

## 日志位置与后续边界

远端实验根目录：

```text
/root/autodl-tmp/finqa/experiments/smoke-20261106T202835Z/
```

- `console.log`：完整异常链；`result.json`、`summary.json`：退出码、计数、资源摘要和费用。
- `manifest.json`、`versions.json`、`git-*`、`configs/`、`cuda-uv.lock`：本次实际来源和配置快照。
- `resources.jsonl`：44 个采样点；`audit/events-*.jsonl`：分进程诊断事件。
- `hydra-output/`：原始 Hydra 配置与 `gpu-preflight.json`，对应 `/root/autodl-tmp/finqa/outputs/qwen3-4b-a800-smoke/2026-11-07_04-29-17/`。
- `pip-check.log`：未隐藏的 NumPy 元数据例外；`compatibility-check.json`：失败后的非训练修复验证。

本地备份位于忽略目录 `.cache/deploy/smoke-20261106T202835Z/`。本轮只进行了一次失败尝试，没有重试训练，也没有开始正式 Baseline。后续若再次授权，应使用新的实验目录，保留本轮失败证据；在真正完成单步之前，训练六项验收仍不能标为通过。
