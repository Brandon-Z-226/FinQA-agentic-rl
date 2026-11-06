# 单张 A800 80GB 部署与验收

这是独立的 Linux x86_64 / Python 3.11 uv 环境，与根目录的 macOS `.venv` 分离。2026-11-07 已在 AutoDL A800 上安装并锁定依赖，通过 GPU 算子、LoRA 参数、数据/配置检查与两条真实 Agent 推理；**尚未执行训练或 Optimizer Step**。`uv pip check` 仍有固定 verl/vLLM 组合的 NumPy 元数据冲突，不能称为完整依赖验收通过。实测版本、网络方案、日志和限制见 [AutoDL 部署记录](../../docs/deployment-autodl.md)。根项目锁文件不能代替 CUDA lock。

## 机器与 CUDA 版本

原租机截图为 A800 80GB × 1、14 核 CPU、120GB RAM、驱动 580.65.06、30GB 系统盘和 50GB 数据盘；实际服务器为驱动 590.48.01、CUDA Toolkit 13.0.88、90 GiB 数据盘。LoRA、CPU offload 和有限 rollout 并发是本配置的起点，训练峰值显存和速度必须通过 smoke 测量。4B 全量训练并非理论上绝对不可能，但不能只按 BF16 权重约 8GB 计算，还要包含梯度、Adam 状态、激活、vLLM 和权重同步峰值。

固定 rLLM 的 extras 指定 `verl==0.8.0`、`vllm==0.22.1`、`flash-attn==2.8.3`。vLLM 0.22.1 的 [CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.22.1/requirements/cuda.txt) 指定 PyTorch 2.11.0 / torchaudio 2.11.0 / torchvision 0.26.0，其 [Dockerfile](https://github.com/vllm-project/vllm/blob/v0.22.1/docker/Dockerfile) 使用 CUDA 13.0.2。本环境选择 cu130 PyTorch 索引和 CUDA 13.0 开发镜像，不任意混装最新 torch 或 CUDA 12.x wheel。

租机时选 **Linux x86_64、CUDA 13.0 devel（含 nvcc）、Python 3、git、g++、make、uv** 的镜像。脚本不修改宿主驱动或系统 Python，不依赖预装 torch，只创建数据盘上的 uv 环境。`flash-attn` 可能需要编译，默认 `MAX_JOBS=4` 控制内存，`FLASH_ATTN_CUDA_ARCHS=80` 仅构建 A800 架构；本次确认配额后使用 MAX_JOBS=10 编译成功。平台“CUDA ≤13.0”表示驱动能力，不表示镜像已安装 nvcc。[CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/) 列出 580.65.06 驱动；具体 wheel 与算子兼容性仍由 `check` 实测。

**磁盘需要提前确认。** 50GB 要同时容纳环境/构建缓存、约 8GB 模型、输出和 checkpoint，余量很紧。此 verl 版本保存可恢复的 FSDP checkpoint 时包含模型状态，并非只有小型 adapter，FP32 基座状态可能约 16GB。配置每个实验最多保留一个 actor checkpoint，但 smoke 和正式训练的目录各自保留一个，安装/保存也可能需要临时空间。建议选择可提供约 80–100GB 可用数据空间的方案；截图的扩容上限若确实只有 9GB，应先向平台确认更大磁盘或挂载方案。这是容量规划建议，不是实测峰值。

## 环境、模型和数据

在 GPU 机器 clone 本仓库后，从仓库根目录执行。`FINQA_GPU_ROOT` 必须指向实际数据盘，下面只是路径示例。将已验证的 judge 配置安全复制到服务器 `.env`，不要提交密钥。

```bash
export FINQA_GPU_ROOT=/root/autodl-tmp/finqa
export CUDA_VISIBLE_DEVICES=0

bash scripts/gpu.sh setup
bash scripts/gpu.sh download
bash scripts/gpu.sh prepare
bash scripts/gpu.sh check
bash scripts/gpu.sh lora-check
bash scripts/gpu.sh config > "$FINQA_GPU_ROOT/checks/composed-config.yaml"
```

`setup` 检查平台、nvcc、构建工具和至少 40GiB 安装前空闲空间，再生成 `environments/cuda/uv.lock` 并用 `--locked` 同步和执行 `uv pip check`。首次成功安装后保存并提交 CUDA lock，同时记录镜像名称/digest。依赖解析或编译失败时保留错误，修订环境定义，不绕过检查启动训练。

**当前已知例外：** 依赖安装与运行时检查成功，但 `setup` 的最后一步仍报告 `verl requires numpy<2.0.0, but 2.3.5 is installed` 并返回非零。固定 rLLM 的 override 与新版 vLLM 所需 NumPy 2 存在这一元数据差异；本仓库只补上 `<2.4` 上界，没有篡改第三方声明或关闭 `pip check`。详见部署记录。不能用 `|| true` 将它视为安装验收成功，也不要因此直接启动训练。

`download` 按 `model.json` 固定下载 `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`，直接使用 HF snapshot，避免第二份模型副本。默认路径是：

```text
$FINQA_GPU_ROOT/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
```

已有权重可 `export FINQA_MODEL_PATH=/absolute/path/to/model`，自行确认 revision 并记入实验记录。`download` 始终下载清单中的固定版本。`prepare` 复用现有数据 SHA256 校验和三 split 注册逻辑，支持 `--offline` / `--archive`。不要把 `FINQA_SKIP_PRELOAD=1` 用于真实评测或训练；`config` 内部只在配置检查时设置它。

大文件放在数据盘的 `venv/`、`uv-cache/`、`python/`、`huggingface/`、`data/`、`rllm/`、`tmp/`、`outputs/`、`checkpoints/`。uv cache 与环境在同盘复用硬链接。Ray 默认使用短路径 `/tmp/finqa-ray`，避免 Unix socket 路径过长；可将 `RAY_TMPDIR` 设为数据盘上的短路径，并监控日志占用。

`check` 导入实际 CUDA 栈，检查包版本、单卡可见性、BF16 和显存容量，执行小型 CUDA 矩阵运算与 FlashAttention 前向/反向。安装、下载后至少要保留 24GiB 空间，训练入口再次检查。该阈值用于尽早发现容量不足，不能保证整个 epoch 不会写满。

`lora-check` 在 meta device 构建同一架构并应用 PEFT，不分配完整权重，确认可训练参数只有 LoRA 且比例低于 5%。这验证参数选择；真正 FSDP worker 是否启用 LoRA，需由训练和 checkpoint 验证。

## 两条评测与一步训练

先在一个终端运行独立 vLLM，仅监听 localhost；另一终端使用相同 `FINQA_GPU_ROOT`：

```bash
# 终端 A
bash scripts/gpu.sh serve

# 终端 B：2 条 val 样本，temperature=0.6，max_tokens=4096
bash scripts/gpu.sh eval
```

检查真实工具调用、SQL、FINAL ANSWER、judge、episode 和 gateway 数据。正常退出不代表 judge 无错误：上游 evaluator 在调用异常时返回 0，必须查看日志，不能把网络失败当作模型答错。本地 judge 三条合成验证不能代替云端连通性和真实 Agent 评测。

完成后用 Ctrl-C 停止终端 A 的独立 vLLM，再启动训练。rLLM/verl 自己管理训练用的同卡 vLLM，不能留下额外服务占用显存。`pipefail` 保留训练失败的退出码：

```bash
set -o pipefail
mkdir -p "$FINQA_GPU_ROOT/logs"
bash scripts/gpu.sh smoke 2>&1 | tee "$FINQA_GPU_ROOT/logs/smoke.log"
```

smoke 取 2 条 train / 2 条 val，group size 仍为 4，执行一个训练 batch，关闭验证、在第 1 step 保存 checkpoint。训练前执行配置/凭证/模型/磁盘/CUDA 检查，保存 `effective-config.yaml` 和 `gpu-preflight.json`。训练返回后检查 `global_step_1/actor/lora_train_meta.json` 的 rank/alpha 均为 32；缺少文件或参数不匹配会报错。

验收还需核实有限的 loss、梯度/优化器更新、rollout 权重同步、token/logprob 对齐和 checkpoint 可恢复性。若组内奖励全相同，优势可能为零；smoke 验证执行链路，不能证明模型已学到东西。断点恢复要另测，不能仅凭保存成功认定可恢复。

## 正式训练与记录

smoke 通过并处理其 checkpoint 的磁盘占用后：

```bash
bash scripts/gpu.sh train 2>&1 | tee "$FINQA_GPU_ROOT/logs/train.log"
```

| 参数 | 原始 Cookbook | 单 A800 配置 |
| --- | --- | --- |
| 模型 / LoRA / group | 4B / nested rank=alpha=32 / 4 | 同模型固定 revision；flat 与 nested 均 32 / 4 |
| GPU / TP | 8 / 1 | 1 / 1，actor 与 rollout 共卡 |
| train batch / PPO minibatch | 16 / 64 | 16 / 64，动态 batch |
| val batch / workflow 并发 | 64（rLLM 默认另有来源）/ 256 | 两个命名空间均 8 / 8 |
| prompt / response / model 上限 | 8192 / 4096 / 16384 | 相同 |
| vLLM 显存比例 / eager / max_num_seqs | 0.8 / false / 默认 | 0.5 / true / 8 |
| lr / epochs / temperature | 1e-6 / 1 / 0.6 | 相同 |
| param / optimizer offload | true / true | true / true |
| checkpoint 保留 | 默认不限 | 每个实验最多 1 个 actor checkpoint |

`rollout.mode=async` 是 vLLM 服务模式；`rllm.async_training.enable=false` 才是这里的共卡同步训练设置。不要开启需要额外 rollout 资源的分离训练。LoRA、gradient checkpointing、BF16 混合精度降低占用，但基座参数仍采用 FSDP 默认 FP32。

OOM 时先保留错误和显存记录，再降低执行并发，如 `rllm.workflow.n_parallel_tasks=4 actor_rollout_ref.rollout.max_num_seqs=4`。不要悄悄改变 group 或上下文长度；必要的预算变化另存实验配置。更低的 `gpu_memory_utilization` 可能导致 KV cache 不足，不能无限调低。

共享参数显式映射到 rLLM 命名空间，避免默认值覆盖配置。CLI 覆盖项传在最后；恢复可用 `trainer.resume_mode=auto`，新实验使用不同名称。`configs/a800_common.yaml` 是共同参数，`a800_80gb.yaml` 和 `a800_smoke.yaml` 是两个入口。

保留 console/UI 日志、resolved config、模型/数据 revision、CUDA lock、驱动/镜像信息、checkpoint，记录 Reward、Loss、成功率、token、GPU 小时与 judge 账单。官方日志和验收文件已有入口，自动成本汇总尚未实现。截图小时价不代表端到端成本，offload、20 轮工具调用、judge 延迟和验证频率都会影响用时。

## 本地静态检查与修正依据

```bash
# 首次仅下载官方 verl wheel、校验 SHA256 并提取配置，不安装 GPU 包。
bash scripts/dev.sh python scripts/check_gpu_config.py --fetch
# 后续离线检查。
bash scripts/dev.sh python scripts/check_gpu_config.py
bash scripts/dev.sh pytest -q --disable-warnings
bash scripts/dev.sh ruff check .
bash scripts/dev.sh ruff format --check .
bash -n scripts/gpu.sh
```

检查使用真实 verl 0.8.0 YAML 和固定 rLLM 源码中的纯配置同步函数，验证 Hydra 合成及同步后的 LoRA/group/epoch/step。没有导入 GPU backend、实例化 verl dataclass 或运行模型。

verl 0.8.0 的 [FSDP transformer 实现](https://github.com/verl-project/verl/blob/v0.8.0/verl/workers/engine/fsdp/transformer_impl.py) 用 `model_config.lora_rank > 0` 决定启用 LoRA，并读 flat `lora_alpha`；[默认配置](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/config/model/hf_model.yaml) 的 flat rank 为 0。原 Cookbook 只设置 nested 字段，因此迁移脚本补齐 flat 字段。原始 `upstream/finqa/train_verl.sh` 不变，差异有回归测试。checkpoint 容量和 metadata 检查依据 [FSDP checkpoint manager](https://github.com/verl-project/verl/blob/v0.8.0/verl/utils/checkpoint/fsdp_checkpoint_manager.py)。
