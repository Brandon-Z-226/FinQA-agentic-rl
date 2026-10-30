# 后续 Linux / CUDA 环境

本目录是独立的 uv 项目，目标为 Linux x86_64、Python 3.11 和 NVIDIA GPU。它引用根项目和同一 commit 的 `rllm[verl]`，与 macOS 本地 `.venv` 分离。本阶段未租 GPU、未安装或验证 CUDA 栈，也未生成经 Linux 验证的锁文件。

上游 extras 当前指定 `verl==0.8.0`、`vllm==0.22.1`、`flash-attn==2.8.3`、`torch>=2.10.0`、`transformers>=5.5.3` 等。这里保留这些约束，不预先承诺它们在未来选定的驱动、CUDA、显卡和镜像上兼容。`flash-attn` 的构建依赖与运行时 torch 对齐。

确定 GPU 镜像后，在仓库根目录执行以下环境准备步骤，再保存 `environments/cuda/uv.lock`、镜像 digest、驱动/CUDA 和 GPU 信息：

```bash
uv sync --project environments/cuda
export RLLM_HOME="$PWD/.rllm"
export FINQA_DATA_DIR="$PWD/data"
environments/cuda/.venv/bin/finqa-prepare
FINQA_SKIP_PRELOAD=1 FINQA_PYTHON="$PWD/environments/cuda/.venv/bin/python" bash scripts/train_verl.sh --cfg job
```

最后一条仅解析配置；不要把成功解析视为训练成功。正式启动沿用同一命令并移除 `--cfg job`，此前必须配置可用的 judge 凭证、完成两条真实 Agent 评测和一次 optimizer step。

原始 4B 配置使用 8 GPU、group size 4、train batch 16、PPO minibatch 64、LoRA rank/alpha 32、lr 1e-6、8192/4096 prompt/response 上限、max model len 16384。降低 GPU 数量需要重新核算显存、有效 batch、动态 batch、offload 和并行设置；只改 `n_gpus_per_node` 不保证能运行。

资源适配应保存为新实验配置，保留 `upstream/finqa/train_verl.sh` 的原始基线。未来比较 Credit Assignment 时固定任务、split、模型初始 checkpoint、judge/prompt、采样设置、种子和训练预算，并记录任何资源适配差异。
