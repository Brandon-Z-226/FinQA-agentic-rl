# AutoDL A800 部署记录

日期：2026-11-07（Asia/Shanghai）。本轮通过 SSH 在真实 AutoDL Linux 服务器完成环境安装、GPU 算子验证、数据准备和两条真实 Agent 推理。**未启动 GRPO、未实例化训练器、未执行任何 Optimizer Step。依赖元数据检查仍有一项已知冲突，不能称为完整训练环境验收通过。**

## 服务器与目录

| 项目 | 实测值 |
| --- | --- |
| hostname | `autodl-container-66714bb469-f35807ff` |
| OS | Ubuntu 22.04.5 / Linux x86_64 |
| GPU | NVIDIA A800 80GB PCIe × 1；nvidia-smi 总容量 81,920 MiB |
| 驱动 | 590.48.01；未重装或修改 |
| CUDA Toolkit | `/usr/local/cuda-13.0/bin/nvcc`，13.0.88 |
| 容器配额 | CPU 14 核、RAM 120 GiB；没有将宿主机 `free` 输出当成容器配额 |
| 系统盘 | 30 GiB，部署后约 29 GiB 可用 |
| 数据盘 | `/root/autodl-tmp`，90 GiB，部署后约 69 GiB 可用 |
| 登录目录 / 工作目录 | `/root` / `/root/autodl-tmp/FinQA-agentic-rl` |
| 运行数据根目录 | `/root/autodl-tmp/finqa` |
| 项目基线提交 | `4d29910f588ca880c2b0bf9b81d7556589d120ec` |

远端项目从本地已提交历史的 Git bundle 克隆，随后将 origin 设置为 `https://github.com/Brandon-Z-226/FinQA-agentic-rl.git`；GitHub API 核对 main 为同一提交。没有将本地未提交的 README 或编辑器配置混入基线。后续部署补丁单独保留。没有删除数据盘内容，也没有改变预装 Conda 环境。

`/root/autodl-tmp/finqa/env.sh` 设置 CUDA_HOME/PATH、uv 与 HF 缓存位置，不含凭证，也不修改全局 shell 文件。使用现有 Conda Python 的 pip 将 uv 0.12.22 安装到独立 bootstrap target，再由 uv 下载 Python 3.11.17，创建数据盘上的独立环境。

```bash
source /root/autodl-tmp/finqa/env.sh
cd /root/autodl-tmp/FinQA-agentic-rl
```

| 目录 | 内容 |
| --- | --- |
| `$FINQA_GPU_ROOT/venv` | 独立 CUDA Python 环境 |
| `python`, `uv-cache`, `tmp` | uv Python、构建缓存和临时文件 |
| `huggingface` | 固定 revision 的模型 snapshot，不复制第二份权重 |
| `data`, `rllm` | 已校验 FinQA 数据与 DatasetRegistry/parquet |
| `checks`, `logs` | 检查报告、推理轨迹、安装与服务日志、Git diff |

## 已安装与已验证的版本

GPU 锁文件为 `environments/cuda/uv.lock`，解析 297 个包，实际环境 295 个包。它与根目录的本地开发锁文件独立。

| 组件 | 版本 / 来源 |
| --- | --- |
| rLLM / gateway | 0.3.0rc0 / 0.1.0，同一官方提交 `4e828c81680e5a85b638ceedeaf09d1e90109ca8` |
| PyTorch / CUDA runtime | 2.11.0+cu130 / 13.0 |
| torchaudio / torchvision | 2.11.0+cu130 / 0.26.0+cu130 |
| vLLM / verl | 0.22.1 / 0.8.0 |
| FlashAttention | 2.8.3，使用本机 nvcc 源码编译 |
| Transformers / PEFT | 5.22.2 / 0.21.2 |
| NumPy | 2.3.5；下文说明官方包元数据冲突 |

`gpu.sh check` 实际导入上述 CUDA 栈，执行 BF16 矩阵乘法、FlashAttention 前向与反向并同步 GPU，全部通过。该反向运算仅针对小型随机张量，没有模型训练或优化器更新。`VerlBackend` 与 `verl.workers.engine.fsdp.transformer_impl` 的真实导入也通过，没有实例化训练器。

## 模型、数据和 LoRA

模型固定为 `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`。服务器直连 Hugging Face 出现网络重置，改用 `HF_ENDPOINT=https://hf-mirror.com` 和 `HF_HUB_DISABLE_XET=1` 下载。随后从官方 Hugging Face API 获取该 revision 的权重大小和 SHA256，在服务器逐一校验三个 safetensors 分片，全部一致。缓存约 7.51 GiB。

```text
/root/autodl-tmp/finqa/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
```

数据使用已有的官方固定归档，通过 SSH 传输后重新校验 SHA256，并复用 `gpu.sh prepare --offline --archive ...` 解压、注册。train / val / test 为 **4,030 / 522 / 558**，207 家公司、6,923 条表元数据、6,110 张实际 SQLite 表。三个 split 的 registry 和 parquet 均可加载；Agent/Evaluator entry point 均通过真实 loader 检查。

`gpu.sh lora-check` 在 meta device 上构造同一架构、应用 rank=alpha=32 的 all-linear LoRA。可训练参数 **66,060,288**，含 adapter 总参数 **4,477,484,544**，比例 **1.4754%**，所有可训练参数名称均属于 LoRA。它不证明实际 FSDP worker 已完成 LoRA 更新。

`gpu.sh config` 成功；另外使用实际安装的 `rllm.trainer.verl.utils.sync_config` 合成并检查正式与 smoke 配置，均为单卡、LoRA 32、group=4。正式 batch=16，smoke batch=2 / total_training_steps=1。这里只合成配置，**没有调用 smoke 命令**。

## 真实两样本推理

使用现有 `gpu.sh serve`，固定模型、BF16、Hermes parser、16384 上下文、max_num_seqs=8、显存比例 0.5、eager 模式，仅监听 `127.0.0.1:8000`。随后运行：

```bash
bash scripts/gpu.sh eval --concurrency 2 --no-ui \
  --output "$FINQA_GPU_ROOT/checks/agent-eval.json" \
  --episodes-dir "$FINQA_GPU_ROOT/checks/agent-eval"
```

Judge 使用安全复制到远端 `.env` 的 Qwen 配置，文件权限为 600；没有打印或提交凭证。云端连通性预检成功，真实 evaluator 评分也成功。两条均为默认 val 的前两条 `american_water` 样本，temperature=0.6、每轮 max_tokens=4096。

| 样本 | 模型答案 | 模型轮次 | reward / table_access | 单条 rollout 耗时 |
| --- | --- | ---: | --- | ---: |
| AWCC 占总长期债务发行比例 | 96.82%（1400 / 1446） | 4 | 1 / 1 | 22.70 秒 |
| 2025 养老金缴款分配至 plan trusts 的比例 | 95.65%（44 / 46） | 5 | 1 / 1 | 27.70 秒 |

两条均正常结束，errors=0。轨迹包含真实的 `get_table_names`、`get_table_info`、`sql_query`；第二条还执行了 `calculator`。Judge 耗时分别约 5.99 / 5.54 秒，正奖励来自真实评分，不是异常 fallback。

vLLM 服务指标记录 **9 次成功请求、17,424 prompt tokens、1,746 generation tokens**，无 length/abort/error 结束。这是模型服务累计 token，包含多轮重复上下文，不是 judge token 或完整训练成本。两条共享启动后的约 28 秒 rollout 窗口，不含环境安装、模型启动及 CLI 数据加载时间。

推理服务启动后约 40,447 MiB，推理后观察到 **40,543 MiB（约 39.6 GiB）**。这是采样观察值，不是持续监控得到的严格峰值，更不是训练峰值。完成验证后只停止本次启动的 vLLM 服务，最终 nvidia-smi 显存为 **0 MiB**；租赁服务器仍开机。

推理 episode 保存了 4 / 5 个 gateway steps 和完整对话，但 `prompt_ids`、`response_ids`、`logprobs` 为空，token 长度指标也为 0。因此本轮证明 API/gateway/工具/judge 推理链路可运行，**不能据此宣称训练用 token/logprob 对齐已验证**。两条 2/2 也不能代表完整 FinQA 准确率。

## 实际问题与保留的修正

1. 初始 PATH 不含 nvcc：通过独立 `env.sh` 补齐 CUDA_HOME/PATH，未改动驱动。
2. GitHub/Hugging Face 网络不稳定：项目用已提交 Git bundle 迁移，模型用镜像下载加官方 SHA256 校验；rLLM 的固定 Git 依赖成功从官方源获取。PyPI/cu130 索引可访问。
3. uv extra-build-dependencies 原有无 match-runtime 的 requirement 字典产生警告：将 ninja/packaging/psutil 改为普通字符串，保留 torch match-runtime。
4. FlashAttention 默认编译多种架构耗时较长：增加 `FLASH_ATTN_CUDA_ARCHS=80` 默认值，仅构建 A800 的 SM80。基于实测 CPU/RAM 配额将本次服务器 MAX_JOBS 从 4 调至 10，最终 73 个目标编译成功。中断与重启的构建日志均保留；没有删除缓存或用户数据。
5. **NumPy 元数据冲突尚未完全消除。** 原覆盖无上界，解析到 2.4.6，违反 mistral-common 的 Python 3.11 上界。收紧为 `numpy>=2,<2.4` 后安装 2.3.5，消除该项冲突。但 verl 0.8.0 的官方元数据要求 `<2`，vLLM 0.22.1 的 OpenCV 依赖要求 `>=2`，两者无交集。移除覆盖后 uv resolver 明确失败。保留固定 rLLM 自身采用的 NumPy override 方案，没有修改第三方 METADATA 或伪造检查结果。`uv pip check` 目前仍报告这 **1 项不兼容**，`gpu.sh setup` 因此仍返回非零；安装已完成，后续独立运行时验证和推理通过。没有绕过该问题启动训练。
6. FSDP 模块导入有非本任务 NPU router replay 警告；vLLM 正常退出时 Python resource_tracker 提示清理 1 个 semaphore。未影响本次结果，退出后 GPU 已释放。

依赖冲突依据为已安装 wheel 元数据、`logs/lock-02.log` 的无解报告及固定源码：[rLLM override](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/pyproject.toml)、[vLLM 0.22.1 requirements](https://github.com/vllm-project/vllm/blob/v0.22.1/requirements/common.txt)、[verl 0.8.0 requirements](https://github.com/verl-project/verl/blob/v0.8.0/requirements.txt)。运行通过不能证明 NumPy 2 对未执行的全部训练路径兼容。

代码修正仅涉及 CUDA `pyproject.toml` 的构建依赖语法/NumPy 上界及 `gpu.sh` 的 SM80 默认值；新增 CUDA lock 和本报告。业务 Agent、Tools、Evaluator、原始训练参数及上游快照未改变。修改保留为 Git diff，尚未提交或 push。本地离线回归 **68 passed**，Ruff lint/format 通过。

## 证据与下一步

远端 `$FINQA_GPU_ROOT/checks/` 保存 host/model/SHA256/runtime/LoRA/backend-import 报告、两套 resolved YAML、Agent 结果与 episodes、vLLM Prometheus 指标和显存采样。`logs/` 保存 setup-01 至 setup-04、lock-02/03、runtime-check、prepare、runtime-audit、lora-check、vllm、agent-eval 以及 deployment.patch。模型与 CUDA lock、必要修正已回传本地；凭证只存在忽略的 `.env`。

硬件、权重、数据、LoRA 参数选择、配置和真实推理已具备进一步诊断的基础，**还不能标记为依赖检查全绿、GRPO Smoke Test 已通过或正式训练就绪**。下一步需明确记录是否沿用 rLLM 的已知 NumPy 元数据例外，或另行验证兼容版本组合；不能静默升级固定的 verl/vLLM。得到用户对首次 Optimizer Step 的明确授权后，才运行现有一步 smoke，验证实际 FSDP LoRA、rollout token/logprob 对齐、权重同步、有限 loss/梯度、checkpoint 保存及真实峰值显存。checkpoint 恢复和完整 epoch 仍需后续单独验证。
