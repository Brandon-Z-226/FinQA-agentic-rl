# FinQA Agentic RL

以第三方依赖方式使用 rLLM，迁移其官方 FinQA Cookbook，先建立可审查、可复现的工程与离线验证基础，再进行真实模型评测、4B GRPO 训练及 Credit Assignment 对比。

本次固定的 rLLM commit 为 [`4e828c81680e5a85b638ceedeaf09d1e90109ca8`](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8/cookbooks/finqa)。本地迁移与离线验证不等同于完成官方训练结果复现：真实 Agent 成功率、GRPO Reward/Loss、训练 token 和成本，需要模型、judge 与 GPU 环境就绪后测量。

## 目录

```text
src/finqa_agentic_rl/
  finqa_flow.py             # 官方多轮 AgentFlow
  finqa_tools.py            # SQLite 表工具与 calculator
  finqa_eval.py             # 官方 judge/rubric
  finqa_constants.py        # 数据和提示词路径
  prepare_finqa_data.py     # 固定数据下载、校验、注册
  train.py                 # 官方 AgentTrainer 入口
  gpu.py                   # GPU 准备、预检查与启动命令
  prompts/                 # 3 个原始提示词
  data_source.json         # 数据 revision / SHA256
scripts/
  dev.sh                   # 项目内环境与缓存
  gpu.sh                   # 独立 Linux GPU 环境入口
  check_gpu_config.py      # 不安装 GPU 包的 Hydra/同步检查
  train_{verl,tinker,fireworks}.sh
configs/a800_*.yaml         # 单 A800 LoRA 配置与一步 smoke
environments/cuda/          # 独立 Linux GPU 环境定义
upstream/finqa/             # 完整未修改的官方 Cookbook 快照
upstream/manifest.json      # 上游逐文件 SHA256
tests/                     # 官方测试及离线集成验证
docs/                      # 迁移差异与本地验证记录
licenses/                  # 上游许可证原文
pyproject.toml / uv.lock    # 本地环境定义与锁文件
data/ / .rllm/ / .cache/    # 运行产物，不提交到 Git
```

## 依赖和本地环境

使用 uv、Python 3.11，项目内 `.venv`。`.python-version` 固定本次使用的 3.11.17；无需改动系统 Python、Conda、shell 配置或全局 rLLM 注册。

| 分组 | 内容 |
| --- | --- |
| 运行依赖 | 固定 Git commit 的 `rllm` 和 `rllm-model-gateway`；OpenAI SDK、pandas、asteval、huggingface-hub、Hydra |
| `dev` | pytest、Ruff、build |
| 后续 CUDA | 独立 `environments/cuda` 项目引用根包及 `rllm[verl]`，不随本地同步安装 verl/vLLM/flash-attn |

rLLM 的基础依赖本身包含 Tinker SDK/cookbook 及其传递依赖；“本地环境”不表示删掉官方依赖或完全没有 PyTorch。框架仍由第三方包提供，仓库只迁移 FinQA 业务层。两份环境定义显式固定 gateway 同一提交，并保留上游 `litellm==1.83.0` 等 uv overrides。依赖固定使用 [uv Git source 的 rev/subdirectory 机制](https://docs.astral.sh/uv/concepts/projects/dependencies/#git)，没有对同级 `../rllm` 的运行时路径依赖。

```bash
# 在项目根目录；首次同步需要联网。
uv sync --locked --group dev

# 后续命令通过脚本统一项目路径和缓存。
bash scripts/dev.sh python --version
bash scripts/dev.sh rllm agent list
bash scripts/dev.sh python -c 'from rllm.eval.evaluator_loader import load_evaluator; print(load_evaluator("finqa"))'
```

`scripts/dev.sh` 默认设置 `RLLM_HOME=$PWD/.rllm`、`FINQA_DATA_DIR=$PWD/data`、`HF_HOME=$PWD/.cache/huggingface`，执行 `uv run --locked`。显式设置的前三个环境变量优先。若项目根目录存在 `.env`，脚本通过 uv 的 `--env-file` 载入；已导出的环境变量优先。

本机 PyPI 连接在默认 TLS 配置下失败，使用系统证书后安装成功，因此项目的 `tool.uv.system-certs` 已开启。初次安装可将缓存也放进项目：`UV_CACHE_DIR=.cache/uv uv sync --locked --group dev`。

## Judge API 配置

在项目根目录的 `.env` 中填写下面三项。首次使用且文件不存在时，可从 `.env.example` 复制。`.env` 已被 Git 忽略，只在本机保存凭证。

```dotenv
FINQA_JUDGE_API_KEY=你的服务商密钥
FINQA_JUDGE_BASE_URL=https://服务商提供的接口地址/v1
FINQA_JUDGE_MODEL=服务商提供的模型标识
FINQA_JUDGE_API_STYLE=chat_completions
```

`BASE_URL` 使用服务商给出的 API 根地址，通常以 `/v1` 结尾，不要追加 `/chat/completions`。模板默认使用 OpenAI 兼容 Chat Completions 协议，默认不发送 reasoning/verbosity 参数，也不要求服务商支持 JSON Schema；多表评分仍要求模型按原提示词返回 JSON。支持原始 Responses 请求参数的服务可将 `API_STYLE` 改为 `responses`。

支持推理档位的模型可设置 `FINQA_JUDGE_REASONING_EFFORT=low`，同时应用于单表和多表。Chat Completions 使用 `reasoning_effort` 与包含推理 token 的 `max_completion_tokens`；Responses 使用 `reasoning.effort`。留空时保留原行为：Chat Completions 不传推理档位，Responses 单表 low / 多表 medium。

单表、多表默认都使用 `FINQA_JUDGE_MODEL`；需要不同模型时另设 `FINQA_JUDGE_MULTI_TABLE_MODEL`。这些变量只配置 judge，不改变 Agent 的 `config.base_url` 或训练模型。自定义 judge 地址不会回退使用全局 `OPENAI_API_KEY`。不设置 `FINQA_JUDGE_*` 时保留官方模型与 Responses 行为。

通过 `bash scripts/dev.sh ...` 启动会自动读取 `.env`；直接运行 Python 不会自动读取它。CUDA 环境可用 `uv run --project environments/cuda --env-file .env ...`，从项目根目录启动。judge 在导入时读取配置，修改 `.env` 后应重启进程。更换 judge 会改变评测条件，后续 GRPO / Credit Assignment 实验应固定并记录同一 judge 配置。

## 数据准备

```bash
# 下载固定 revision 的官方压缩包，校验 SHA256，再解压和注册。
bash scripts/dev.sh finqa-prepare

# 已有官方压缩包时，可完全离线运行。
bash scripts/dev.sh finqa-prepare --offline --archive .cache/data.tar.gz

# 已准备的数据重复运行时，先校验文件，再重建注册，避免重复累积样本。
bash scripts/dev.sh finqa-prepare --offline
```

数据固定为 `rLLM/finqa` 的 revision `33f2eb380e6cba84b8df2dd831d33c4ded8f6093`，该名称当前重定向到 `rLLM/rLLM-FinQA-Dataset`。压缩包 SHA256 为 `24d9c89688160742dffd629c1c79f57f6f8870e5536c8c3172158588b05f086a`。

| 注册 split | 记录数 |
| --- | ---: |
| `finqa/train` | 4,030 |
| `finqa/val` | 522 |
| `finqa/test` | 558 |

官方压缩包包含 207 家公司、6,923 个表元数据条目和 5,110 条单表问题。数据卡另列了多表数据，但 Cookbook 的下载/预处理入口只使用 `data.tar.gz` 中的这三个 CSV；本阶段不额外混入多表 parquet。多表 evaluator 代码照原样保留并用假 judge 测试。

实测 813 个元数据条目对应空表，按原始逻辑不导入 SQL；SQLite 中实际有 6,110 张表。三个 split 的全部问题引用表均可查询。

数据位于 `data/`，注册和 verl 配套 parquet 位于 `.rllm/datasets/`。`data/.finqa-source.json` 记录来源及所有解压文件的摘要。发现缺文件、内容变化或不完整目录时明确失败；请先移走需保留的旧目录再重新准备。`FINQA_TABLES_ROOT` 仍保留为单独覆盖表目录的官方选项。

## 本地验证

```bash
bash scripts/dev.sh pytest -q
bash scripts/dev.sh ruff check .
bash scripts/dev.sh ruff format --check .
uv lock --check
bash scripts/dev.sh python -m build --installer uv

# 解析官方 Tinker/Fireworks 配置，不启动训练。
FINQA_SKIP_PRELOAD=1 bash scripts/dev.sh bash scripts/train_tinker.sh --cfg job
FINQA_SKIP_PRELOAD=1 bash scripts/dev.sh bash scripts/train_fireworks.sh --cfg job
```

pytest 屏蔽网络连接，清除测试进程的 judge key，并隔离 DatasetRegistry。覆盖原始单元测试、插件发现、导入、多轮工具调用、20 轮上限、失败路径、单/多表评分、数据预处理、注册 round-trip、官方数据校验与真实 SQLite 表加载。未下载真实数据时，仅两个标记 `data` 的集成测试会跳过；不能把此时的结果称为完整数据验证。

另对核心行为 AST、提示词字节、原始文件摘要、所有训练 override 做一致性检查。wheel 构建后还需验证其中包含提示词、数据来源配置和插件入口。执行结果及已知限制见 [本地验证记录](docs/validation.md)。

本地没有安装 verl。单卡配置通过 `scripts/check_gpu_config.py` 使用官方 verl wheel 中的 YAML 以及固定 rLLM 的配置同步函数进行实际合成和参数检查；不导入 GPU backend，执行验证仍留待 Linux。Tinker/Fireworks 的配置解析会提示 verl search path 不存在，但可以完成合成。原始表加载的 pandas 弃用警告未影响验证。

## 保持的官方行为

- Agent 使用原生 OpenAI tool calling，最多 20 轮，四个工具不变；模型请求使用注入的 `config.base_url`，以便训练 gateway 捕获 token/logprob。
- 未配置覆盖项时，单表 judge 使用 `gpt-5-nano`；多表 judge 使用 `gpt-5-mini` 加原始加权 rubric。可按上述方式替换 judge 服务，评分解析与 rubric 保持不变。
- `EvalOutput.reward` 只有 correctness；table-access 是信号/元数据。没有按 README 中的描述额外把 bonus 加入 reward。
- 无可用 judge key 或 judge 调用异常时，evaluator 返回 0。离线测试验证这个行为，但这种 0 不能作为模型效果评测结果。
- 所有原始配置保存在 `upstream/finqa/`；迁移脚本调整入口，并为 verl FSDP 补齐真正启用 LoRA 的 flat rank/alpha 字段。具体差异见 [迁移说明](docs/migration.md)。

## GRPO 配置与下一阶段

`scripts/train_verl.sh` 保留官方 4B GRPO 参数和 8 GPU 入口，并修正 LoRA：verl 0.8.0 FSDP 读取 `model.lora_rank/lora_alpha`，只设置原来的 nested `model.lora.rank/alpha` 会留下 flat rank=0。现在两组字段均为 32；原始快照未改。**尚未跑出 GRPO baseline 结果**。

单 A800 使用新增的 `configs/a800_80gb.yaml`：保留 group 4、train batch 16、PPO minibatch 64、lr 1e-6、1 epoch、8192/4096/16384 长度及 Hermes parser，改为 1 GPU、CPU offload、8 并发、vLLM 显存比例 0.5 和 eager 模式。先用 `a800_smoke.yaml` 跑 2 条样本、1 step，再跑完整训练。环境安装、固定模型下载、两条评测、LoRA 检查、训练和 checkpoint 验收见 [A800 部署说明](environments/cuda/README.md)。50GB 数据盘余量紧，应在租机前确认容量方案。

Tinker 和 Fireworks 脚本也保留，但分别使用上游 30B/35B MoE 模型，不能拿它们的结果当作 4B 公平对照。它们是远程训练后端，不会在这台 Mac 上本地训练相应模型。

真正上云前还需完成：

1. 在 A800 上选定 CUDA 13.0 devel 镜像，实测 PyTorch 2.11.0 / verl 0.8.0 / vLLM 0.22.1 / flash-attn 2.8.3，生成并提交 CUDA 锁文件，确认磁盘容量。详见 [CUDA 环境说明](environments/cuda/README.md)。
2. 部署兼容工具调用的 4B 模型，验证 tokenizer、chat template、Hermes parser，以及 gateway 的 token IDs/logprobs 捕获。
3. 在进程启动前配置可用的 judge 凭证，验证 judge 失败率和计费；先运行 2 条真实 Agent/evaluator 评测，再完成一次 optimizer step。
4. 实测新增单卡配置的峰值显存、LoRA checkpoint、恢复能力和稳定性；保留原始 8 GPU 配置作为来源参照。
5. 确认 Reward、Loss、成功率、prompt/completion token、judge token、GPU 小时及实际账单的记录位置。官方 console/UI logger 被保留；成本汇总尚未实现，也没有任何实测成本。
6. 在固定 split、初始模型、judge/prompt、采样、seed 和预算后完成标准 GRPO，再开展 Credit Assignment 对比；本阶段未加入改进算法。

模型和 judge 就绪后，可使用官方 CLI 做两条在线评测（当前阶段不执行）：

```bash
bash scripts/dev.sh rllm eval finqa --agent finqa --evaluator finqa \
  --model "$FINQA_MODEL" --base-url "$FINQA_MODEL_BASE_URL" \
  --split test --max-examples 2
```

## 来源与许可证

来源、许可证不一致说明和数据集卡见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。上游代码实际随附的许可证正文为 Apache-2.0，已完整保留在 `licenses/rllm-LICENSE`；修改文件带有来源及修改声明。数据、缓存、环境、训练输出和凭证不纳入 Git。
