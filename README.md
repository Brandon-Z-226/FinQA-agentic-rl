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
  prompts/                 # 3 个原始提示词
  data_source.json         # 数据 revision / SHA256
scripts/
  dev.sh                   # 项目内环境与缓存
  train_{verl,tinker,fireworks}.sh
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

`scripts/dev.sh` 默认设置 `RLLM_HOME=$PWD/.rllm`、`FINQA_DATA_DIR=$PWD/data`、`HF_HOME=$PWD/.cache/huggingface`，执行 `uv run --locked`。显式设置的前三个环境变量优先。脚本不会写入或载入 API key。

本机 PyPI 连接在默认 TLS 配置下失败，使用系统证书后安装成功，因此项目的 `tool.uv.system-certs` 已开启。初次安装可将缓存也放进项目：`UV_CACHE_DIR=.cache/uv uv sync --locked --group dev`。

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

本地没有 verl，Hydra 解析 Tinker/Fireworks 时会提示 verl search path 不存在，但可以完成配置合成；verl 配置本阶段检查原始参数一致性和 shell 语法，完整合成与执行留待 Linux 环境。原始表加载在当前 pandas 版本会产生弃用警告，未影响本次验证。

## 保持的官方行为

- Agent 使用原生 OpenAI tool calling，最多 20 轮，四个工具不变；模型请求使用注入的 `config.base_url`，以便训练 gateway 捕获 token/logprob。
- 单表 judge 使用 `gpt-5-nano`；多表 judge 使用 `gpt-5-mini` 加原始加权 rubric。judge 与被训练的模型是两条独立调用链。
- `EvalOutput.reward` 只有 correctness；table-access 是信号/元数据。没有按 README 中的描述额外把 bonus 加入 reward。
- 无 `OPENAI_API_KEY` 或 judge 调用异常时，官方 evaluator 返回 0。离线测试验证这个行为，但这种 0 不能作为模型效果评测结果。
- 所有原始配置保存在 `upstream/finqa/`；迁移训练脚本只调整入口。具体差异见 [迁移说明](docs/migration.md)。

## GRPO 配置与下一阶段

`scripts/train_verl.sh` 保留官方 4B GRPO 配置：`Qwen/Qwen3-4B-Instruct-2507`、LoRA rank/alpha 32、group size 4、batch 16、PPO minibatch 64、学习率 1e-6、8 GPU、1 epoch、8192/4096 prompt/response 上限、max model len 16384、async vLLM、Hermes tool parser。原配置已归档，**尚未跑出 GRPO baseline 结果**。

Tinker 和 Fireworks 脚本也保留，但分别使用上游 30B/35B MoE 模型，不能拿它们的结果当作 4B 公平对照。它们是远程训练后端，不会在这台 Mac 上本地训练相应模型。

真正上云前还需完成：

1. 选择可承受的 GPU、显存和镜像，验证驱动/CUDA/PyTorch/verl/vLLM/flash-attn 组合，并生成、提交 CUDA 环境锁文件。详见 [CUDA 环境说明](environments/cuda/README.md)。
2. 部署兼容工具调用的 4B 模型，验证 tokenizer、chat template、Hermes parser，以及 gateway 的 token IDs/logprobs 捕获。
3. 在进程启动前配置可用的 judge 凭证，验证 judge 失败率和计费；先运行 2 条真实 Agent/evaluator 评测，再完成一次 optimizer step。
4. 为资源适配另存配置，核对 group/batch/minibatch、并行与 offload；保留原始 8 GPU 基线作为比较参照。
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
