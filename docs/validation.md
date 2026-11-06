# 本地验证记录

初始离线验证日期：2026-10-31（Asia/Shanghai）。环境：macOS / Apple Silicon arm64，uv 0.12.22，Python 3.11.17，项目 `.venv`。初始验证没有调用真实模型、judge、远程训练服务，也没有启动 GPU 作业。2026-11-03 新增的在线 judge 验证见文末。

## 环境与依赖

- `uv sync --group dev --system-certs` 成功，生成 `uv.lock`；锁定 179 个跨平台解析条目，本机安装 156 个包。
- `uv lock --check --offline` 通过，`uv pip check` 确认已安装包依赖兼容。
- rLLM 与 gateway 均来自 `4e828c81680e5a85b638ceedeaf09d1e90109ca8`，版本分别为 `0.3.0rc0`（上游写作 `0.3.0.pre`）和 `0.1.0`。
- 关键本机锁定版本：OpenAI SDK 3.41.0、pandas 3.1.0、PyTorch 2.15.0、Tinker 0.36.0、tinker-cookbook 0.5.9、pytest 9.2.0、Ruff 0.17.3。
- 本机未安装 `verl`、`vllm`、`flash_attn` 或 `ray`。PyTorch 来自官方基础依赖的传递依赖；未安装 CUDA 训练后端。
- 初次 Git 下载触发 uv 缓存锁超时后，复用了同级本地 rLLM 的同一提交填充项目缓存；最终 pyproject/lock 仍指向官方 Git URL，没有同级目录或本机绝对路径依赖。
- 默认 TLS 访问 PyPI 失败，使用 macOS 系统证书后成功；已将 `system-certs = true` 写入项目 uv 配置。

## 数据与离线行为

通过官方固定 URL 下载了 7,530,234 字节的 `data.tar.gz`，SHA256 与数据集仓库 LFS 元数据一致。首次执行 `finqa-prepare --offline --archive .cache/data.tar.gz` 完成校验、解压与注册；再次执行 `finqa-prepare --offline` 校验成功，样本数不变。

| 项目 | 实测结果 |
| --- | ---: |
| train / val / test | 4,030 / 522 / 558 |
| 单表 / 多表样本 | 5,110 / 0 |
| 公司数 | 207 |
| 表元数据条目 | 6,923 |
| 空表条目 | 813 |
| SQLite 实际表 | 6,110 |
| 问题引用但不可查询的表 | train 0 / val 0 / test 0 |

三份 DatasetRegistry 数据和三份 verl 配套 parquet 均生成，并实际加载验证。测试使用临时 registry，项目真实数据注册保存在 `.rllm/datasets`。

`pytest -q`：**39 项通过，无跳过、无失败**。包含官方 17 项测试，以及数据/路径/注册、真实表加载与 SQL、多轮 AgentFlow 假模型、20 轮上限、调用失败、judge 缺失与单/多表 rubric、源码行为与配置一致性验证。pytest 禁止网络连接，未将模拟评测结果当作真实成功率。

真实表加载出现 6,923 条同类 pandas `convert_dates` 弃用警告（每张表一次），来源为原始工具实现。保留实现以维持复现行为，本次读取、SQL 和问题表引用验证均通过。

## 配置、注册与打包

| 检查 | 结果 |
| --- | --- |
| rLLM Agent entry point | `rllm agent list` 显示 `finqa` plugin |
| rLLM Evaluator entry point | `load_evaluator("finqa")` 返回 `EvaluatorFn('finqa_evaluator')` |
| Tinker Hydra 配置 | 原始脚本 `--cfg job` 成功：30B-A3B、group 8、data batch 16 |
| Fireworks Hydra 配置 | 原始脚本 `--cfg job` 成功：35B-A3B、group 8、data batch 16 |
| verl 配置 | shell 语法、所有原始 override、4B/GRPO/Hermes 等关键参数及用户 override 顺序通过检查 |
| Ruff | lint 和 format check 通过 |
| 上游源码 | 全量快照 SHA256、提示词字节、核心 flow/tools/evaluator 的非导入 AST 一致 |
| 构建 | sdist 和 wheel 成功构建 |
| wheel smoke | wheel 安装到临时 target，从仓库外 cwd 且移除源码路径后导入成功；两个插件、三个 prompt、数据来源和许可证均存在 |

Tinker/Fireworks 的 `--cfg job` 只做配置合成，没有构造或启动 trainer；合成输出仍是 runtime `sync_config` 之前的配置。Hydra 提示缺少 `verl.trainer.config` search path，符合本地未安装 verl 的预期；没有将其解释成 GPU 配置已验证。该版本 CLI 没有 `rllm evaluator list` 子命令，Evaluator 通过真实 loader 和 entry point 验证。

## 当前复现进度

第一阶段已完成仓库初始化、本地依赖锁定、官方业务代码迁移、数据准备、注册/导入/行为/打包的离线验证，以及原始 GRPO 配置归档。Git 仓库已初始化。

尚未完成真实模型的两样本评测、gateway token/logprob 链路、一次 optimizer step 或完整 GRPO baseline。第二阶段训练与第三阶段 Credit Assignment 没有开始。CUDA 定义是待 Linux 验证的环境入口，尚无已验证的 GPU lock。

上云前按 README 完成 GPU/镜像兼容性、模型/tool parser/gateway、judge、资源适配配置、指标与成本记录，再开始训练。当前没有实测 Reward、Loss、Agent 成功率、模型 token 消耗或训练成本。

## 2026-11-03：Judge 接入与 low 档位验证

按用户授权使用本地 `.env` 中的配置，调用 OpenAI 官方 `https://api.openai.com/v1` 的 `gpt-6.1-sol`，协议为 Chat Completions。单表和多表请求均显式发送 `reasoning_effort=low`，分别使用 512 / 5000 的 `max_completion_tokens` 上限。每次请求超时上限 60 秒、禁用自动重试。密钥未写入文档、日志或 Git。

| 合成测试 | 实际响应与 evaluator 结果 | API 耗时 |
| --- | --- | --- |
| 单表正确答案：120 - 80 = 40 million | 官方格式 `{True, explanation}`；reward=1，is_correct=true | 3.23 秒 |
| 单表错误答案：120 - 80 = 400 million | 官方格式 `{False, explanation}`；reward=0，is_correct=false | 3.24 秒 |
| 多表：资产 200、负债 80，权益 120、负债率 40% | 返回完整 JSON，六项评分均为 100；reward=1，is_correct=true | 2.53 秒 |

共发送三次真实请求，均收到正常响应、`finish_reason=stop`，返回模型为 `gpt-6.1-sol`。核对了原始响应后再核对 evaluator 结果，错误答案的 0 分不是请求异常触发的 fallback。API usage 合计输入 2,375 token、输出 168 token、总计 2,543 token；返回的 reasoning token 数为 0，不能用它推断服务内部的实际推理过程。这里验证的是 low 参数已发送且请求成功。

离线回归 `pytest -q --disable-warnings`：**50 项通过，无跳过、无失败**，约 11 秒；仍有原始数据加载的 6,923 条 pandas 弃用警告。覆盖两种 API 协议、单表/多表 low 档位覆盖、默认官方行为、凭证隔离、评分与原始源码一致性。Ruff lint、format check 和 `git diff --check` 通过。

本次仅验证 judge 接入和少量合成答案评分，没有测量 FinQA 全集 judge 质量、真实 Agent 成功率或 GRPO 训练效果。

## 2026-11-03：LoRA 修正与单 A800 部署准备

根据固定 verl 0.8.0 源码，FSDP 用 flat `model.lora_rank` 判定 LoRA，默认是 0。原 Cookbook 只设置 nested `model.lora.rank/alpha`，现迁移脚本补齐 flat rank/alpha=32；原始快照完全保留，来源测试仅允许这两项额外参数。

新增独立的单卡训练配置、一步 smoke 配置、Linux uv 环境脚本、固定 revision 的模型下载、CUDA/FlashAttention 运算预检查、meta 模型 LoRA 参数检查及 smoke checkpoint metadata 校验。CUDA 依赖定义选择 torch 2.11.0 cu130 / vLLM 0.22.1 / verl 0.8.0 / flash-attn 2.8.3；没有在本机安装这些 GPU 包，也没有生成经 Linux 验证的 CUDA lock。

本轮实际执行结果：

| 检查 | 结果 |
| --- | --- |
| 全量离线 pytest | **63 项通过，无跳过、无失败**，11.27 秒；原始 6,923 条 pandas 警告不变 |
| LoRA/单卡防回归 | 拒绝 flat rank=0、两套 LoRA 字段不一致、多卡/TP、分离训练、超限长度、无效 GRPO group、过少 smoke 样本 |
| 部署保护 | 磁盘不足拒绝启动；macOS 在安装前退出；smoke 缺失/错误 LoRA metadata 报错 |
| 真实 Hydra 合成与 rLLM sync_config | 两个 profile 均通过；正式配置 rank=32、n=4、batch=16、1 epoch；smoke batch=2、1 step |
| 训练模块 dry run | `python -m finqa_agentic_rl.train --config-name a800_smoke --cfg job --resolve` 成功，未进入训练函数 |
| Ruff / shell / Git diff | lint、format、bash 语法和空白检查通过 |
| 根 uv lock | `uv lock --check --offline` 通过，179 个解析条目；根依赖未改变 |

静态配置验证从 PyPI 官方 verl 0.8.0 wheel 校验 SHA256 后提取配置到忽略的 `.cache/gpu-static/`，没有安装 wheel。脚本仅执行已安装且固定的 rLLM 源码中的配置同步函数，检查实际同步后的值，不使用手写替代规则。`--fetch` 可重新获取，之后可离线复查。该检查没有实例化 verl 配置 dataclass、导入 CUDA backend 或执行 GPU 算子。

尚未执行：云端 uv lock/sync、CUDA runtime check、权重下载、真实 Agent 两样本评测、meta LoRA 检查、FSDP optimizer step、checkpoint 保存/恢复、完整 epoch、显存峰值和吞吐测量。部署入口中的这些动作是待实机执行的验收步骤，不能把新增脚本或本地 63 项测试视作它们已经通过。当前仍无 GRPO baseline 或成本测量。

## 2026-11-06：Qwen judge 接入验证

用户更新本地 `.env` 后，实际测试 `https://maas.qianwenaiapi.com/compatible-mode/v1` 上的 `qwen3.8-max-0902`，采用 Chat Completions；单表和多表使用同一模型。四次请求均携带 `reasoning_effort=low`，均正常返回该模型标识和 `finish_reason=stop`。这证明参数随请求发送且调用成功，不单独证明服务内部如何实现 low。每次超时 60 秒、禁用自动重试，未修改 `.env` 或输出密钥。

| 合成测试 | evaluator 结果 | API 耗时 |
| --- | --- | --- |
| 单表正确：120 - 80 = 40 million | reward=1，is_correct=true | 3.75 秒 |
| 单表错误：120 - 80 = 400 million | reward=0，is_correct=false；真实响应明确判错，并非异常 fallback | 4.32 秒 |
| 多表首次：资产/负债推导权益与比例 | 服务返回六项 100 分，但 JSON 带 Markdown 代码块；原解析得到 reward=0 | 9.61 秒 |
| 多表兼容修复后复测 | 成功解析六项评分，reward=0.9825，is_correct=true | 12.11 秒 |

兼容修复仅针对 Chat Completions 多表响应：若完整响应是一个 JSON Markdown 代码块，移除外层标记，再交给原有 JSON 解析及加权评分。不会从附带其他说明的文本中任意抽取 JSON，也不改变单表判定、rubric、权重或 Responses 分支。复测中 Qwen 给 reasoning/structure 分别 90/95，其余四项为 100，原有权重计算得到 0.9825；不能将两次评分差异解释为修复改变了评分规则。

API usage 合计输入 4,305、输出 1,200、总计 5,505 token，包括首次发现格式问题的请求。这里只是少量合成用例的接入和解析检查，尚未评估 Qwen 在完整 FinQA 上的评分质量或与原 judge 的一致性。

完整离线回归：**68 项通过，无跳过、无失败**，13.54 秒；仍有原始表加载产生的 6,923 条 pandas 警告。新增代码块大小写/无语言标签兼容，以及无效或嵌入式 JSON 的回归测试。原始快照和核心评分 AST 一致性、Ruff lint/format 均通过。
