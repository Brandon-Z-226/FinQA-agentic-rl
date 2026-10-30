# 本地验证记录

验证日期：2026-10-31（Asia/Shanghai）。环境：macOS / Apple Silicon arm64，uv 0.12.22，Python 3.11.17，项目 `.venv`。没有调用真实模型、judge、远程训练服务，也没有启动 GPU 作业。

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
