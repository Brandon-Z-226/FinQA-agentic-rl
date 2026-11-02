# 迁移范围与差异

上游版本：`rllm-org/rllm@4e828c81680e5a85b638ceedeaf09d1e90109ca8`。原始快照见 `upstream/finqa`，逐文件校验见 `upstream/manifest.json`。

| 部分 | 保持的行为 | 工程适配 |
| --- | --- | --- |
| AgentFlow | 20 轮上限、4 个原生工具、8,000 字符工具输出截断、同样的异常处理和 episode artifacts | 导入改为包内相对导入 |
| Tools | 同样的 SQLite 预加载、表名规范化、SQL 过滤、asteval 计算和工具 schema | 导入改为包内相对导入 |
| Evaluator | 单表 judge、多表 rubric、阈值、FINAL ANSWER 抽取、无 key/调用失败时返回 0 | 包内导入；增加 `FINQA_JUDGE_*` 配置和可选 Chat Completions 传输适配，默认保留官方 Responses 请求 |
| Prompts | 3 个提示词逐字保留 | 作为 wheel package data 分发 |
| 数据映射 | 保留原 CSV 列映射和 `_parse_json_list`，注册 `finqa/train`、`finqa/val`、`finqa/test` | 数据移到项目 `data/`；可用 `FINQA_DATA_DIR` 覆盖；固定下载 revision/SHA256、安全解压、暂存后移动、完整性检查、离线入口 |
| 训练入口 | 原 AgentTrainer 和 dataset split | 包模块/console script 启动，延迟 AgentTrainer 导入；单 A800 profile 增加预检查、有效配置记录和 smoke 样本限制 |
| 训练脚本 | 原 override 值与用户 override 顺序；原始快照不变 | 包模块入口与 `FINQA_PYTHON`；verl 补充 flat `lora_rank/lora_alpha=32`；新增单 A800 配置和部署入口 |
| rLLM | 使用官方框架、注册、gateway、trainer 和算法 | rLLM 与 gateway 固定同一 Git commit；根 uv overrides 显式带入本项目 |

本阶段没有重写 reward、修改工具执行策略或增加 Credit Assignment 算法。单 A800 配置的资源差异单独记录，保留 group、学习率、上下文和正式训练 batch 等参数。测试通过 AST 对比约束 flow/tools 和 evaluator 的评分逻辑与原始版本相同；明确排除 judge 配置、客户端构造和传输适配，并用离线假客户端检查请求及评分。提示词及原始快照通过 SHA256/字节比较验证。

2026-11-03：按用户要求支持替换 judge API。`.env.example` 提供密钥、接口地址、模型名和协议模板；`scripts/dev.sh` 在 `.env` 存在时通过 uv 载入。自定义端点不会继承全局 OpenAI 密钥。Chat Completions 分支保留 system/user 提示词与输出 token 上限，默认不发送 reasoning/verbosity 或强制 JSON Schema 参数。更换模型和接口意味着 judge 条件改变，不能称为原始 GPT-5 judge 结果复现。

同日按用户要求增加 `FINQA_JUDGE_REASONING_EFFORT`。配置为 low 时同时覆盖单表和多表：Responses 发送 `reasoning.effort`；Chat Completions 发送 `reasoning_effort`，并使用包含推理 token 的 `max_completion_tokens`。未设置时保持此前行为。通过 `gpt-6.1-sol` 完成三次在线 judge 请求，详见 [验证记录](validation.md)。

源码优先于过时说明：上游 README 说会把 table-access bonus 加到奖励，但实际 `EvalOutput.reward` 只有 correctness。本项目保持实际实现；`table_access` 和 `right_table_access_reward` 仅作为信号/元数据。上游还会记录失败的 `get_table_info` 尝试；该行为也保留。

2026-11-03：核对 verl 0.8.0 后修正 FSDP LoRA 开关，迁移的 8 GPU 脚本只新增 flat `lora_rank/lora_alpha=32` 两项，原 nested 字段与原始快照不变。另建单 A800 配置，限制 GPU 数、执行并发、vLLM 显存比例和 checkpoint 保留数，明确对齐 rLLM/verl 共享配置；smoke 单独限制样本及 step。新增 Linux uv 部署入口、固定模型下载、运行时检查和 LoRA checkpoint 校验，不在 Mac 安装 CUDA 包。版本依据、参数差异及尚未完成的实机验收见 [A800 部署说明](../environments/cuda/README.md)。

`scripts/dev.sh` 将 rLLM 注册目录和缓存放在项目内，避免写入或覆盖已有的全局注册；可用环境变量显式覆盖。环境变量应在进程启动前设置，因为上游 DatasetRegistry 和 judge 会在导入时读取配置。无需为 entry-point 注册写用户 registry 文件。
