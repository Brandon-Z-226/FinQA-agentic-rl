# 迁移范围与差异

上游版本：`rllm-org/rllm@4e828c81680e5a85b638ceedeaf09d1e90109ca8`。原始快照见 `upstream/finqa`，逐文件校验见 `upstream/manifest.json`。

| 部分 | 保持的行为 | 工程适配 |
| --- | --- | --- |
| AgentFlow | 20 轮上限、4 个原生工具、8,000 字符工具输出截断、同样的异常处理和 episode artifacts | 导入改为包内相对导入 |
| Tools | 同样的 SQLite 预加载、表名规范化、SQL 过滤、asteval 计算和工具 schema | 导入改为包内相对导入 |
| Evaluator | 单表 judge、多表 rubric、阈值、FINAL ANSWER 抽取、无 key/调用失败时返回 0 | 导入改为包内相对导入 |
| Prompts | 3 个提示词逐字保留 | 作为 wheel package data 分发 |
| 数据映射 | 保留原 CSV 列映射和 `_parse_json_list`，注册 `finqa/train`、`finqa/val`、`finqa/test` | 数据移到项目 `data/`；可用 `FINQA_DATA_DIR` 覆盖；固定下载 revision/SHA256、安全解压、暂存后移动、完整性检查、离线入口 |
| 训练入口 | 原 AgentTrainer、Hydra unified 配置和 dataset split | 用包模块/console script 启动，延迟 AgentTrainer 导入，更新错误提示 |
| 训练脚本 | 3 个后端的所有原始 override 值及用户 override 顺序 | `python train.py` 改为 `python -m finqa_agentic_rl.train`，支持 `FINQA_PYTHON` |
| rLLM | 使用官方框架、注册、gateway、trainer 和算法 | rLLM 与 gateway 固定同一 Git commit；根 uv overrides 显式带入本项目 |

本阶段没有重写 reward、修改工具执行策略、增加 Credit Assignment 算法或调整 GRPO 超参数。测试通过 AST 对比约束 flow/tools/evaluator 的行为代码与原始版本相同，提示词及原始快照通过 SHA256/字节比较验证。

源码优先于过时说明：上游 README 说会把 table-access bonus 加到奖励，但实际 `EvalOutput.reward` 只有 correctness。本项目保持实际实现；`table_access` 和 `right_table_access_reward` 仅作为信号/元数据。上游还会记录失败的 `get_table_info` 尝试；该行为也保留。

`scripts/dev.sh` 将 rLLM 注册目录和缓存放在项目内，避免写入或覆盖已有的全局注册；可用环境变量显式覆盖。环境变量应在进程启动前设置，因为上游 DatasetRegistry 和 judge 会在导入时读取配置。无需为 entry-point 注册写用户 registry 文件。
