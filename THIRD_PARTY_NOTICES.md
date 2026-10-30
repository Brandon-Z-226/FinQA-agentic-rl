# 第三方来源与许可证

本项目迁移自 [rllm-org/rllm](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8/cookbooks/finqa)，固定 commit `4e828c81680e5a85b638ceedeaf09d1e90109ca8`。原作者为 rLLM 项目贡献者。

- `upstream/finqa/` 保存完整、未修改的 Cookbook 快照；`upstream/manifest.json` 记录逐文件 SHA256。
- `src/finqa_agentic_rl/`、`scripts/train_*.sh`、`tests/test_upstream.py` 是迁移版本；修改文件带有来源和修改声明，差异见 `docs/migration.md`。提示词保持原始字节。
- 该 commit 的根 `LICENSE` 正文是 **Apache License 2.0**，完整原文保存在 `licenses/rllm-LICENSE`。根 `pyproject.toml` 的 classifier 却标记 MIT，二者存在不一致；本项目保留并遵循随代码分发的 Apache-2.0 正文，不把 classifier 当作重新授权。
- `rllm` 和 `rllm-model-gateway` 作为第三方包安装，未把其框架实现复制进本项目。gateway 的包元数据声明 Apache-2.0。其他依赖保留各自的许可证。
- 数据来自 [rLLM FinQA Dataset](https://huggingface.co/datasets/rLLM/rLLM-FinQA-Dataset/tree/33f2eb380e6cba84b8df2dd831d33c4ded8f6093)；数据集卡声明 `apache-2.0`，卡片原文保存在 `upstream/DATASET_CARD.md`。数据包通过 `src/finqa_agentic_rl/data_source.json` 固定 revision 与 SHA256，下载内容不提交到 Git。

这里复现的是 rLLM Cookbook 使用的金融表格工具任务，具体数据以其 `data.tar.gz` 为准；不把其他同名 FinQA 数据集或数据卡上的全部配置自动合并进复现范围。
