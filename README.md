# ModelRank AI 🏆

这是一个自动更新的开源大语言模型排行榜，数据来源于 HuggingFace 的 Open LLM Leaderboard。通过本项目，您可以方便地查看和比较各种大语言模型的性能表现。

## 项目特点

- 🔄 **自动更新**：通过 GitHub Actions 每天自动从 HuggingFace 获取最新的模型评测数据
- 📊 **完整数据**：提供完整的排行榜数据，包括模型名称、参数量、各项评测分数等
- 📱 **响应式设计**：支持在各种设备上查看排行榜数据
- 🔍 **搜索和排序**：在完整排行榜页面支持按不同指标搜索和排序
- 📥 **数据下载**：提供 JSON 和 CSV 格式的数据下载

## 🏆 ModelRank AI 排行榜

*最后更新时间: 将由脚本自动更新*

排行榜数据将在首次运行 GitHub Action 后显示。

## 完整数据

完整的排行榜数据可以通过以下方式查看：

- [在线完整排行榜](https://chenjy16.github.io/modelrank_ai/)
- [JSON 格式数据](https://chenjy16.github.io/modelrank_ai/leaderboard.json)
- [CSV 格式数据](https://chenjy16.github.io/modelrank_ai/leaderboard.csv)

## 评测指标说明

排行榜包含以下主要评测指标：

- **Average ⬆️**：所有评测的平均分数
- **IFEval**：指令跟随能力评测
- **BBH**：大型语言模型行为基准测试
- **MATH Lvl 5**：数学问题解决能力评测
- **GPQA**：通用物理问答评测
- **MUSR**：多步推理评测
- **MMLU-PRO**：大规模多任务语言理解专业版评测

## 本地开发

### 前提条件

- Python 3.10+
- HuggingFace API 令牌

### 安装步骤

1. 克隆仓库
   ```bash
   git clone https://github.com/chenjy16/modelrank_ai.git
   cd modelrank_ai
   ```

## 数据来源

数据来自HuggingFace

## 许可证

本项目基于MIT许可证开源。