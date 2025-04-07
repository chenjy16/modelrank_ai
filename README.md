# ModelRank AI 🏆

This is an automatically updated open-source large language model leaderboard with data sourced from HuggingFace. Through this project, you can easily view and compare the performance of various large language models.

## Project Features

- 🔄 **Automatic Updates**: Automatically fetches the latest model evaluation data from HuggingFace daily via GitHub Actions
- 📊 **Complete Data**: Provides comprehensive leaderboard data, including model names, parameter counts, and various evaluation scores
- 📱 **Responsive Design**: Supports viewing leaderboard data on various devices
- 🔍 **Search and Sort**: Supports searching and sorting by different metrics on the complete leaderboard page
- 📥 **Data Download**: Provides data in JSON and CSV formats for download

## 🏆 ModelRank AI Leaderboard

*Last updated: 2025-04-07 06:36:11 UTC*

| Rank | Model | Avg | Params(B) | IFEval | BBH | MATH | GPQA | MUSR | MMLU-PRO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [<a target="_blank" href="https://huggingface.co/MaziyarPanahi/calme-3.2-instruct-78b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">MaziyarPanahi/calme-3.2-instruct-78b</a>  <a target="_blank" href="https://huggingface.co/datasets/open-llm-leaderboard/MaziyarPanahi__calme-3.2-instruct-78b-details" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">📑</a>](https://huggingface.co/MaziyarPanahi/calme-3.2-instruct-78b) | 52.08 | 78.0 | 80.63 | 62.61 | 40.33 | 20.36 | 38.53 | 70.03 |
<!-- 保留现有排行榜内容 -->

[View Complete Online Leaderboard](https://chenjy16.github.io/modelrank_ai/)

## Domain-Specific Leaderboards

Domain-specific model leaderboards can be accessed via the following links:

- [Medical Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/medical_leaderboard.html)
- [Legal Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/legal_leaderboard.html)
- [Finance Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/finance_leaderboard.html)

## 🏥 Medical Domain Leaderboard

*Top 10 models shown. Last updated: 2025-04-07 06:36:11 UTC*

| Rank | Model | Avg Score | Params(B) | MMLU-PRO | BBH | GPQA |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [<a target="_blank" href="https://huggingface.co/MaziyarPanahi/calme-3.2-instruct-78b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">MaziyarPanahi/calme-3.2-instruct-78b</a>  <a target="_blank" href="https://huggingface.co/datasets/open-llm-leaderboard/MaziyarPanahi__calme-3.2-instruct-78b-details" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">📑</a>](https://huggingface.co/MaziyarPanahi/calme-3.2-instruct-78b) | 57.87 | 78.0 | 70.03 | 62.61 | 20.36 |
<!-- 保留现有医疗领域排行榜内容 -->

[View Full Medical Leaderboard](https://chenjy16.github.io/modelrank_ai/medical_leaderboard.html)

## ⚖️ Legal Domain Leaderboard

*Top 10 models shown. Last updated: 2025-04-07 06:36:11 UTC*

| Rank | Model | Avg Score | Params(B) | MMLU-PRO | BBH |
| --- | --- | --- | --- | --- | --- |
| 1 | [<a target="_blank" href="https://huggingface.co/MaziyarPanahi/calme-3.2-instruct-78b" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">MaziyarPanahi/calme-3.2-instruct-78b</a>  <a target="_blank" href="https://huggingface.co/datasets/open-llm-leaderboard/MaziyarPanahi__calme-3.2-instruct-78b-details" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">📑</a>](https://huggingface.co/MaziyarPanahi/calme-3.2-instruct-78b) | 67.06 | 78.0 | 70.03 | 62.61 |
<!-- 保留现有法律领域排行榜内容 -->

[View Full Legal Leaderboard](https://chenjy16.github.io/modelrank_ai/legal_leaderboard.html)

## 💰 Finance Domain Leaderboard

*Top 10 models shown. Last updated: 2025-04-07 06:36:11 UTC*

| Rank | Model | Avg Score | Params(B) | MATH Lvl 5 | MMLU-PRO | BBH |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [<a target="_blank" href="https://huggingface.co/maldv/Awqward2.5-32B-Instruct" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">maldv/Awqward2.5-32B-Instruct</a>  <a target="_blank" href="https://huggingface.co/datasets/open-llm-leaderboard/maldv__Awqward2.5-32B-Instruct-details" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">📑</a>](https://huggingface.co/maldv/Awqward2.5-32B-Instruct) | 58.34 | 32.8 | 62.31 | 52.48 | 57.21 |
<!-- 保留现有金融领域排行榜内容 -->

[View Full Finance Leaderboard](https://chenjy16.github.io/modelrank_ai/finance_leaderboard.html)

## Evaluation Metrics Explanation

The leaderboard includes the following main evaluation metrics:

- **Average ⬆️**: Average score of all evaluations
- **IFEval**: Instruction following capability evaluation
- **BBH**: Big-Bench Hard benchmark for large language models
- **MATH Lvl 5**: Mathematical problem-solving capability evaluation
- **GPQA**: General Physics Question Answering evaluation
- **MUSR**: Multi-step reasoning evaluation
- **MMLU-PRO**: Massive Multitask Language Understanding Professional version evaluation

## Local Development

### Prerequisites

- Python 3.10+
- HuggingFace API token

### Installation Steps

1. Clone the repository
   ```bash
   git clone https://github.com/chenjy16/modelrank_ai.git
   cd modelrank_ai
   ```


## License

This project is open-sourced under the MIT License.




## Data Source

Data is sourced from HuggingFace.

