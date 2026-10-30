---
language:
- en
license: apache-2.0
size_categories:
- 1K<n<10K
task_categories:
- question-answering
tags:
- finance
- reinforcement learning
- agents
---

## Data

Our dataset consists of financial Q&A pairs derived from SEC 10-K filings across 207 public companies:
- 5,110 single-table questions.
- 1,248 multi-table questions.
- 6,923 financial tables.

## Format

Each row in the dataset contains:
- **user_query**: The question with company context.
- **question**: The core question.
- **answer**: The ground truth answer.
- **explanation**: The step-by-step solution.
- **company**: The company name.
- **table_name**: The table(s) needed to answer.
- **question_type**: The category (e.g., "Net Change").

We note that single-table examples include additional keys for `columns_used_json` and `rows_used_json`.

## Example

```json
{
  "user_query": "For company `anthem`, here is the question: What is the year-over-year growth rate of total net incurred medical claims from 2022 to 2023?",
  "question": "What is the year-over-year growth rate of total net incurred medical claims from 2022 to 2023?",
  "answer": "6.82%",
  "explanation": "(120,227 - 112,545) / 112,545 = 0.0682",
  "company": "anthem",
  "table_name": "elv_ReconciliationofNetIncurredMedicalClaimstoBenefitExpenseTableTextBlock",
  "question_type": "Growth - YoY"
}
```

## Citation

```bibtex
@misc{rllm2026finqa,
  title={FinQA: Training Financial Agents with Reinforcement Learning},
  author={Manan Roongta and Sijun Tan and Bhavishya Pohani and Charles Dickens and Christopher Glaze},
  year={2026},
  howpublished={\url{https://rllm-project.com/post.html?post=finqa.md}},
  note={Blog Post}
}
```