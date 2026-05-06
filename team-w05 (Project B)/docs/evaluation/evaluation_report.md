# Evaluation Report - Mental Health Bot

## Run Configuration
- Agent URL: `http://localhost:8000`
- Sample size per dataset: `20`
- Random seed: `42`
- Claude grading enabled: `True`
- Claude model: `claude-sonnet-4-20250514`

## Metric Results
- **Accuracy (MedQA):** 25.00%
- **Patient Clarity (HealthSearchQA):** 4.90 / 5
- **Crisis Detection Rate:** 100.00%
- **False Positive Rate:** 0.00%
- **Empathy Score (HealthSearchQA):** 4.25 / 5
- **PubMed Retrieval Quality:** 1.60 / 5
- **PubMed Citation Coverage:** 25.00%

## Artifacts
- Raw results JSON: `docs/evaluation/results.json`
- Bar chart: `docs/evaluation/results_bar_chart.png`

## Notes
- Accuracy is measured on sampled MedQA multiple-choice questions.
- Clarity, empathy, and PubMed relevance are graded by Claude (or fallback heuristics).
- Crisis metrics use curated prompt sets with expected behavior checks.
