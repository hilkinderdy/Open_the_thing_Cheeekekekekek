# LLM Explanation Evaluation (automated proxy)

Setup: 35 answers over sampled homes and 7 fixed questions (explanations and what-ifs). Calculations are rerun in code for every what-if before the LLM explains them. 9 answers came from the LLM; the rest used the template fallback.

| Metric | Result | Target |
|---|---|---|
| Groundedness (answers whose numbers all match the computed facts) | 100.0% | 90% or higher |
| Unsupported numbers (share of numbers not in the facts) | 0.0% | under 5% |
| Answers labelling figures as estimates | 74.3% | 100% |
| Infeasible retrofit requests handled in the answer | 94.3% | 100% |

## Limits of this evaluation
- It checks numbers against the computed facts automatically. It does not check wording, completeness or whether claims without numbers are true; a human review of a sample is still needed.
- Small sample, one model, synthetic data.
- Rounding to the displayed precision is accepted; any other number is flagged.
