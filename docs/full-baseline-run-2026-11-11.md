# Full GRPO baseline run log (2026-11-11)

## Attempt 1: stopped during pre-training validation

- Git commit: `39e30067597287feb7ddc0dc048900160560ae5e`
- Remote run directory: `/root/autodl-tmp/finqa/experiments/full-baseline-20261111T134100Z`
- Result: failed before the first optimizer step; no LoRA weights were updated and no training checkpoint was created.
- Wall time: 1,002.5 seconds
- GPU window cost at CNY 5.59/hour: CNY 1.56
- Completed validation examples before failure: 80
- Judge SDK/HTTP attempts: 107
- Judge usage: 70,007 prompt tokens and 28,188 completion tokens, including 23,704 reasoning tokens

The Qwen judge repeatedly reached the configured 512 completion-token limit with all 512 tokens reported as hidden reasoning. It consequently returned no visible `true` or `false` verdict. The evaluator exhausted its finite malformed-response retries and raised `JudgeResponseError` as designed, preventing the API failure from being recorded as reward zero.

The compatibility fix keeps the official 512-token default and adds the bounded `FINQA_JUDGE_SINGLE_MAX_OUTPUT_TOKENS` override. The AutoDL Qwen deployment uses 2,048 tokens so hidden reasoning has room to finish before the verdict. Judge model, prompt, parser, retry policy, reward, and GRPO configuration are unchanged. Audit events now include finish reason and visible-content presence/length without recording response content.
