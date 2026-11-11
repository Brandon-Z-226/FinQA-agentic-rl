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

## Attempt 2: stopped after complete pre-training validation

- Git commit: `27582bc78f60a090b7698af2bdee4a526691de43`
- Remote run directory: `/root/autodl-tmp/finqa/experiments/full-baseline-retry1b-20261111T151000Z`
- Result: all 522 pre-training validation examples completed, then the judge provider exhausted its free quota during the first 64-rollout training batch. No optimizer step ran and no checkpoint was created.
- Wall time: 3,448.9 seconds
- GPU window cost at CNY 5.59/hour: CNY 5.36
- Base-model validation: accuracy/pass@1 0.647510, mean reward 0.651515, and table access 0.793103
- Judge SDK attempts: 560 (546 successful responses and 14 permission errors)
- Judge usage from successful responses: 354,722 prompt tokens and 119,992 completion tokens, including 91,057 reasoning tokens
- Judge cost at the recorded CNY 12/CNY 36 per million input/output tokens: CNY 8.58

The provider returned HTTP 403 with `Free quota exhausted` and instructed the account owner to add funds or disable its free-tier-only setting. The evaluator exposed the failure and stopped training rather than converting it to reward zero. After account access is restored and a live probe succeeds, the next run can reuse this preserved step-0 validation result and disable repeated pre-training validation; it must still start from the original base model and run the complete 522-example post-training validation.
