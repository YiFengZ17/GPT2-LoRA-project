# Report build notes

- Delivery: self-contained portable HTML.
- Audience: technical (student and research supervisor).
- Question: whether LoRA provides a better SST-5 performance–efficiency tradeoff than a frozen encoder or full fine-tuning, and which tested rank should be carried forward.
- Comparison basis: six configurations, five seeds per configuration, five epochs per run, identical dataset splits.

## Required-structure mapping

- Title: title block.
- Technical summary: `technical_summary` plus headline metric strip.
- Key findings with visual evidence: performance, efficiency, and overfitting sections.
- Scope, data, and metric definitions: `scope`.
- Methodology and validation: `methodology`.
- Limitations and uncertainty: `limitations`.
- Recommended next steps: `next_steps`.
- Further questions: `questions`.

## Chart map

1. `test_performance`: grouped bar chart; compares test accuracy and macro-F1 across the six configurations on one percentage-point scale. Supports the performance–efficiency finding.
2. `generalization_gap`: single-series bar chart; compares final-epoch train–validation accuracy gaps. Supports the overfitting finding.

Error bars are not supported by the selected native chart contract, so sample standard deviations are retained in tooltips, metric cards, and the exact comparison table. Resource measures are kept in the table because parameter count, checkpoint size, memory, and time have incompatible units and scales. Timing is descriptive because the original and added-seed launcher waves used different concurrency.

## Reproducibility

`analysis/analyze_results.py` performs the quality checks, verifies the recorded training-code hashes, and recomputes all aggregates using Python's standard library. `analysis/build_report_artifact.py` converts reviewed outputs into the canonical report artifact. The source archive remains unchanged in the Windows Downloads folder; the extracted copy is under `runs/main-study`.
