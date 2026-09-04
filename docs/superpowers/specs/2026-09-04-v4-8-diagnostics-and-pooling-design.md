# V4.8 Diagnostics, Pooling, and Calibration Design

## Purpose

V4.8 determines whether the V4.0 custom Transformer is limited primarily by fit, pooling, source shift, or score calibration before changing the training data. It creates new V4.8 artifacts only; V4.0 artifacts and the sealed RAID-derived cohort remain immutable.

## Boundaries

- V4.0 remains the existing PADBen-plus-Beemo control reference.
- All V4.8 model selection reads only V4 `train` and `development` partitions.
- Calibration work reads only `calibration` after an architecture is selected. It never reads `sealed_test`.
- RAID remains one-time final evaluation and is excluded from every V4.8 command.
- GRADTEX remains a known regression benchmark and is excluded from V4.8 selection.
- V4.8 does not add generated or hard-negative data; that is a later V5 data intervention.

## Diagnostics

The selected V4.0 and V4.8 Transformer checkpoints must produce a text-free diagnostic report containing:

- aggregate train and development ROC-AUC, PR-AUC, Brier score, ECE, and their AUC gap;
- development human-negative count and one-negative FPR resolution;
- human-versus-positive-provenance metrics for `raw_ai`, `expert_edited_ai`, and `llm_edited_ai` when both classes exist;
- stratified bootstrap 95% ROC-AUC confidence intervals.

Training metrics are diagnostic only. They must not choose a checkpoint or threshold.

## Architecture ablation

V4.8 compares the existing first-token window representation with masked mean pooling across valid token states.

- The encoder still applies correct token padding masks.
- Cross-window learned attention pooling remains unchanged.
- Structural features, source/label sampler, data manifest, random seed, learning rate, epochs, batch size, dropout, smoothing, and weight decay remain unchanged for the pooling-only experiment.
- New checkpoints store `token_pooling` in `ModelConfig`; absent old checkpoint fields default to `first` for compatibility.

## Internal PADBen health check

Rows not selected into the V4.0 PADBen training sample form a separate `padben_diagnostic.jsonl` collection. It is never used for fitting, threshold selection, or aggregation with Beemo development. It answers only whether a model learns the PADBen source family in-domain.

## Calibration

V4.8 adds group-cross-fitted Platt calibration using Beemo prompt-family (`lineage_id`) folds.

- Out-of-fold calibrated probabilities report Brier score and ECE without scoring a calibrator on the rows that fitted it.
- The final deployable Platt model fits all calibration rows only after architecture selection is frozen.
- Fixed 1%, 2%, and 5% human-FPR thresholds are still fitted on all calibration rows after selection. FPR is reported with its discrete human-negative resolution.
- The API output remains a risk score unless a deployment class prior is explicitly chosen and recorded.

## Optimisation ablation

Only after diagnostics and pooling selection, run one explicit V4.8 training-recipe experiment:

- `lr=1e-4`;
- linear warm-up for 400 steps;
- gradient clipping at 1.0;
- label smoothing 0.02;
- same data, masked pooling choice, capacity, epochs, and checkpoint selection as the selected pooling run.

This is a compound training-recipe test and must be reported as such; it does not establish which individual setting caused any gain.

## Promotion evidence

V4.8 selects a model on development ranking metrics and confidence intervals. Calibration is used only to fit and report operating thresholds for the already-selected candidate. No V4.8 model is promoted merely for a higher aggregate AUC; report subtype results, fixed-FPR TPR, and uncertainty.
