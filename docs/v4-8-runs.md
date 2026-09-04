# V4.8 diagnostic runs

V4.8 is a controlled follow-on to the frozen V4.0 control. It does not rebuild data, touch GRADTEX, or read the sealed RAID-derived cohort. V4.0's first-token checkpoint and its original results remain unchanged.

The selection order is deliberate:

1. Compare masked-mean pooling against V4.0's first-token pooling using only the existing development split.
2. Inspect fit diagnostics and the independent unused-PADBen diagnostic.
3. Only if those diagnostics justify it, run one predeclared optimiser recipe.
4. Choose one V4.8 candidate from development results, then calibrate it once. Calibration is not a model-selection leaderboard.

## 1. Refresh Colab

```python
%cd /content/humanized-ai-likelihood
!git pull -q
%cd /content/humanized-ai-likelihood/ml
!pip install -q -e .

from pathlib import Path

CONTROL_DATA_DIR = Path('/content/drive/MyDrive/v4-data/control-v1')
PADBEN_DIAGNOSTIC_DIR = Path('/content/drive/MyDrive/v4-data/padben-unused-diagnostic-v1')
V48_ROOT = Path('/content/drive/MyDrive/v4-artifacts/v4-8')
```

## 2. Create the unused PADBen diagnostic once

This uses PADBen records excluded from every non-sealed V4 control role. It is a diagnostic cohort only: do not train or calibrate on it.

```python
from datasets import load_dataset
from humanized_detector.v4_control_data import prepare_padben_diagnostic

padben_rows = [dict(row) for row in load_dataset('JonathanZha/PADBen', 'exhaustive-task5', split='train')]
print(prepare_padben_diagnostic(padben_rows, CONTROL_DATA_DIR, PADBEN_DIAGNOSTIC_DIR, samples_per_class=1000))
```

## 3. Controlled architecture run

This is the first V4.8 training run. It changes only within-window token pooling from `first` to `masked_mean`; data, capacity, epochs, seed and core optimiser settings remain V4.0-compatible.

```python
!python -u -m humanized_detector.v4_train \
  --data-dir $CONTROL_DATA_DIR \
  --artifacts-dir $V48_ROOT/masked_mean_base \
  --capacity 5m --token-pooling masked_mean \
  --epochs 6 --batch-size 64 --lr 3e-5 --weight-decay 0.01
```

## 4. Read-only diagnostics

Run this after the checkpoint exists. It reads only `train.jsonl`, `development.jsonl`, and the separate PADBen diagnostic. It does not load calibration or sealed data.

```python
from humanized_detector.v4_diagnostics import score_padben_diagnostic, write_v4_fit_diagnostics

ARTIFACTS_DIR = V48_ROOT / 'masked_mean_base'
fit = write_v4_fit_diagnostics(CONTROL_DATA_DIR, ARTIFACTS_DIR, bootstrap_iterations=1000)
padben = score_padben_diagnostic(PADBEN_DIAGNOSTIC_DIR, ARTIFACTS_DIR)
print('Train/dev AUC gap:', fit['train_development_roc_auc_gap'])
print('Development subtype metrics:', fit['development']['subtypes'])
print('Unused PADBen diagnostic:', padben)
```

Interpret the fit gap before changing the learning rate. Do not call an optimisation problem "underfitting" without evidence from these diagnostics.

## 5. One conditional optimiser run

Only run this if the masked-mean diagnostic supports testing optimisation (for example, no large train-to-development ranking gap). It is one prespecified companion recipe, not an open-ended sweep: `1e-4` learning rate, 400-step warm-up, clip norm 1.0, and label smoothing 0.02.

```python
!python -u -m humanized_detector.v4_train \
  --data-dir $CONTROL_DATA_DIR \
  --artifacts-dir $V48_ROOT/masked_mean_optimised \
  --capacity 5m --token-pooling masked_mean \
  --epochs 6 --batch-size 64 --lr 1e-4 --weight-decay 0.01 \
  --label-smoothing 0.02 --warmup-steps 400 --grad-clip-norm 1.0
```

Use the same diagnostics cell on this artefact. Select at most one model using development ranking and subtype results; retain the other run as an experiment record.

## 6. Cross-fitted Platt calibration after selection

Only after one candidate has been chosen, use the existing calibration partition. This reports raw score quality plus group-cross-fitted Platt diagnostics by `lineage_id`, and saves a final full-calibration scaler for deployment. The cross-fitted values are a calibration-quality assessment, not another way to select between models.

```python
SELECTED_ARTIFACTS = V48_ROOT / 'masked_mean_base'  # replace only after development selection is written down
!python -u -m humanized_detector.v4_calibrate --data-dir $CONTROL_DATA_DIR --artifacts-dir $SELECTED_ARTIFACTS
```

The output preserves raw-probability thresholds at 1%, 2%, and 5% human FPR for comparison with V4.0. It also reports out-of-fold Platt Brier score/ECE and records the final scaler parameters. Calibration cannot repair a poor ROC-AUC; it makes confidence and thresholds more honest for the calibration distribution.

## Boundaries

- Never train on, inspect, calibrate against, or evaluate the sealed RAID-derived cohort before V4.8 selection and calibration are frozen.
- GRADTEX is a known regression benchmark, not a tuning target.
- The PADBen diagnostic answers an in-source question only. It does not replace Beemo development or sealed OOD evaluation.
