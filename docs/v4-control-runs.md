# V4 control runs in Google Colab

These commands create the V4 control manifest and compare three models on the same eligible data: word TF-IDF + logistic regression, character n-gram TF-IDF + logistic regression, and the existing fusion-concat Transformer. They do not perform final calibration or evaluate `sealed_test`.

## 1. Bootstrap the runtime

```python
from pathlib import Path

if not Path('/content/humanized-ai-likelihood').exists():
    !git clone https://github.com/bonbon1235312/googlecolab-humanize-detector.git /content/humanized-ai-likelihood

%cd /content/humanized-ai-likelihood
!git pull -q
%cd /content/humanized-ai-likelihood/ml
!pip install -q -e .

from google.colab import drive
drive.mount('/content/drive')

CONTROL_DATA_DIR = Path('/content/drive/MyDrive/v4-data/control-v1')
ARTIFACTS_ROOT = Path('/content/drive/MyDrive/v4-artifacts/control-v1')
```

Select a T4 GPU before the Transformer run. The two TF-IDF baselines use CPU and should finish quickly.

## 2. Build the eligible control manifest

This recreates the V3-style PADBen + prompt-atomic Beemo control data in the stricter V4 schema. PADBen contributes only training rows. Beemo prompt lineages are split across train/development/calibration as whole groups.

```python
!wget -q https://raw.githubusercontent.com/Toloka/beemo/main/dataset.parquet -O /content/beemo.parquet

from datasets import load_dataset
from humanized_detector.beemo import load_beemo_parquet
from humanized_detector.v4_control_data import V4ControlDataConfig, prepare_v4_control_dataset

padben_rows = [dict(row) for row in load_dataset('JonathanZha/PADBen', 'exhaustive-task5', split='train')]
beemo_rows = load_beemo_parquet(Path('/content/beemo.parquet'))

report = prepare_v4_control_dataset(
    padben_rows,
    beemo_rows,
    CONTROL_DATA_DIR,
    V4ControlDataConfig(padben_samples_per_class=15_000),
)
print(report)
```

Do not place GRADTEX files, RAID files, or any previously sealed benchmark rows in `CONTROL_DATA_DIR`.

## 3. Run the classical controls

```python
!python -u -m humanized_detector.v4_baselines --data-dir $CONTROL_DATA_DIR --artifacts-dir $ARTIFACTS_ROOT/word_tfidf_lr --variant word_tfidf_lr
!python -u -m humanized_detector.v4_baselines --data-dir $CONTROL_DATA_DIR --artifacts-dir $ARTIFACTS_ROOT/char_tfidf_lr --variant char_tfidf_lr
```

Each run writes `model.joblib`, `development_metrics.json`, and non-text `development_predictions.jsonl`.

## 4. Run the 5M control Transformer

```python
!python -u -m humanized_detector.v4_train --data-dir $CONTROL_DATA_DIR --artifacts-dir $ARTIFACTS_ROOT/fusion_concat_5m --capacity 5m --epochs 6 --batch-size 64 --lr 3e-5 --weight-decay 0.01
```

The command prints development ROC-AUC after each epoch, keeps its best development checkpoint, removes the temporary tokenizer training corpus, and writes calibration predictions without selecting a threshold.

## 5. Optional 12M capacity ablation

Only run this after recording the 5M development metrics. It must use the exact same `CONTROL_DATA_DIR`.

```python
!python -u -m humanized_detector.v4_train --data-dir $CONTROL_DATA_DIR --artifacts-dir $ARTIFACTS_ROOT/fusion_concat_12m --capacity 12m --epochs 6 --batch-size 64 --lr 3e-5 --weight-decay 0.01
```

## What not to do yet

Do not read, train on, calibrate against, or evaluate `sealed_test`. Do not tune using GRADTEX or the sealed RAID-derived cohort. Once a single model/capacity is selected using development results, the next stage is threshold calibration from that selected model's `calibration_predictions.jsonl`; final RAID evaluation happens once after calibration is frozen.
