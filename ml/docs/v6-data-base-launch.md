# V6 DATA-BASE launch

Only launch this run from the authorized V6 export. It intentionally keeps the
V4.8 model contract unchanged: 5M fusion-concat, masked-mean token pooling,
three 256-token windows, 4,000-token byte-BPE configuration, the existing
feature extraction, AdamW settings, loss, and six epochs.

The data-specific differences are the frozen V6 export and its audited
sampling weights. The launcher embeds the three V6 provenance hashes in the
checkpoint and refuses a data directory that claims sealed input was read.

```python
%cd /content
!git clone --branch codex/v5-1-12m https://github.com/bonbon1235312/googlecolab-humanize-detector.git humanized-ai-likelihood
%cd /content/humanized-ai-likelihood/ml
!pip -q install -e .

V6_DATA_DIR = "/content/drive/MyDrive/v6-data/base-v1"
V6_ARTIFACTS = "/content/drive/MyDrive/v6-artifacts/v4-8-data-base"

!python -u -m humanized_detector.v6_train \
  --data-dir "$V6_DATA_DIR" \
  --artifacts-dir "$V6_ARTIFACTS" \
  --epochs 6 \
  --batch-size 64 \
  --lr 3e-5 \
  --weight-decay 0.01 \
  --label-smoothing 0.1
```

Expected authorized contract:

- manifest SHA-256: `af1d8343379a05aac4b9a0409c3b0b00f49b92e1f936da0cc0092bffb0eb8501`
- train/development/calibration: `11060` / `1240` / `681`
- `sealed_data_loaded: false`

Do not read, mount, or evaluate RAID in this run. RAID remains a one-time
sealed final evaluation after DATA-BASE vs DATA-CURRICULUM selection,
calibration, and artifact freezing.
