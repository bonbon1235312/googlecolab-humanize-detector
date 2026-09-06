# V6 DATA-BASE result

Status: **completed, not promotable**.

The run used the manifest-authorized V6 export with the unchanged V4.8 model
contract: 5M fusion-concat, masked-mean pooling, three 256-token windows,
4,000-token byte-BPE configuration, existing structural features, AdamW,
label smoothing `0.1`, and six epochs.

## Provenance

- V6 manifest SHA-256:
  `af1d8343379a05aac4b9a0409c3b0b00f49b92e1f936da0cc0092bffb0eb8501`
- train / source-held selection-dev / calibration rows: `11060` / `1240` /
  `681`
- checkpoint provenance includes the manifest, sampling-policy, and dedup
  policy hashes.
- `sealed_data_loaded` was `false` in the export contract and checkpoint.
- RAID text was not read.

## Selection-dev result

| Epoch | ROC-AUC |
| --- | ---: |
| 1 | 0.4700 |
| 2 | 0.4596 |
| 3 | 0.4813 |
| 4 | 0.4860 |
| 5 | **0.4896** |
| 6 | 0.4890 |

The selected checkpoint was epoch 5. Its source-held selection-dev metrics
were ROC-AUC `0.4896`, PR-AUC `0.5166`, and TPR@1% FPR `0.0045`.

This is below chance-like ranking on the new held-out source view and therefore
does not satisfy the predeclared conditions for a V4.8 replacement. Do not
calibrate, promote, or run DATA-CURRICULUM from this checkpoint.

The artifact archive was preserved locally as
`v6-artifacts/v6-data-base-final-artifacts.tar.gz`, SHA-256
`d44859b16d4b9de3296c197268e6d52fd0c99b339d0b961cb05535c72251254b`.

The known Beemo regression view and sealed RAID evaluation were intentionally
not used to rescue or tune this non-promotable candidate.
