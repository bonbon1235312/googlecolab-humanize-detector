# V5.1 12M curriculum experiment

## Purpose

Test whether a richer 12M representation can absorb the H1 expert-edited-AI
curriculum without sacrificing LLM-edited transfer. This is a release-oriented
bundle test, not an attribution study of individual architectural changes.

## Fixed data and isolation

- Use only `/content/drive/MyDrive/v4-data/control-v1` partitions.
- Never read GRADTEX, RAID, or any sealed cohort.
- Preserve the existing lineage-safe train, development, and calibration roles.
- Reuse the fixed seed and H1 hierarchical sampling schedule.

## Candidate architecture

The candidate has a shared 12M-class text encoder (384 hidden dimensions, six
layers, eight heads), masked-mean token pooling, three beginning/middle/end
windows, one lightweight cross-window self-attention block, FiLM conditioning
from the normalised structural vector, and a shallow binary head. The bundled
architecture is evaluated as a whole; a positive result does not attribute a
gain to any one component.

## Two-stage training

1. Train a balanced 12M base for six epochs with the V4 source/label-balanced
   sampling. Preserve its selected development checkpoint.
2. Continue that checkpoint for up to four H1 curriculum epochs. Within Beemo
   positives, the expert/LLM/raw mix is 25/50/25 at epoch one, 40/35/25 at
   epoch two, and 50/30/20 at epochs three and four. Epoch zero of continuation
   is the selected balanced 12M base and can remain the winner.

The first complete balanced epoch is timed. Continue only when the extrapolated
10-epoch job fits the release runway; otherwise retain V4.8 and do not begin a
partial V5.1 job.

## Selection and promotion

Select the best checkpoint inside the 12M candidate using development data
only. Compare that candidate with V4.8 using paired, lineage-group bootstrap
differences. Promotion requires credible positive macro-edit AUC improvement,
no meaningful expert- or LLM-edited AUC regression, and no meaningful subtype
TPR@5%-human-FPR regression under one global development-human threshold.
Borderline or inconclusive differences are no-promotion outcomes. Report the
expert-edited bootstrap result separately as improved, inconclusive, or
regressed; deployment promotion does not imply that expert-edit detection was
solved.

## Calibration and artifacts

Only after selection is frozen, fit the final Platt scaler and freeze 1%, 2%,
and 5% human-FPR thresholds on the separate calibration partition. Save
checkpoints, normaliser/tokenizer assets, run manifests, paired-bootstrap
reports, selection decision, and calibration files in a new V5.1 artifact
folder. V4.8 is never overwritten.
