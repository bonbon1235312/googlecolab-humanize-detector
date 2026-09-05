# Humanized-AI Likelihood Detector

A small Transformer classifier trained from random weights to estimate whether English text resembles deeply paraphrased AI examples in PADBen. It is an educational experiment, not authorship proof or an academic-integrity tool.

## What is included

- deterministic PADBen `exhaustive-task5` preparation;
- byte-level BPE plus a compact four-layer Transformer;
- validation-F1 checkpoint selection and ONNX export;
- AdamW weight decay, cosine learning-rate scheduling, dropout, and binary label smoothing;
- a Colab notebook that stores only generated artifacts in Drive.

## Fastest path: Google Colab

Open `notebooks/train_humanized_ai_detector.ipynb` in Colab, select **T4 GPU**, and run cells from top to bottom. The notebook downloads PADBen to the runtime, saves the tuned model/tokenizer/metrics under `MyDrive/humanized-ai-likelihood-artifacts-v2`, and never puts a dataset or model in Git.

## Important limitation

The result is a similarity score relative to PADBen's deeply paraphrased-AI examples. It does not prove that a person used AI, a paraphrasing tool, or any particular humanizer. External Beemo evaluation is required before making a robustness claim.

## V4 provenance-safe data workflow

V4 starts with data integrity rather than a larger model: a shared manifest preserves lineage, transformation provenance, content hashes, and split roles for all model families. It blocks cross-split duplicate/near-duplicate leakage and prevents non-final workflows from reading sealed-test data. See [the V4 data protocol](docs/v4-data-protocol.md).

The sealed V4 RAID-derived paraphrase cohort is tracked by hash and provenance only in [the sealed benchmark registry](docs/v4-sealed-benchmark-registry.md); its texts remain outside Git and must not be used before final evaluation.

Run the shared V4 control models in Colab using [the V4 control-run guide](docs/v4-control-runs.md). It builds only train/development/calibration data and deliberately excludes the sealed RAID cohort.

## V5 H1 curriculum continuation

H1 continues the available V4.8 masked-mean checkpoint for four curriculum epochs. Sampling is hierarchical: PADBen human, PADBen humanized AI, Beemo human, and Beemo AI each receive 25% of the epoch; the Beemo-AI quarter progressively shifts toward expert-edited examples while lineages remain equally weighted. The untouched V4.8 checkpoint is also eligible to win, so continuation cannot silently replace it with a worse development checkpoint.

Open [`notebooks/train_v5_h1.ipynb`](notebooks/train_v5_h1.ipynb) in Colab and run it one cell at a time. The notebook performs a fail-fast lineage/checkpoint preflight, trains with per-update warmup and cosine decay, saves a resumable last state after every epoch, selects checkpoints on expert/LLM-edited macro AUC with low-FPR tie-breakers, and freezes 1%, 2%, and 5% human-FPR calibration points. It does not read GRADTEX or the sealed RAID-derived cohort.

## Sources

- [PADBen](https://huggingface.co/datasets/JonathanZha/PADBen), MIT licence.
- [Beemo](https://github.com/Toloka/beemo), retained only for separate evaluation; consult its source/model licence terms.
