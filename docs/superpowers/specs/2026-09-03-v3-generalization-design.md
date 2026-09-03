# V3 Generalization Protocol

## Goal

Build a reproducible, from-scratch humanized-AI likelihood experiment that learns from mixed PADBen and Beemo data without leaking Beemo prompt lineages, and evaluates once on a separately frozen external benchmark. It is an educational detector, not authorship proof or an academic-integrity decision tool.

## Fixed data roles

| Dataset | Role | Permitted use |
| --- | --- | --- |
| PADBen `exhaustive-task5` | training source | train only after deterministic duplicate removal |
| Beemo | development source | lineage-isolated train, development, and calibration partitions |
| GRADTEX Test C | final external OOD benchmark | download, hash, and manifest only until model selection and calibration finish |

GRADTEX Test C is the final external benchmark because its four held-out scenario families are sentence-level polishing, middle completion, skeptical style rewriting, and maximal style rewriting. Final reporting must include an aggregate and one result per family. The project pins Hugging Face revision `553d859da0255d75a39c385c208f7522a2007f53`, the downloaded `test.parquet` SHA-256, source URL, split name, and a generated row-ID manifest before V3 training. Its texts and labels are never loaded by a V3 training, selection, or calibration command.

## Provenance and splitting

Every Beemo `prompt_id` is an atomic lineage. Its human response, raw `model_output`, expert-edited output, and every GPT-4o/Llama edit remain in one partition. Rows retain `source`, `lineage_id`, and `provenance` metadata:

- `human` has binary label `0`;
- `raw_ai`, `expert_edited_ai`, and `llm_edited_ai` have binary label `1`;
- provenance is retained even where the binary head uses the same positive label.

The Beemo partitioner uses seed `20260903` and assigns prompt groups to train (70%), development/model selection (15%), and calibration (15%). It never produces a Beemo test claim. It reports group IDs and per-provenance counts in an ignored manifest.

Before splitting, text is Unicode-normalised and exact hashes are recorded. Near-duplicate detection uses a deterministic 5-word-shingle signature; any cross-partition candidate at or above Jaccard similarity `0.85` is reported and causes preparation to fail rather than silently permit leakage. Exact duplicate text across different lineages similarly causes preparation to fail. Same-lineage variants remain allowed because the complete lineage is atomic.

Mixed training uses source-and-label-balanced sampling: PADBen human, PADBen positive, Beemo human, and Beemo positive each receive equal sampling mass, subject to non-empty strata. This is a training sampler only; deployment calibration reflects the chosen calibration population and never inherits the sampler prior.

## V3 models

All text variants share the existing byte-level BPE vocabulary and tiny Transformer encoder. A passage is represented with up to three 256-token windows: beginning, middle, and end. The same encoder processes each valid window. Models compare mean pooling with learned attention pooling over valid window embeddings.

The first structural cache contains deterministic, dependency-free statistics: word and sentence counts; sentence-length mean, standard deviation, and skew proxy; paragraph statistics; type-token ratio; hapax rate; character entropy mean and standard deviation across windows; word-length statistics; repeated-token rate; contraction rate; function-word ratio; newline density; and separate punctuation densities for semicolons, colons, hyphens, commas, exclamation marks, and question marks. The feature cache is keyed by immutable sample ID and content hash. Feature means and standard deviations are fitted only on the training partition and stored with the artifact.

The ablations are fixed before training:

1. multi-window Transformer with mean pooling;
2. multi-window Transformer with learned attention pooling;
3. structural-feature MLP only;
4. Transformer plus structural concatenation;
5. Transformer plus gated structural fusion and feature dropout.

No syntactic parser, contrastive loss, or domain-adversarial head is part of this initial V3 ablation. They are follow-up work only if the frozen evaluation demonstrates a remaining source-specific failure.

## Training, selection, and calibration

All variants retain AdamW, cosine scheduling, 0.15 Transformer dropout, 0.01 weight decay, and 0.1 binary label smoothing. The selected model is the highest Beemo-development ROC-AUC. Ties use PR-AUC, then lower human false-positive rate. The operating threshold and its 5% human-false-positive-rate guard are selected only on the separate calibration partition.

Temperature scaling and operating thresholds are fitted only after model selection, using the separate Beemo calibration partition. The public demo reports likelihood bands rather than authorship verdicts and has an abstention region selected on that calibration partition. It never presents the balanced-sampler output as a population probability.

## Metrics and reporting

For Beemo development, calibration, and the one-time GRADTEX Test C evaluation, report ROC-AUC, PR-AUC, F1, precision, recall, accuracy, Brier score, expected calibration error, confusion matrix, human false-positive rate, true-positive rate at 1% false-positive rate when supported by the human sample size, and abstention coverage/accuracy.

Report metrics by data source and Beemo provenance. GRADTEX Test C additionally reports each unseen scenario family individually and the aggregate. Confidence intervals use a seeded, lineage-group bootstrap where group identifiers exist; otherwise row bootstrap is labelled as such. Every artifact stores the seed, split manifest hash, dataset revision/hash, tokenizer hash, configuration, feature schema, checkpoint hash, ONNX hash, metrics, and timestamp.

## Acceptance criteria

The repository must have tests proving raw Beemo AI expansion, atomic prompt-lineage assignment, duplicate-boundary rejection, training-only feature normalisation, deterministic window extraction, source-balanced weights, model forward shapes for each ablation, calibration isolation, and per-scenario metric grouping. Colab commands prepare data and cache features without committing any corpus or model to Git. A V3 result may claim only the exact frozen benchmark and scenario coverage measured.
