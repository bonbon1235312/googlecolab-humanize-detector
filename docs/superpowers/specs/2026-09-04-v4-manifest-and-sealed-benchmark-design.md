# V4 Humanizer Data and Sealed-Benchmark Protocol

## Goal

Create the reproducible data-integrity layer for V4 humanized-AI likelihood experiments. It must support a controlled comparison of word TF-IDF + Logistic Regression, character n-gram TF-IDF + Logistic Regression, the existing 5M from-scratch Transformer, and one small pretrained prose encoder without allowing data, lineage, or model-selection leakage.

V4 asks one question:

> Can one controlled hard-negative and matched-mirror data intervention improve humanized-AI out-of-distribution detection across model families without increasing genuine-human false positives?

This work creates manifests, audits, and benchmark sealing only. It does not train V4 models, alter V3 artifacts, or re-evaluate GRADTEX Test C.

## Fixed task definition

The binary label is always:

- `0`: genuine human-origin text, including untouched human text and benignly grammar/style-polished human text;
- `1`: AI-origin text, including raw AI text, AI text that is paraphrased, style-rewritten, polished, partially rewritten, completed, or passed through a humanizer.

Each row retains the more specific `provenance`, `generator_family`, `editor_family`, and `transformation_family` fields. A binary classifier must not discard this evidence from its manifest, even where it trains on the binary label.

The humanizer task remains distinct from raw AI-vs-human detection. Only shared non-task-specific utilities, such as manifests, duplicate checks, calibration mechanics, and evaluation formatting, may be reused later.

## Data roles

V4 has four non-interchangeable data roles:

| Role | Permitted use |
| --- | --- |
| `train` | Model fitting and train-only feature/vectorizer fitting. |
| `development` | Model-family and configuration selection only. |
| `calibration` | Post-selection probability/threshold calibration only. |
| `sealed_test` | One-time final evaluation after every model, configuration, calibrator, and threshold is fixed. |

The V3 GRADTEX Test C result is a known regression benchmark only. Its already observed metrics must never be used as a pristine V4 final result. V4 preparation must not modify V3 data, checkpoints, reports, or GRADTEX files.

The V4 final test will be publicly reproducible: a public source is downloaded at a pinned revision, filtered deterministically, hashed, and represented by a separate sealed manifest. Raw benchmark text and labels are ignored by Git. The V4 training, development, and calibration commands must not accept a sealed manifest as an input.

## Manifest schema

Every record is JSON-serializable and has the following required fields:

| Field | Meaning |
| --- | --- |
| `id` | Immutable unique sample ID. |
| `lineage_id` | Atomic origin lineage; variants from one original prompt/document share this value. |
| `text` | Normalised text, stored only in ignored data files. |
| `text_sha256` | SHA-256 of Unicode-normalised text. |
| `label` | Binary `0` human-origin or `1` AI-origin label. |
| `source` | Dataset or collection name. |
| `domain` | Source domain, or `unknown` when no reliable metadata exists. |
| `provenance` | Specific provenance such as `human`, `human_polished`, `raw_ai`, or `ai_humanized`. |
| `generator_family` | Generator family, `human`, or `unknown`. |
| `editor_family` | Editor/humanizer family, `none`, or `unknown`. |
| `transformation_family` | Transformation type, including `none`, `grammar_polish`, `paraphrase`, `style_rewrite`, `partial_rewrite`, or `completion`. |
| `split` | Exactly one of `train`, `development`, `calibration`, or `sealed_test`. |
| `sealed` | `true` only for `sealed_test` records. |
| `train_eligible` | `true` only for records that can be included in training. |
| `parent_id` | Optional immediate source record ID for generated/polished/mirror variants. |

The manifest writer produces one data JSONL file per role and one public metadata-only manifest. The metadata manifest contains no text: it provides row counts, record IDs, lineage IDs, text hashes, source revisions, filtering rules, and a deterministic SHA-256 digest of its canonical JSON representation.

## Integrity requirements

- A `lineage_id` is atomic. It cannot occur in two distinct splits.
- Exact duplicate normalised text cannot cross split boundaries.
- Cross-split near duplicates at or above 0.85 Jaccard similarity over deterministic 5-word shingles cause preparation to fail.
- Each parent/child relationship must remain in one split. A non-empty `parent_id` must resolve to a manifest record.
- `sealed_test` records require `sealed=true` and `train_eligible=false`; all other split roles require `sealed=false`.
- A sealed benchmark source must include a source locator, pinned revision/version where available, raw-download SHA-256, row-selection rule, and selection seed in its metadata manifest.
- Every hard human, matched AI mirror, humanized mirror, and benignly polished human must include a lineage and parent relationship sufficient to reconstruct its origin.
- Hard-negative mining can only score and select records designated `train_eligible=true`; it cannot draw from development, calibration, sealed, or V3 GRADTEX rows.

## Controlled V4 data intervention

The control manifest is the existing eligible mixed source data, represented in V4 schema. The intervention manifest adds one fixed round of records:

1. score a large train-eligible human pool with V3;
2. select high-scoring human false positives while capping selections per source/domain;
3. remove exact and near duplicates and any lineage intersecting an evaluation partition;
4. create matched AI mirrors with similar topic, length, register, and formatting;
5. create AI-origin humanized variants across several transformation families;
6. add benignly polished genuine-human variants as negative examples;
7. record every transformation and parent relationship.

The control and intervention manifests must use the same development, calibration, and sealed-test partitions. The only intentional difference is the additional train-eligible intervention lineages.

## Benchmark sealing procedure

Before any V4 model fitting:

1. select the public benchmark source and exact revision;
2. define the row filtering/label conversion rule and lineage strategy;
3. deterministically select only complete lineages into `sealed_test`;
4. run integrity audits against all non-sealed partitions;
5. write and hash the sealed metadata manifest; and
6. archive the source metadata, manifest digest, and an explicit `sealed_at` timestamp.

After this point, only the final V4 evaluator may load the sealed data. Its output is written to a new V4 artifact directory and cannot alter model selection or calibration inputs.

## Evaluation contract

Every V4 model family consumes exactly the same manifest partitions. It is promoted only if, relative to its matching control run, it:

1. improves macro OOD ROC-AUC;
2. improves TPR at fixed 1%, 2%, and/or 5% human-FPR operating points; and
3. does not introduce a material worst-source human-FPR regression.

Evaluation records aggregate metrics and source, provenance, editor-family, transformation-family, and domain breakdowns where those groups contain both binary classes. For one-class scenario slices, report count and recall/false-positive rate as applicable, not misleading precision or F1.

The four planned model tracks are word TF-IDF + Logistic Regression, character n-gram TF-IDF + Logistic Regression, the existing 5M custom Transformer, and one explicitly named pretrained prose encoder. They will be implemented in later V4 work after this data contract is verified.

## Acceptance criteria

Tests must demonstrate required-field validation, sealed-role constraints, atomic lineage assignment, parent-child boundary rejection, exact and near-duplicate cross-split rejection, deterministic metadata-manifest digests, metadata-manifest text exclusion, and rejection of sealed data by train/development/calibration loaders.

The repository documents the schema and makes the V4 manifest tooling usable in Colab without committing any corpus, model artifact, or sealed-benchmark text to Git.
