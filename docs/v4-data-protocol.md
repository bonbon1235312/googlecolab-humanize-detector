# V4 data protocol

V4 compares four model families against the same provenance-safe data partitions. It is an educational humanized-AI likelihood experiment, not authorship proof or an academic-integrity decision tool.

## Roles

Every V4 record belongs to exactly one role:

- `train`: fitting models and train-only feature/vectorizer fitting;
- `development`: choosing a model/configuration;
- `calibration`: fitting probability mappings and fixed-human-FPR thresholds after selection;
- `sealed_test`: final evaluation only, after all model and calibration choices are frozen.

`sealed_test` records must set `sealed=true` and `train_eligible=false`. Non-final loaders deliberately reject a sealed partition. The known V3 GRADTEX Test C result remains a regression benchmark; do not overwrite it or present it as V4’s sealed final result.

## Record contract

Every record has an immutable `id`, atomic `lineage_id`, normalised `text`, canonical `text_sha256`, binary label, source/domain, detailed provenance, generator/editor/transformation families, role, sealing state, training eligibility, and optional `parent_id`. Dataset-specific, text-safe identifiers such as an external source ID or attack identifier belong in `source_fields`; raw text, prompts, and generations must never be copied there.

Binary labels are `0` for genuine human-origin text—including benignly polished human text—and `1` for AI-origin text, including humanized or otherwise transformed AI text. Parent and child variants must remain in the same split.

The writer rejects cross-split lineages, parent relationships, exact duplicates, and near duplicates at 0.85 five-word-shingle Jaccard similarity.

## Writing a controlled manifest in Colab

Raw V4 data and model artifacts belong in ignored locations such as `v4-data/` and `v4-artifacts/`; never commit a corpus or sealed benchmark text.

```python
from pathlib import Path

from humanized_detector.v4_manifest import V4Record
from humanized_detector.v4_prepare import write_v4_dataset

# Build records only from explicitly train-eligible sources.
# Do not include GRADTEX or another sealed benchmark in this input.
records = [V4Record.from_mapping(row) for row in source_rows]

report = write_v4_dataset(
    records,
    Path('/content/drive/MyDrive/v4-data/control'),
    {'source_locator': '...', 'revision': '...'},
)
print(report['metadata_manifest_sha256'])
```

The writer produces one JSONL file per role plus `metadata_manifest.json`. The metadata manifest is safe to publish because it excludes text; it stores IDs, lineage IDs, text hashes, split counts, and the supplied source metadata. A manifest that contains `sealed_test` rows additionally requires `source_locator`, `revision`, `raw_download_sha256`, `row_selection_rule`, `selection_seed`, and `sealed_at`.

## What comes next

The next V4 step is selecting and sealing a public reproducible benchmark before model training. Its exact revision, deterministic filtering rule, lineage strategy, raw-download hash, and metadata-manifest digest must be recorded first. Only then may the control and hard-negative/matched-mirror intervention datasets be assembled.
