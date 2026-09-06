# V6 pretrain manifest

V6 does not authorize training until this command finishes with
`"TRAINING AUTHORIZED": "YES"`.

## Beemo adapter

Run this immediately after acquiring the pinned Beemo parquet. It writes an
**unsplit** candidate pool and prints the raw download SHA-256. It refuses an
AI-origin row when the source record lacks an explicit generator family.

```bash
python -m humanized_detector.v6_beemo \
  --input-parquet /content/drive/MyDrive/v6-raw/beemo.parquet \
  --output-jsonl /content/drive/MyDrive/v6-normalized/beemo_candidates.jsonl
```

Splitting and sampling occur later, across the combined candidate pool.

```bash
python -m humanized_detector.v6_manifest \
  --records-jsonl /content/drive/MyDrive/v6-data/normalized_records.jsonl \
  --sources-json /content/drive/MyDrive/v6-data/sources.json \
  --sampling-policy-json /content/drive/MyDrive/v6-data/sampling_policy.json \
  --sealed-eval-metadata-json /content/drive/MyDrive/v6-data/sealed_raid_metadata.json \
  --output /content/drive/MyDrive/v6-data/v6_manifest.json \
  --isolate-fields template_id
```

`normalized_records.jsonl` contains one record per line. It must include every
field below; use `not_applicable` where a field truly does not apply, never a
blank value, `TBD`, `unknown`, or `null`.

```json
{
  "id": "beemo:expert:source-001",
  "lineage_id": "beemo:source-001",
  "parent_lineage_id": null,
  "text": "The source text, held only in the private input JSONL.",
  "dataset": "beemo",
  "domain": "instruction_response",
  "source_family": "no_robots",
  "template_id": "not_applicable",
  "origin": "ai",
  "edit_actor": "human_expert",
  "editor_family": "beemo_expert_editor",
  "generator_family": "generator_name",
  "transformation": "expert_edit",
  "split": "train",
  "sampling_weight": 1.0
}
```

`origin` is only `human` or `ai`; labels are derived, never supplied.
`origin=human, edit_actor=llm` is therefore a negative. `origin=ai,
edit_actor=human_expert` is a positive.

```json
[
  {
    "dataset": "beemo",
    "revision": "pinned-source-revision",
    "download_sha256": "<64 lowercase hex characters>",
    "license": "verified licence identifier",
    "deployment_allowed": true,
    "provenance_quality": "documented",
    "permitted_roles": ["train", "selection_dev", "calibration"]
  }
]
```

The policy JSON must contain `class_balance`, `source_weights`,
`subtype_weights`, `length_weights`, `max_variants_per_lineage`, `jfleg_cap`,
`humanizerbench_cap`, and `mage_cap`. Every value must be established.

The sealed RAID metadata JSON is identifiers, hashes, revision, and expected
count only. It must not contain text, prompts, generations, or content.

The generated `v6_manifest.json` deliberately excludes all raw text. Its
hashes are the provenance values stored in each V6 checkpoint.
