# V4 Manifest and Sealed Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the V4 provenance manifest, integrity audits, and sealed-benchmark preparation tooling without training any V4 model or modifying V3 artifacts.

**Architecture:** `v4_manifest.py` owns schema validation, canonical hashes, metadata-only manifest production, and loading. `v4_audit.py` owns split/lineage/parent/duplicate integrity checks. `v4_prepare.py` is the small command-line orchestration layer that validates input JSONL records, writes role-partition JSONL files plus public metadata, and refuses sealed input in non-final roles. Each later model track reads the same validated partition files through this layer.

**Tech Stack:** Python 3.11+, stdlib (`dataclasses`, `hashlib`, `json`, `unicodedata`), existing pytest suite.

**Spec:** `docs/superpowers/specs/2026-09-04-v4-manifest-and-sealed-benchmark-design.md`

## Global Constraints

- Do not train V4 models, download a benchmark, generate mirrors, or modify V3/GRADTEX artifacts in this implementation.
- Store corpus text only in ignored JSONL data files; public metadata manifests must never include a `text` key or text contents.
- Binary labels are exactly `0` for human-origin and `1` for AI-origin.
- Valid roles are exactly `train`, `development`, `calibration`, and `sealed_test`.
- A sealed-test row has `sealed=true` and `train_eligible=false`; all other roles have `sealed=false`.
- Enforce atomic `lineage_id`, resolved same-split `parent_id`, exact duplicate rejection, and 5-word-shingle Jaccard cross-split rejection at `0.85`.
- Use deterministic canonical JSON and SHA-256 digests for public metadata.
- Use test-driven development: each production behavior begins with a focused failing pytest assertion.

---

### Task 1: Typed V4 records and metadata-only digest

**Files:**
- Create: `ml/humanized_detector/v4_manifest.py`
- Create: `ml/tests/test_v4_manifest.py`

**Interfaces:**
- Produces `V4Record.from_mapping(record: Mapping[str, object]) -> V4Record`.
- Produces `V4Record.to_row() -> dict[str, object]`.
- Produces `metadata_manifest(records: Sequence[V4Record], source_metadata: Mapping[str, object]) -> dict[str, object]`.
- Produces `manifest_digest(payload: Mapping[str, object]) -> str`.

- [ ] **Step 1: Write the failing record-validation test**

```python
from humanized_detector.v4_manifest import V4Record


def test_v4_record_requires_complete_provenance_and_correct_text_hash() -> None:
    record = V4Record.from_mapping({
        "id": "human:1", "lineage_id": "lineage:1", "text": "A human passage.",
        "text_sha256": "incorrect", "label": 0, "source": "example", "domain": "essay",
        "provenance": "human", "generator_family": "human", "editor_family": "none",
        "transformation_family": "none", "split": "train", "sealed": False,
        "train_eligible": True, "parent_id": None,
    })

    assert record.text_sha256 != "incorrect"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v4_manifest.py::test_v4_record_requires_complete_provenance_and_correct_text_hash -v`

Expected: FAIL because `humanized_detector.v4_manifest` does not exist.

- [ ] **Step 3: Implement the minimal record model**

```python
@dataclass(frozen=True)
class V4Record:
    id: str
    lineage_id: str
    text: str
    text_sha256: str
    label: int
    source: str
    domain: str
    provenance: str
    generator_family: str
    editor_family: str
    transformation_family: str
    split: str
    sealed: bool
    train_eligible: bool
    parent_id: str | None

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "V4Record":
        # normalise Unicode NFC; validate required values; compute the canonical hash
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_v4_manifest.py::test_v4_record_requires_complete_provenance_and_correct_text_hash -v`

Expected: PASS.

- [ ] **Step 5: Write the failing metadata privacy/determinism test**

```python
from humanized_detector.v4_manifest import V4Record, manifest_digest, metadata_manifest


def test_metadata_manifest_is_text_free_and_has_a_stable_digest() -> None:
    record = V4Record.from_mapping(make_row("a", "lineage:a", "Hidden source text.", "train"))

    first = metadata_manifest([record], {"source_locator": "https://example.test", "revision": "v1"})
    second = metadata_manifest([record], {"revision": "v1", "source_locator": "https://example.test"})

    assert "text" not in str(first)
    assert manifest_digest(first) == manifest_digest(second)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_v4_manifest.py::test_metadata_manifest_is_text_free_and_has_a_stable_digest -v`

Expected: FAIL because `metadata_manifest` and `manifest_digest` do not exist.

- [ ] **Step 7: Implement canonical metadata and digest helpers**

```python
def metadata_manifest(records: Sequence[V4Record], source_metadata: Mapping[str, object]) -> dict[str, object]:
    rows = [{key: value for key, value in record.to_row().items() if key != "text"} for record in records]
    return {"source": dict(source_metadata), "records": rows, "counts": dict(Counter(row["split"] for row in rows))}


def manifest_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 8: Run focused tests to verify they pass**

Run: `pytest tests/test_v4_manifest.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add ml/humanized_detector/v4_manifest.py ml/tests/test_v4_manifest.py
git commit -m "feat: add V4 provenance manifest records"
```

### Task 2: V4 cross-boundary integrity audit

**Files:**
- Create: `ml/humanized_detector/v4_audit.py`
- Modify: `ml/tests/test_v4_manifest.py`

**Interfaces:**
- Consumes `Sequence[V4Record]` from Task 1.
- Produces `audit_v4_records(records: Sequence[V4Record], near_duplicate_threshold: float = 0.85) -> V4Audit`.
- `V4Audit` exposes `checked_records: int` and `split_counts: Mapping[str, int]`.

- [ ] **Step 1: Write the failing sealed-role test**

```python
import pytest
from humanized_detector.v4_audit import audit_v4_records
from humanized_detector.v4_manifest import V4Record


def test_audit_rejects_a_train_eligible_sealed_record() -> None:
    row = make_row("sealed:1", "lineage:1", "sealed human text", "sealed_test")
    row.update({"sealed": True, "train_eligible": True})

    with pytest.raises(ValueError, match="sealed_test"):
        audit_v4_records([V4Record.from_mapping(row)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v4_manifest.py::test_audit_rejects_a_train_eligible_sealed_record -v`

Expected: FAIL because `humanized_detector.v4_audit` does not exist.

- [ ] **Step 3: Implement split/sealing invariants**

```python
def audit_v4_records(records: Sequence[V4Record], near_duplicate_threshold: float = 0.85) -> V4Audit:
    for record in records:
        if record.split == "sealed_test" and (not record.sealed or record.train_eligible):
            raise ValueError("sealed_test records must be sealed and not train eligible")
        if record.split != "sealed_test" and record.sealed:
            raise ValueError("only sealed_test records may be sealed")
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_v4_manifest.py::test_audit_rejects_a_train_eligible_sealed_record -v`

Expected: PASS.

- [ ] **Step 5: Write failing lineage/parent and near-duplicate tests**

```python
def test_audit_rejects_lineage_and_parent_crossing_boundaries() -> None:
    parent = V4Record.from_mapping(make_row("parent", "lineage:x", "one two three four five six", "train"))
    child = V4Record.from_mapping(make_row("child", "lineage:y", "a generated variant", "development", parent_id="parent"))

    with pytest.raises(ValueError, match="parent"):
        audit_v4_records([parent, child])


def test_audit_rejects_cross_split_near_duplicates() -> None:
    train = V4Record.from_mapping(make_row("a", "la", "one two three four five six seven eight nine ten", "train"))
    dev = V4Record.from_mapping(make_row("b", "lb", "one two three four five six seven eight nine replacement", "development"))

    with pytest.raises(ValueError, match="near duplicate"):
        audit_v4_records([train, dev])
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_v4_manifest.py -v`

Expected: FAIL because lineage, parent, and shingle checks are missing.

- [ ] **Step 7: Implement atomic-lineage, resolved-parent, exact, and shingle checks**

```python
def _shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = canonical_text(text).split()
    return {tuple(words[index:index + size]) for index in range(max(1, len(words) - size + 1))}


def _require_atomic_lineages(records: Sequence[V4Record]) -> None:
    memberships: dict[str, str] = {}
    for record in records:
        prior = memberships.setdefault(record.lineage_id, record.split)
        if prior != record.split:
            raise ValueError(f"lineage crosses {prior} and {record.split}: {record.lineage_id}")
```

- [ ] **Step 8: Run focused V4 tests to verify they pass**

Run: `pytest tests/test_v4_manifest.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add ml/humanized_detector/v4_audit.py ml/tests/test_v4_manifest.py
git commit -m "feat: enforce V4 manifest integrity boundaries"
```

### Task 3: Partition writer and safe role loader

**Files:**
- Create: `ml/humanized_detector/v4_prepare.py`
- Create: `ml/tests/test_v4_prepare.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes `V4Record`, `audit_v4_records`, `metadata_manifest`, and `manifest_digest`.
- Produces `write_v4_dataset(records: Sequence[V4Record], output_dir: Path, source_metadata: Mapping[str, object]) -> dict[str, object]`.
- Produces `load_v4_partition(data_dir: Path, split: Literal["train", "development", "calibration"]) -> list[dict[str, object]]`.
- Writes `<split>.jsonl` for all roles, `metadata_manifest.json`, and `report.json`.

- [ ] **Step 1: Write the failing partition output test**

```python
import json
from humanized_detector.v4_manifest import V4Record
from humanized_detector.v4_prepare import write_v4_dataset


def test_writer_outputs_role_files_and_text_free_metadata(tmp_path: Path) -> None:
    records = [
        V4Record.from_mapping(make_row("train:1", "l1", "Train secret", "train")),
        V4Record.from_mapping(make_row("sealed:1", "l2", "Sealed secret", "sealed_test", sealed=True, train_eligible=False)),
    ]

    report = write_v4_dataset(records, tmp_path, {"source_locator": "https://example.test", "revision": "v1"})

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "sealed_test.jsonl").exists()
    assert "Train secret" not in (tmp_path / "metadata_manifest.json").read_text(encoding="utf-8")
    assert report["metadata_manifest_sha256"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v4_prepare.py::test_writer_outputs_role_files_and_text_free_metadata -v`

Expected: FAIL because `humanized_detector.v4_prepare` does not exist.

- [ ] **Step 3: Implement atomic output writing**

```python
def write_v4_dataset(records: Sequence[V4Record], output_dir: Path, source_metadata: Mapping[str, object]) -> dict[str, object]:
    audit = audit_v4_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in VALID_SPLITS:
        rows = [record.to_row() for record in records if record.split == split]
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
    metadata = metadata_manifest(records, source_metadata)
    digest = manifest_digest(metadata)
    _write_json(output_dir / "metadata_manifest.json", {**metadata, "sha256": digest})
    return {"checked_records": audit.checked_records, "metadata_manifest_sha256": digest}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_v4_prepare.py::test_writer_outputs_role_files_and_text_free_metadata -v`

Expected: PASS.

- [ ] **Step 5: Write the failing safe-loader test**

```python
import pytest
from humanized_detector.v4_prepare import load_v4_partition


def test_non_final_loader_refuses_to_load_a_sealed_partition(tmp_path: Path) -> None:
    (tmp_path / "sealed_test.jsonl").write_text('{"id":"sealed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="sealed"):
        load_v4_partition(tmp_path, "sealed_test")  # type: ignore[arg-type]
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_v4_prepare.py::test_non_final_loader_refuses_to_load_a_sealed_partition -v`

Expected: FAIL because the loader does not exist or accepts `sealed_test`.

- [ ] **Step 7: Implement non-final role loading and ignore V4 raw data**

```python
def load_v4_partition(data_dir: Path, split: str) -> list[dict[str, object]]:
    if split not in {"train", "development", "calibration"}:
        raise ValueError("non-final loaders cannot load sealed partitions")
    return [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
```

Add `v4-data/` and `v4-artifacts/` to `.gitignore` so Colab corpus/model files cannot be committed accidentally.

- [ ] **Step 8: Run V4 tests to verify they pass**

Run: `pytest tests/test_v4_manifest.py tests/test_v4_prepare.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add ml/humanized_detector/v4_prepare.py ml/tests/test_v4_prepare.py .gitignore
git commit -m "feat: write and safely load V4 manifests"
```

### Task 4: Colab-facing documentation and full regression verification

**Files:**
- Modify: `README.md`
- Create: `docs/v4-data-protocol.md`

**Interfaces:**
- Documents `write_v4_dataset` and `load_v4_partition` from Task 3.
- Documents that no V4 final benchmark source is downloaded until a public source/revision and exact selection rule are committed in a follow-up change.

- [ ] **Step 1: Write the failing documentation-presence test**

```python
from pathlib import Path


def test_v4_data_protocol_documents_sealed_manifest_rules() -> None:
    text = Path(__file__).parents[2] / "docs" / "v4-data-protocol.md"

    assert "sealed_test" in text.read_text(encoding="utf-8")
    assert "GRADTEX" in text.read_text(encoding="utf-8")
    assert "metadata_manifest.json" in text.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v4_prepare.py::test_v4_data_protocol_documents_sealed_manifest_rules -v`

Expected: FAIL because the protocol document does not exist.

- [ ] **Step 3: Write concise V4 data-protocol documentation**

Document the role definitions, schema fields, V3/GRADTEX immutability, the text-free public metadata manifest, and the exact Colab-safe usage pattern:

```python
from humanized_detector.v4_manifest import V4Record
from humanized_detector.v4_prepare import write_v4_dataset

# Build records only from train-eligible sources. Do not pass GRADTEX files.
report = write_v4_dataset(records, output_dir, source_metadata)
print(report["metadata_manifest_sha256"])
```

- [ ] **Step 4: Run documentation test to verify it passes**

Run: `pytest tests/test_v4_prepare.py::test_v4_data_protocol_documents_sealed_manifest_rules -v`

Expected: PASS.

- [ ] **Step 5: Run the full ML suite**

Run: `pytest -q`

Expected: all existing V3 tests plus the new V4 tests pass. Record any pre-existing PyTorch nested-tensor warning separately from failures.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/v4-data-protocol.md ml/tests/test_v4_prepare.py
git commit -m "docs: explain V4 sealed data workflow"
```
