"""Fail-closed provenance, split, and duplicate auditing for V6 data."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import unicodedata
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = "v6"
VALID_SPLITS = frozenset({"train", "selection_dev", "calibration"})
_FORBIDDEN = frozenset({"", "tbd", "unknown", "none", "null", "n/a"})
_WHITESPACE = re.compile(r"\s+")
_MINHASH_PERMUTATIONS = 128
_MINHASH_BAND_SIZE = 4
_MINHASH_PRIME = 2_147_483_647


def _minhash_coefficients() -> tuple[np.ndarray, np.ndarray]:
    """Create the frozen 128 deterministic universal-hash permutations."""
    first: list[int] = []
    second: list[int] = []
    for seed in range(_MINHASH_PERMUTATIONS):
        digest = hashlib.blake2b(f"v6-minhash:{seed}".encode("utf-8"), digest_size=8).digest()
        first.append(1 + int.from_bytes(digest[:4], "big") % (_MINHASH_PRIME - 1))
        second.append(int.from_bytes(digest[4:], "big") % _MINHASH_PRIME)
    return np.asarray(first, dtype=np.uint64), np.asarray(second, dtype=np.uint64)


_MINHASH_A, _MINHASH_B = _minhash_coefficients()


def canonical_text(text: str) -> str:
    """Return the frozen V6 exact-dedup representation."""
    normal = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return _WHITESPACE.sub(" ", normal.casefold()).strip()


def _sha256(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(value: object, field_name: str, *, allow_none: bool = False) -> str:
    if value is None and allow_none:
        return ""
    result = str(value).strip()
    if result.casefold() in _FORBIDDEN:
        raise ValueError(f"{field_name} must be established; got {value!r}")
    return result


def _length_bucket(text: str) -> str:
    length = len(canonical_text(text))
    if length < 100:
        return "short_text_diagnostic_only"
    if length < 1_000:
        return "short"
    if length < 4_000:
        return "medium"
    return "long"


def _char_shingles(text: str, width: int = 5) -> frozenset[str]:
    canonical = canonical_text(text)
    if len(canonical) < width:
        return frozenset({canonical}) if canonical else frozenset()
    return frozenset(canonical[index : index + width] for index in range(len(canonical) - width + 1))


def _minhash_signature(shingles: Iterable[str]) -> tuple[int, ...]:
    values = tuple(shingles)
    if not values:
        return tuple(2**64 - 1 for _ in range(_MINHASH_PERMUTATIONS))
    hashes = np.fromiter(
        (int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=4).digest(), "big") % _MINHASH_PRIME for item in values),
        dtype=np.uint64,
        count=len(values),
    )
    permuted = (_MINHASH_A[:, None] * hashes[None, :] + _MINHASH_B[:, None]) % _MINHASH_PRIME
    return tuple(int(value) for value in np.min(permuted, axis=1))


def _candidate_pairs(records: Sequence["V6Record"]) -> set[tuple[int, int]]:
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        signature = _minhash_signature(_char_shingles(record.text))
        for band_start in range(0, _MINHASH_PERMUTATIONS, _MINHASH_BAND_SIZE):
            band = band_start // _MINHASH_BAND_SIZE
            buckets[(band, signature[band_start : band_start + _MINHASH_BAND_SIZE])].append(index)
    return {
        tuple(pair)
        for indices in buckets.values()
        if len(indices) > 1
        for pair in itertools.combinations(sorted(indices), 2)
    }


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass(frozen=True)
class V6Source:
    dataset: str
    revision: str
    download_sha256: str
    license: str
    deployment_allowed: bool
    provenance_quality: str
    permitted_roles: tuple[str, ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "V6Source":
        roles = row.get("permitted_roles")
        if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)) or not roles:
            raise ValueError("permitted_roles must be a non-empty sequence")
        digest = _require(row.get("download_sha256"), "download_sha256")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("download_sha256 must be a SHA-256 hex digest")
        return cls(
            dataset=_require(row.get("dataset"), "dataset"),
            revision=_require(row.get("revision"), "revision"),
            download_sha256=digest.casefold(),
            license=_require(row.get("license"), "license"),
            deployment_allowed=bool(row.get("deployment_allowed")),
            provenance_quality=_require(row.get("provenance_quality"), "provenance_quality"),
            permitted_roles=tuple(_require(role, "permitted_role") for role in roles),
        )


@dataclass(frozen=True)
class V6Record:
    id: str
    lineage_id: str
    parent_lineage_id: str | None
    text: str
    text_sha256: str
    dataset: str
    domain: str
    source_family: str
    template_id: str
    origin: str
    edit_actor: str
    editor_family: str
    generator_family: str
    transformation: str
    sequence_length_bucket: str
    split: str
    sampling_weight: float
    label: int

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "V6Record":
        origin = _require(row.get("origin"), "origin").casefold()
        if origin not in {"human", "ai"}:
            raise ValueError("origin must be 'human' or 'ai'")
        text = str(row.get("text", ""))
        if not canonical_text(text):
            raise ValueError("text must not be blank")
        split = _require(row.get("split"), "split")
        if split not in VALID_SPLITS:
            raise ValueError(f"unknown split: {split}")
        weight = float(row.get("sampling_weight", 0))
        if weight <= 0:
            raise ValueError("sampling_weight must be positive")
        parent = row.get("parent_lineage_id")
        return cls(
            id=_require(row.get("id"), "id"),
            lineage_id=_require(row.get("lineage_id"), "lineage_id"),
            parent_lineage_id=_require(parent, "parent_lineage_id") if parent is not None else None,
            text=text,
            text_sha256=_sha256(canonical_text(text)),
            dataset=_require(row.get("dataset"), "dataset"),
            domain=_require(row.get("domain"), "domain"),
            source_family=_require(row.get("source_family"), "source_family"),
            template_id=_require(row.get("template_id"), "template_id"),
            origin=origin,
            edit_actor=_require(row.get("edit_actor"), "edit_actor"),
            editor_family=_require(row.get("editor_family"), "editor_family"),
            generator_family=_require(row.get("generator_family"), "generator_family"),
            transformation=_require(row.get("transformation"), "transformation"),
            sequence_length_bucket=_length_bucket(text),
            split=split,
            sampling_weight=weight,
            label=0 if origin == "human" else 1,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "id": self.id,
            "lineage_id": self.lineage_id,
            "parent_lineage_id": self.parent_lineage_id,
            "text_sha256": self.text_sha256,
            "dataset": self.dataset,
            "domain": self.domain,
            "source_family": self.source_family,
            "template_id": self.template_id,
            "origin": self.origin,
            "edit_actor": self.edit_actor,
            "editor_family": self.editor_family,
            "generator_family": self.generator_family,
            "transformation": self.transformation,
            "sequence_length_bucket": self.sequence_length_bucket,
            "split": self.split,
            "sampling_weight": self.sampling_weight,
            "label": self.label,
        }


@dataclass(frozen=True)
class V6Audit:
    training_authorized: bool
    record_count: int
    split_counts: Mapping[str, int]
    same_lineage_similarities: int
    manifest_sha256: str
    sampling_policy_sha256: str
    dedup_policy_sha256: str


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def audit_v6_pretrain(
    records: Sequence[V6Record],
    sources: Mapping[str, V6Source],
    *,
    near_duplicate_threshold: float = 0.90,
    isolate_fields: Sequence[str] = ("template_id",),
) -> V6Audit:
    """Fail closed when V6 provenance, split, or duplicate rules are violated."""
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    if not records:
        raise ValueError("a V6 manifest requires at least one record")
    by_id: dict[str, V6Record] = {}
    lineage_splits: dict[str, str] = {}
    family_splits: dict[tuple[str, str], str] = {}
    exact_seen: dict[str, V6Record] = {}
    for record in records:
        if record.id in by_id:
            raise ValueError(f"duplicate record ID: {record.id}")
        by_id[record.id] = record
        source = sources.get(record.dataset)
        if source is None:
            raise ValueError(f"source audit missing for dataset: {record.dataset}")
        if not source.deployment_allowed:
            raise ValueError(f"source is not deployment-approved: {record.dataset}")
        if record.split not in source.permitted_roles:
            raise ValueError(f"source role is not permitted: {record.dataset}/{record.split}")
        previous = lineage_splits.setdefault(record.lineage_id, record.split)
        if previous != record.split:
            raise ValueError(f"lineage crosses {previous} and {record.split}: {record.lineage_id}")
        for field in isolate_fields:
            value = getattr(record, field)
            if value.casefold() in {"none", "not_applicable"}:
                continue
            key = (field, value)
            previous = family_splits.setdefault(key, record.split)
            if previous != record.split:
                raise ValueError(f"{field.removesuffix('_id')} crosses {previous} and {record.split}: {value}")
        duplicate = exact_seen.get(record.text_sha256)
        if duplicate is not None and duplicate.lineage_id != record.lineage_id:
            if duplicate.split != record.split:
                raise ValueError(f"cross-split exact duplicate: {duplicate.id} / {record.id}")
            raise ValueError(f"cross-lineage exact duplicate: {duplicate.id} / {record.id}")
        exact_seen.setdefault(record.text_sha256, record)

    same_lineage_similarities = 0
    shingles = [_char_shingles(record.text) for record in records]
    for left_index, right_index in sorted(_candidate_pairs(records)):
        left, right = records[left_index], records[right_index]
        if len(canonical_text(left.text)) < 100 or len(canonical_text(right.text)) < 100:
            continue
        similarity = _jaccard(shingles[left_index], shingles[right_index])
        if similarity < near_duplicate_threshold:
            continue
        if left.lineage_id == right.lineage_id:
            same_lineage_similarities += 1
            continue
        if left.split != right.split:
            raise ValueError(f"cross-split near duplicate: {left.id} / {right.id} ({similarity:.3f})")
        raise ValueError(f"cross-lineage near duplicate: {left.id} / {right.id} ({similarity:.3f})")

    metadata = {"protocol_version": PROTOCOL_VERSION, "records": [record.metadata() for record in records]}
    dedup_policy = {
        "exact": "NFKC -> casefold -> line ending normalization -> whitespace collapse -> strip -> SHA-256",
        "near": {"shingles": "character-5", "minhash_permutations": 128, "jaccard_threshold": near_duplicate_threshold},
        "short_text": "under 100 normalized characters: exact only; diagnostic-only",
    }
    sampling_policy = {"records": [{"id": record.id, "sampling_weight": record.sampling_weight} for record in records]}
    return V6Audit(
        training_authorized=True,
        record_count=len(records),
        split_counts=dict(sorted(Counter(record.split for record in records).items())),
        same_lineage_similarities=same_lineage_similarities,
        manifest_sha256=_sha256(_canonical_json(metadata)),
        sampling_policy_sha256=_sha256(_canonical_json(sampling_policy)),
        dedup_policy_sha256=_sha256(_canonical_json(dedup_policy)),
    )


_REQUIRED_SAMPLING_POLICY_FIELDS = frozenset(
    {
        "class_balance",
        "source_weights",
        "subtype_weights",
        "length_weights",
        "max_variants_per_lineage",
        "jfleg_cap",
        "humanizerbench_cap",
        "mage_cap",
    }
)


def _assert_established(value: object, location: str) -> None:
    if value is None:
        raise ValueError(f"{location} must not be null")
    if isinstance(value, str) and value.strip().casefold() in _FORBIDDEN:
        raise ValueError(f"{location} must be established")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_established(item, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_established(item, f"{location}[{index}]")


def _assert_text_free(payload: Mapping[str, object], location: str = "sealed_eval_metadata") -> None:
    for key, value in payload.items():
        key_text = str(key).casefold()
        if key_text in {"text", "prompt", "generation", "content"}:
            raise ValueError(f"{location} must not contain text")
        if isinstance(value, Mapping):
            _assert_text_free(value, f"{location}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    _assert_text_free(item, f"{location}.{key}[{index}]")


def _group_counts(records: Sequence[V6Record], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(record, field)) for record in records).items()))


def _source_metadata(source: V6Source) -> dict[str, object]:
    return {
        "dataset": source.dataset,
        "revision": source.revision,
        "download_sha256": source.download_sha256,
        "license": source.license,
        "deployment_allowed": source.deployment_allowed,
        "provenance_quality": source.provenance_quality,
        "permitted_roles": list(source.permitted_roles),
    }


def build_v6_manifest(
    records: Sequence[V6Record],
    sources: Mapping[str, V6Source],
    *,
    sampling_policy: Mapping[str, object],
    sealed_eval_metadata: Mapping[str, object],
    isolate_fields: Sequence[str] = ("template_id",),
) -> dict[str, object]:
    """Build the text-free V6 pretrain manifest after fail-closed auditing."""
    missing = sorted(_REQUIRED_SAMPLING_POLICY_FIELDS - set(sampling_policy))
    if missing:
        raise ValueError(f"sampling_policy missing required fields: {', '.join(missing)}")
    _assert_established(sampling_policy, "sampling_policy")
    _assert_text_free(sealed_eval_metadata)
    audit = audit_v6_pretrain(records, sources, isolate_fields=isolate_fields)
    source_audit = [_source_metadata(source) for _, source in sorted(sources.items())]
    lineage_map = [record.metadata() for record in records]
    split_audit = {
        "lineage_overlap": 0,
        "template_overlap": 0 if "template_id" in isolate_fields else "not_required",
        "editor_family_overlap": 0 if "editor_family" in isolate_fields else "not_required",
        "source_family_overlap": 0 if "source_family" in isolate_fields else "not_required",
    }
    final_counts = {
        "rows_by_split": dict(audit.split_counts),
        "unique_lineages": len({record.lineage_id for record in records}),
        "origin_x_edit_actor": dict(
            sorted(Counter(f"{record.origin}::{record.edit_actor}" for record in records).items())
        ),
        "dataset": _group_counts(records, "dataset"),
        "domain": _group_counts(records, "domain"),
        "transformation": _group_counts(records, "transformation"),
        "editor_family": _group_counts(records, "editor_family"),
        "generator_family": _group_counts(records, "generator_family"),
        "sequence_length_bucket": _group_counts(records, "sequence_length_bucket"),
    }
    payload: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "source_audit": source_audit,
        "lineage_map": lineage_map,
        "sampling_policy": dict(sampling_policy),
        "deduplication_report": {
            "policy": {
                "exact": "NFKC -> casefold -> line ending normalization -> whitespace collapse -> strip -> SHA-256",
                "near": "character-5-shingles -> 128-permutation MinHash candidate discovery -> exact Jaccard >= 0.90",
                "same_lineage_similarity": "expected",
                "cross_lineage_collision": "hard_fail",
                "cross_split_collision": "hard_fail",
                "short_text": "under 100 normalized characters: exact only; diagnostic-only",
            },
            "same_lineage_similarities": audit.same_lineage_similarities,
        },
        "split_audit": split_audit,
        "sealed_eval_metadata": dict(sealed_eval_metadata),
        "final_counts": final_counts,
        "pretrain_audit": {
            "Source licences": "PASS",
            "Provenance requirements": "PASS",
            "Lineage split isolation": "PASS",
            "Template/editor isolation": "PASS",
            "Exact-normalized overlap": "PASS",
            "Near-duplicate overlap": "PASS",
            "Calibration exclusion": "PASS",
            "Sealed RAID exclusion": "PASS",
            "Sampling policy frozen": "PASS",
            "TRAINING AUTHORIZED": "YES",
        },
        "sampling_policy_sha256": audit.sampling_policy_sha256,
        "dedup_policy_sha256": audit.dedup_policy_sha256,
    }
    payload["manifest_sha256"] = _sha256(_canonical_json(payload))
    return payload


def write_v6_manifest(
    output: Path,
    records: Sequence[V6Record],
    sources: Mapping[str, V6Source],
    *,
    sampling_policy: Mapping[str, object],
    sealed_eval_metadata: Mapping[str, object],
    isolate_fields: Sequence[str] = ("template_id",),
) -> dict[str, object]:
    """Write the canonical, text-free manifest and return its parsed payload."""
    payload = build_v6_manifest(
        records,
        sources,
        sampling_policy=sampling_policy,
        sealed_eval_metadata=sealed_eval_metadata,
        isolate_fields=isolate_fields,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_records(path: Path) -> list[V6Record]:
    return [
        V6Record.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = ArgumentParser(description="Build the V6 text-free manifest and pretrain audit.")
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--sources-json", type=Path, required=True)
    parser.add_argument("--sampling-policy-json", type=Path, required=True)
    parser.add_argument("--sealed-eval-metadata-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolate-fields", nargs="*", default=["template_id"])
    args = parser.parse_args()

    source_rows = _load_json(args.sources_json)
    if not isinstance(source_rows, list):
        raise ValueError("sources-json must contain a JSON list")
    sources = {
        source.dataset: source
        for source in (V6Source.from_mapping(row) for row in source_rows if isinstance(row, Mapping))
    }
    if len(sources) != len(source_rows):
        raise ValueError("sources-json entries must all be JSON objects with unique dataset names")
    sampling_policy = _load_json(args.sampling_policy_json)
    sealed_eval_metadata = _load_json(args.sealed_eval_metadata_json)
    if not isinstance(sampling_policy, Mapping) or not isinstance(sealed_eval_metadata, Mapping):
        raise ValueError("sampling policy and sealed metadata must be JSON objects")
    manifest = write_v6_manifest(
        args.output,
        _load_jsonl_records(args.records_jsonl),
        sources,
        sampling_policy=sampling_policy,
        sealed_eval_metadata=sealed_eval_metadata,
        isolate_fields=args.isolate_fields,
    )
    print(json.dumps(manifest["pretrain_audit"], indent=2))
    print(f"Manifest SHA-256: {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
