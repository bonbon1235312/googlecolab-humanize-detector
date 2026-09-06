import time

import pytest

from humanized_detector.v6_manifest import (
    V6Record,
    V6Source,
    _candidate_pairs,
    audit_v6_pretrain,
    build_v6_manifest,
    canonical_text,
    write_v6_manifest,
)


def source(name: str = "beemo", *, deployment_allowed: bool = True) -> V6Source:
    return V6Source.from_mapping(
        {
            "dataset": name,
            "revision": "pinned-revision",
            "download_sha256": "a" * 64,
            "license": "research-safe",
            "deployment_allowed": deployment_allowed,
            "provenance_quality": "documented",
            "permitted_roles": ["train", "selection_dev", "calibration"],
        }
    )


def row(
    identifier: str,
    lineage_id: str,
    text: str,
    split: str,
    *,
    source_name: str = "beemo",
    origin: str = "ai",
    edit_actor: str = "human_expert",
    template_id: str = "not_applicable",
) -> dict[str, object]:
    return {
        "id": identifier,
        "lineage_id": lineage_id,
        "parent_lineage_id": None,
        "text": text,
        "dataset": source_name,
        "domain": "essay",
        "source_family": source_name,
        "template_id": template_id,
        "origin": origin,
        "edit_actor": edit_actor,
        "editor_family": edit_actor,
        "generator_family": "example-generator" if origin == "ai" else "human",
        "transformation": "expert_edit",
        "split": split,
        "sampling_weight": 1.0,
    }


def test_canonical_text_uses_the_frozen_v6_normalisation() -> None:
    assert canonical_text(" A\r\nB\t\tC ") == "a b c"


def test_record_keeps_origin_and_edit_actor_independently() -> None:
    record = V6Record.from_mapping(
        row("gen:edit", "gen:lineage", "A polished human passage.", "train", source_name="gen", origin="human", edit_actor="llm")
    )

    assert record.label == 0
    assert record.origin == "human"
    assert record.edit_actor == "llm"
    assert record.sequence_length_bucket == "short_text_diagnostic_only"


def test_audit_allows_expected_similarity_inside_a_declared_lineage() -> None:
    base = " ".join(f"word{index}" for index in range(200))
    records = [
        V6Record.from_mapping(row("a", "lineage:1", base, "train")),
        V6Record.from_mapping(row("b", "lineage:1", base.replace("word199", "replacement"), "train")),
    ]

    audit = audit_v6_pretrain(records, {"beemo": source()})

    assert audit.training_authorized is True
    assert audit.same_lineage_similarities == 1


def test_audit_hard_fails_a_cross_split_near_duplicate() -> None:
    base = " ".join(f"word{index}" for index in range(200))
    records = [
        V6Record.from_mapping(row("a", "lineage:1", base, "train")),
        V6Record.from_mapping(row("b", "lineage:2", base.replace("word199", "replacement"), "selection_dev")),
    ]

    with pytest.raises(ValueError, match="cross-split near duplicate"):
        audit_v6_pretrain(records, {"beemo": source()})


def test_audit_does_not_apply_shingle_near_dedup_to_short_diagnostic_text() -> None:
    base = "".join(chr(0x400 + index) for index in range(99))
    records = [
        V6Record.from_mapping(row("a", "lineage:1", base, "train")),
        V6Record.from_mapping(row("b", "lineage:2", base[:-1] + "Z", "selection_dev")),
    ]

    audit = audit_v6_pretrain(records, {"beemo": source()})

    assert audit.training_authorized is True


def test_minhash_candidate_discovery_is_practical_for_a_document_cohort() -> None:
    records = [
        V6Record.from_mapping(
            row(
                f"record:{index}",
                f"lineage:{index}",
                " ".join(f"document{index}_token{token}" for token in range(250)),
                "train",
            )
        )
        for index in range(100)
    ]

    started = time.monotonic()
    _candidate_pairs(records)
    assert time.monotonic() - started < 3.0


def test_audit_rejects_a_non_deployable_source_from_training() -> None:
    records = [V6Record.from_mapping(row("gen:1", "gen:1", "A documented training example.", "train", source_name="gen"))]

    with pytest.raises(ValueError, match="not deployment-approved"):
        audit_v6_pretrain(records, {"gen": source("gen", deployment_allowed=False)})


def test_audit_rejects_template_overlap_where_template_isolation_is_required() -> None:
    records = [
        V6Record.from_mapping(row("a", "lineage:1", "A distinct first document.", "train", template_id="template:1")),
        V6Record.from_mapping(row("b", "lineage:2", "A distinct second document.", "selection_dev", template_id="template:1")),
    ]

    with pytest.raises(ValueError, match="template crosses"):
        audit_v6_pretrain(records, {"beemo": source()})


def test_manifest_is_text_free_and_contains_the_frozen_pretrain_gate() -> None:
    records = [V6Record.from_mapping(row("a", "lineage:1", "A sufficiently long documented training example. " * 4, "train"))]
    manifest = build_v6_manifest(
        records,
        {"beemo": source()},
        sampling_policy={
            "class_balance": {"human": 0.5, "ai": 0.5},
            "source_weights": {"beemo": 1.0},
            "subtype_weights": {"expert_edit": 1.0},
            "length_weights": {"short": 1.0},
            "max_variants_per_lineage": 2,
            "jfleg_cap": 0,
            "humanizerbench_cap": 0,
            "mage_cap": 0,
        },
        sealed_eval_metadata={"revision": "raid-pinned", "cohort_ids": ["raid:1"], "expected_count": 1},
    )

    assert manifest["pretrain_audit"]["TRAINING AUTHORIZED"] == "YES"
    assert manifest["sealed_eval_metadata"]["expected_count"] == 1
    assert "documented training example" not in str(manifest)
    assert len(manifest["manifest_sha256"]) == 64


def test_manifest_rejects_sealed_metadata_with_text() -> None:
    record = V6Record.from_mapping(row("a", "lineage:1", "A sufficiently long documented training example. " * 4, "train"))

    with pytest.raises(ValueError, match="sealed_eval_metadata must not contain text"):
        build_v6_manifest(
            [record],
            {"beemo": source()},
            sampling_policy={
                "class_balance": {"human": 0.5, "ai": 0.5}, "source_weights": {"beemo": 1.0},
                "subtype_weights": {"expert_edit": 1.0}, "length_weights": {"short": 1.0},
                "max_variants_per_lineage": 2, "jfleg_cap": 0, "humanizerbench_cap": 0, "mage_cap": 0,
            },
            sealed_eval_metadata={"text": "RAID must never enter preparation"},
        )


def test_manifest_writer_emits_canonical_json(tmp_path) -> None:
    record = V6Record.from_mapping(row("a", "lineage:1", "A sufficiently long documented training example. " * 4, "train"))
    output = tmp_path / "v6_manifest.json"

    manifest = write_v6_manifest(
        output,
        [record],
        {"beemo": source()},
        sampling_policy={
            "class_balance": {"human": 0.5, "ai": 0.5}, "source_weights": {"beemo": 1.0},
            "subtype_weights": {"expert_edit": 1.0}, "length_weights": {"short": 1.0},
            "max_variants_per_lineage": 2, "jfleg_cap": 0, "humanizerbench_cap": 0, "mage_cap": 0,
        },
        sealed_eval_metadata={"revision": "raid-pinned", "cohort_ids": ["raid:1"], "expected_count": 1},
    )

    assert output.read_text(encoding="utf-8").endswith("\n")
    assert output.read_text(encoding="utf-8").find(manifest["manifest_sha256"]) > 0
