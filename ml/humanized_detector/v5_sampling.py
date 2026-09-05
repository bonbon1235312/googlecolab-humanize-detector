"""Hierarchical curriculum sampling for the V5 humanizer continuation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


_POSITIVE_PROVENANCES = ("expert_edited_ai", "llm_edited_ai", "raw_ai")


def curriculum_mix(epoch: int) -> dict[str, float]:
    """Return the predeclared Beemo-positive mix for a continuation epoch."""
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if epoch == 1:
        return {"expert_edited_ai": 0.25, "llm_edited_ai": 0.50, "raw_ai": 0.25}
    if epoch == 2:
        return {"expert_edited_ai": 0.40, "llm_edited_ai": 0.35, "raw_ai": 0.25}
    return {"expert_edited_ai": 0.50, "llm_edited_ai": 0.30, "raw_ai": 0.20}


def _stratum(row: Mapping[str, object]) -> str:
    source = str(row["source"])
    label = int(row["label"])
    if source == "padben":
        return f"padben:{label}"
    if source == "beemo" and label == 0:
        return "beemo:0"
    if source == "beemo" and label == 1:
        return f"beemo:1:{row['provenance']}"
    raise ValueError(f"unsupported training row source/label: {source}/{label}")


def hierarchical_weights(rows: Sequence[Mapping[str, object]], positive_mix: Mapping[str, float]) -> list[float]:
    """Balance source/label mass, Beemo positive subtype, lineage, and variants."""
    if set(positive_mix) != set(_POSITIVE_PROVENANCES):
        raise ValueError(f"positive_mix must contain exactly: {', '.join(_POSITIVE_PROVENANCES)}")
    if any(float(value) < 0 for value in positive_mix.values()) or abs(sum(float(value) for value in positive_mix.values()) - 1.0) > 1e-9:
        raise ValueError("positive_mix values must be non-negative and sum to one")

    desired_mass = {
        "padben:0": 0.25,
        "padben:1": 0.25,
        "beemo:0": 0.25,
        **{f"beemo:1:{name}": 0.25 * float(positive_mix[name]) for name in _POSITIVE_PROVENANCES},
    }
    by_stratum_lineage: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        by_stratum_lineage[_stratum(row)][str(row["lineage_id"])].append(index)
    missing = [name for name, mass in desired_mass.items() if mass > 0 and not by_stratum_lineage.get(name)]
    if missing:
        raise ValueError(f"missing training stratum: {', '.join(sorted(missing))}")

    weights = [0.0] * len(rows)
    for name, mass in desired_mass.items():
        lineages = by_stratum_lineage.get(name, {})
        if not lineages:
            continue
        lineage_mass = mass / len(lineages)
        for indexes in lineages.values():
            row_mass = lineage_mass / len(indexes)
            for index in indexes:
                weights[index] = row_mass
    return weights
