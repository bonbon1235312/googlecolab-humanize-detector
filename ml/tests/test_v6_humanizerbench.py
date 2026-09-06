from humanized_detector.v6_humanizerbench import adapt_humanizerbench_cycle


def test_humanizerbench_adapter_keeps_ai_origin_through_humanizer_rewrite() -> None:
    samples = [{
        "id": "sample-1",
        "prompt_slug": "academic_essay",
        "category": "academic_essay",
        "source_model_slug": "gpt-5-5",
        "input_text": "The original machine draft.",
    }]
    tests = [{
        "id": "test-1",
        "sample_id": "sample-1",
        "humanizer_slug": "rewrite-tool",
        "input_text": "The original machine draft.",
        "output_text": "The humanized machine draft.",
    }]

    candidates = adapt_humanizerbench_cycle(samples, tests, cycle="August 2026")

    assert [(candidate.origin, candidate.edit_actor, candidate.label) for candidate in candidates] == [
        ("ai", "not_applicable", 1),
        ("ai", "llm", 1),
    ]
    assert {candidate.lineage_id for candidate in candidates} == {"humanizerbench:August 2026:sample-1"}
    assert candidates[0].generator_family == "gpt-5-5"
    assert candidates[1].editor_family == "rewrite-tool"
    assert candidates[1].template_id == "academic_essay"


def test_humanizerbench_adapter_rejects_a_test_without_its_declared_source_sample() -> None:
    try:
        adapt_humanizerbench_cycle([], [{
            "id": "test-1", "sample_id": "missing", "humanizer_slug": "rewrite-tool", "output_text": "text"
        }], cycle="August 2026")
    except ValueError as error:
        assert "missing source sample" in str(error)
    else:
        raise AssertionError("expected missing sample provenance to fail closed")
