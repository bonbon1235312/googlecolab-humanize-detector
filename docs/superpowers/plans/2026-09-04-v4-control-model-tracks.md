# V4 Control Model Tracks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible, sealed-safe V4 control training tracks for the existing 5M Transformer and word/character TF-IDF logistic-regression baselines.

**Architecture:** All tracks load only `train`, `development`, and `calibration` partitions through `load_v4_partition`; `sealed_test` is structurally rejected. The Transformer track reuses the existing three-window, fusion-concat architecture with an explicit capacity preset. Classical tracks fit their vectorizer and classifier only on train data, choose no threshold on development, and persist probability predictions for later shared calibration.

**Tech Stack:** Python 3.11+, PyTorch, scikit-learn, NumPy, existing byte-level BPE/tokenizer and V4 manifest utilities.

**Spec:** `docs/superpowers/specs/2026-09-04-v4-manifest-and-sealed-benchmark-design.md`

## Global Constraints

- Read only V4 `train`, `development`, and `calibration` partitions during all control-track commands.
- Never load, inspect, vectorize, tokenize, train on, calibrate on, or evaluate `sealed_test` in these commands.
- Keep V3 artifacts and the known GRADTEX benchmark immutable.
- Fit every learned preprocessing component exclusively on the train partition.
- Treat development as model/capacity selection only; defer threshold selection to a later calibration command.
- Persist only ignored artifacts and text-free metrics/metadata outside Git.
- Implement every production behaviour by TDD: focused test, observed failure, minimal implementation, passing focused test, then full suite.

---

### Task 1: V4 non-sealed data access and capacity presets

**Files:**
- Create: `ml/humanized_detector/v4_control.py`
- Create: `ml/tests/test_v4_control.py`

**Interfaces:**
- Produces `load_v4_control_partitions(data_dir: Path) -> dict[str, list[dict[str, object]]]`.
- Produces `model_config_for_capacity(vocab_size: int, capacity: Literal["5m", "12m"]) -> ModelConfig`.

- [ ] **Step 1: Write the failing partition-loader test**

```python
def test_control_loader_reads_only_nonsealed_roles(tmp_path: Path) -> None:
    for role in ("train", "development", "calibration"):
        (tmp_path / f"{role}.jsonl").write_text('{"id":"' + role + '"}\n', encoding="utf-8")
    (tmp_path / "sealed_test.jsonl").write_text('{"id":"secret"}\n', encoding="utf-8")

    loaded = load_v4_control_partitions(tmp_path)

    assert set(loaded) == {"train", "development", "calibration"}
    assert loaded["train"] == [{"id": "train"}]
```

- [ ] **Step 2: Run the focused test and observe failure**

Run: `pytest tests/test_v4_control.py::test_control_loader_reads_only_nonsealed_roles -v`

Expected: FAIL because `v4_control` does not exist.

- [ ] **Step 3: Implement the safe loader**

```python
CONTROL_ROLES = ("train", "development", "calibration")

def load_v4_control_partitions(data_dir: Path) -> dict[str, list[dict[str, object]]]:
    return {role: load_v4_partition(data_dir, role) for role in CONTROL_ROLES}
```

- [ ] **Step 4: Run the focused test and confirm it passes**

Run: `pytest tests/test_v4_control.py::test_control_loader_reads_only_nonsealed_roles -v`

Expected: PASS.

- [ ] **Step 5: Write the failing capacity-preset test**

```python
def test_capacity_presets_keep_attention_dimensions_valid() -> None:
    small = model_config_for_capacity(4000, "5m")
    larger = model_config_for_capacity(4000, "12m")

    assert small.hidden_size == 192
    assert larger.hidden_size > small.hidden_size
    assert larger.hidden_size % larger.heads == 0
    assert larger.layers >= small.layers
```

- [ ] **Step 6: Run the focused test and observe failure**

Run: `pytest tests/test_v4_control.py::test_capacity_presets_keep_attention_dimensions_valid -v`

Expected: FAIL because `model_config_for_capacity` does not exist.

- [ ] **Step 7: Implement two explicit presets**

```python
def model_config_for_capacity(vocab_size: int, capacity: str) -> ModelConfig:
    if capacity == "5m":
        return ModelConfig(vocab_size=vocab_size, hidden_size=192, heads=6, layers=4)
    if capacity == "12m":
        return ModelConfig(vocab_size=vocab_size, hidden_size=384, heads=8, layers=6)
    raise ValueError("capacity must be '5m' or '12m'")
```

- [ ] **Step 8: Run Task 1 tests and commit**

Run: `pytest tests/test_v4_control.py -v`

Commit:
```bash
git add ml/humanized_detector/v4_control.py ml/tests/test_v4_control.py
git commit -m "feat: add V4 control partition and capacity helpers"
```

### Task 2: Classical word and character control baselines

**Files:**
- Create: `ml/humanized_detector/v4_baselines.py`
- Create: `ml/tests/test_v4_baselines.py`

**Interfaces:**
- Produces `train_tfidf_baseline(train_rows, development_rows, artifact_dir, variant) -> dict[str, object]` where `variant` is exactly `word_tfidf_lr` or `char_tfidf_lr`.
- Produces `run_tfidf_control(data_dir, artifact_dir, variant) -> dict[str, object]` and CLI `python -m humanized_detector.v4_baselines --data-dir ... --artifacts-dir ... --variant word_tfidf_lr`.
- Writes `model.joblib`, `development_metrics.json`, and `development_predictions.jsonl` to the supplied artifact directory.

- [ ] **Step 1: Write the failing word-baseline test**

```python
def test_word_baseline_fits_only_train_and_writes_development_predictions(tmp_path: Path) -> None:
    train = [{"id": "h", "text": "ordinary human prose", "label": 0}, {"id": "a", "text": "machine generated patterned prose", "label": 1}]
    development = [{"id": "d", "text": "ordinary prose", "label": 0}, {"id": "e", "text": "generated patterned prose", "label": 1}]

    result = train_tfidf_baseline(train, development, tmp_path, "word_tfidf_lr")

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "development_predictions.jsonl").exists()
    assert result["variant"] == "word_tfidf_lr"
```

- [ ] **Step 2: Run the focused test and observe failure**

Run: `pytest tests/test_v4_baselines.py::test_word_baseline_fits_only_train_and_writes_development_predictions -v`

Expected: FAIL because `v4_baselines` does not exist.

- [ ] **Step 3: Implement a train-only `TfidfVectorizer` and `LogisticRegression` pipeline**

```python
def _vectorizer(variant: str) -> TfidfVectorizer:
    if variant == "word_tfidf_lr":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    if variant == "char_tfidf_lr":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    raise ValueError("unknown V4 baseline variant")
```

Fit the vectorizer and classifier on `train_rows` only. Use `evaluate_binary` for development ranking metrics and write one non-text row per prediction containing `id`, `label`, and `probability`.

- [ ] **Step 4: Run the focused word test and confirm it passes**

Run: `pytest tests/test_v4_baselines.py::test_word_baseline_fits_only_train_and_writes_development_predictions -v`

Expected: PASS.

- [ ] **Step 5: Write the failing character-baseline test**

```python
def test_character_baseline_is_a_distinct_saved_variant(tmp_path: Path) -> None:
    rows = [{"id": "h", "text": "plain text", "label": 0}, {"id": "a", "text": "stylised---text", "label": 1}]

    result = train_tfidf_baseline(rows, rows, tmp_path, "char_tfidf_lr")

    assert result["variant"] == "char_tfidf_lr"
    assert result["vectorizer_analyzer"] == "char_wb"
```

- [ ] **Step 6: Run the focused test, implement its minimum metadata, then rerun**

Run: `pytest tests/test_v4_baselines.py::test_character_baseline_is_a_distinct_saved_variant -v`

Expected first run: FAIL because the result metadata is absent.

Add `vectorizer_analyzer` to the returned and saved metrics payload, then rerun the same command.

Expected second run: PASS.

- [ ] **Step 7: Run Task 2 tests and commit**

Run: `pytest tests/test_v4_baselines.py -v`

Commit:
```bash
git add ml/humanized_detector/v4_baselines.py ml/tests/test_v4_baselines.py
git commit -m "feat: add V4 TF-IDF logistic baselines"
```

### Task 3: Transformer control runner and development-only checkpointing

**Files:**
- Create: `ml/humanized_detector/v4_train.py`
- Modify: `ml/tests/test_v4_control.py`

**Interfaces:**
- Produces `train_v4_transformer(data_dir, artifact_dir, capacity, epochs, batch_size, learning_rate, weight_decay) -> dict[str, object]`.
- CLI: `python -m humanized_detector.v4_train --data-dir /content/drive/MyDrive/v4-control-data --artifacts-dir /content/drive/MyDrive/v4-artifacts/fusion-concat-5m --capacity 5m`.
- Reuses the `fusion_concat` architecture and V3’s source-label weighting, structural features, BPE and ranking checkpoint selection.

- [ ] **Step 1: Write the failing runner-boundary test**

```python
def test_transformer_runner_uses_only_control_partitions(monkeypatch, tmp_path: Path) -> None:
    requested: list[str] = []

    def fake_load(directory: Path, role: str) -> list[dict[str, object]]:
        requested.append(role)
        return [{"id": role, "text": "text", "label": 0, "source": "test"}]

    monkeypatch.setattr("humanized_detector.v4_control.load_v4_partition", fake_load)
    assert set(load_v4_control_partitions(tmp_path)) == {"train", "development", "calibration"}
    assert requested == ["train", "development", "calibration"]
```

- [ ] **Step 2: Run the focused test and confirm its intended behaviour before runner code**

Run: `pytest tests/test_v4_control.py::test_transformer_runner_uses_only_control_partitions -v`

Expected: PASS after Task 1; this is the regression boundary the runner must use.

- [ ] **Step 3: Write the failing runner configuration test**

```python
def test_v4_train_rejects_unknown_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        train_v4_transformer(Path("data"), Path("artifacts"), "70m", 1, 2, 1e-4, 0.01)
```

- [ ] **Step 4: Run the focused test and observe failure**

Run: `pytest tests/test_v4_control.py::test_v4_train_rejects_unknown_capacity -v`

Expected: FAIL because `v4_train` does not exist.

- [ ] **Step 5: Implement the runner by adapting `train_v3_model`, not by duplicating data rules**

```python
def train_v4_transformer(
    data_dir: Path,
    artifact_dir: Path,
    capacity: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, object]:
    partitions = load_v4_control_partitions(data_dir)
    if capacity not in {"5m", "12m"}:
        raise ValueError("capacity must be '5m' or '12m'")
    # write only temporary non-sealed partition files inside artifact_dir
    # train BPE/features on train only
    # run FusedMultiWindowClassifier(config, len(FEATURE_NAMES), pooling="attention", gated=False)
    # choose checkpoint using development ROC-AUC / PR-AUC / lower human FPR tie-break
    # save development metrics/predictions and calibration predictions; do not calculate a threshold
```

Use the `5m` preset first. Do not add scheduler experiments, warm-up, V4 hard data, ONNX export, or sealed evaluation in this task.

- [ ] **Step 6: Rerun the focused test, then run control tests**

Run: `pytest tests/test_v4_control.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add ml/humanized_detector/v4_train.py ml/tests/test_v4_control.py
git commit -m "feat: add sealed-safe V4 transformer control runner"
```

### Task 4: Colab execution guide and full verification

**Files:**
- Modify: `README.md`
- Create: `docs/v4-control-runs.md`
- Modify: `ml/tests/test_v4_control.py`

**Interfaces:**
- Documents exact 5M baseline commands and a later 12M capacity-ablation command.
- Explicitly states that calibration and the RAID seal are excluded from these commands.

- [ ] **Step 1: Write the failing documentation assertion**

```python
def test_control_run_guide_mentions_5m_and_forbids_sealed_evaluation() -> None:
    text = (Path(__file__).parents[2] / "docs" / "v4-control-runs.md").read_text(encoding="utf-8")
    assert "--capacity 5m" in text
    assert "sealed_test" in text
    assert "Do not" in text
```

- [ ] **Step 2: Run the focused test and observe failure**

Run: `pytest tests/test_v4_control.py::test_control_run_guide_mentions_5m_and_forbids_sealed_evaluation -v`

Expected: FAIL because the guide does not exist.

- [ ] **Step 3: Document the Colab control run**

Include repository install, Drive paths, baseline commands, transformer command, expected artifact locations, and the explicit rule: no calibration, GRADTEX, or RAID seal evaluation during control model selection.

- [ ] **Step 4: Run documentation test and full ML suite**

Run:
```bash
pytest tests/test_v4_control.py::test_control_run_guide_mentions_5m_and_forbids_sealed_evaluation -v
pytest -q
```

Expected: all tests pass; separately record any known PyTorch nested-tensor warning.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/v4-control-runs.md ml/tests/test_v4_control.py
git commit -m "docs: add V4 control-run instructions"
```

## Self-review

- The plan covers all four selected V4 control tracks except the explicitly deferred pretrained encoder: word LR, character LR, and the 5M/12M capacity pair using the custom Transformer.
- It preserves the fixed data roles: all commands structurally load only train/development/calibration.
- It intentionally defers thresholding, V4 hard-data augmentation, sealed RAID loading, GRADTEX use, ONNX export, and deployment until the model family/control decision is frozen.
- The plan contains no placeholder tasks and uses the same function names and capacity values in every task.
