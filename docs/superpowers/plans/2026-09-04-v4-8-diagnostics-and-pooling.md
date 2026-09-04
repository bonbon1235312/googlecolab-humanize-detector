# V4.8 Diagnostics and Pooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add V4.8 diagnostics, a masked-mean-pooling Transformer ablation, PADBen in-source reporting, and group-cross-fitted Platt calibration without reading sealed data.

**Architecture:** V4.8 extends the existing V4 non-sealed partition loaders. `v4_diagnostics.py` scores saved checkpoints without affecting selection. `ModelConfig.token_pooling` preserves legacy first-token behaviour by default while supporting a mean-pooling ablation. Calibration joins non-text predictions to calibration metadata, cross-fits by Beemo lineage, and saves a final post-selection Platt mapper.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, scikit-learn, existing V4 manifest tooling.

**Spec:** `docs/superpowers/specs/2026-09-04-v4-8-diagnostics-and-pooling-design.md`

## Global Constraints

- V4.0 artifacts are never overwritten; V4.8 uses a new artifact directory.
- Non-final V4 code must never load `sealed_test` or RAID raw text.
- Training diagnostics do not select checkpoints or thresholds.
- Development selects a model; calibration fits thresholds only after selection.
- Every production behaviour begins with an observed failing pytest assertion.

---

### Task 1: Checkpoint fit and subtype diagnostics

**Files:**
- Create: `ml/humanized_detector/v4_diagnostics.py`
- Create: `ml/tests/test_v4_diagnostics.py`

**Interfaces:**
- `bootstrap_roc_auc(labels, probabilities, iterations=1000, seed=20260904) -> dict[str, float]`
- `development_subtype_metrics(rows, probabilities) -> dict[str, dict[str, object]]`
- `write_v4_fit_diagnostics(data_dir, artifact_dir) -> dict[str, object]`

- [ ] **Step 1: Write a failing bootstrap test**

```python
def test_bootstrap_auc_is_deterministic_and_contains_the_observed_auc() -> None:
    result = bootstrap_roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], iterations=100, seed=7)
    assert result["lower"] <= result["point"] <= result["upper"]
    assert result == bootstrap_roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], iterations=100, seed=7)
```

- [ ] **Step 2: Run it and observe failure**

Run: `pytest tests/test_v4_diagnostics.py::test_bootstrap_auc_is_deterministic_and_contains_the_observed_auc -v`

Expected: FAIL because `v4_diagnostics` does not exist.

- [ ] **Step 3: Implement stratified bootstrap AUC**

Resample human and AI indexes separately with replacement, calculate `roc_auc_score`, and return 2.5th/50th/97.5th percentiles plus the observed point estimate.

- [ ] **Step 4: Add a failing subtype test and implement human-versus-subtype metrics**

```python
def test_subtype_metrics_compare_each_positive_provenance_with_humans() -> None:
    rows = [
        {"label": 0, "provenance": "human"}, {"label": 1, "provenance": "raw_ai"},
        {"label": 1, "provenance": "expert_edited_ai"},
    ]
    report = development_subtype_metrics(rows, [0.1, 0.9, 0.8])
    assert set(report) == {"raw_ai", "expert_edited_ai"}
```

Run: `pytest tests/test_v4_diagnostics.py -v`

Expected first run: FAIL because subtype metrics are absent. Implement by pairing all human rows with each one positive provenance and calling `evaluate_binary`.

- [ ] **Step 5: Implement checkpoint scoring and report writing**

Load only `train` and `development` through `load_v4_control_partitions`; reconstruct the saved V4 fusion model, produce predictions, and write `fit_diagnostics.json` containing aggregate metrics, AUC gap, subtype metrics, CIs, and negative-count resolution. Do not load calibration or sealed data.

- [ ] **Step 6: Run diagnostics tests and commit**

Run: `pytest tests/test_v4_diagnostics.py -v`

Commit:
```bash
git add ml/humanized_detector/v4_diagnostics.py ml/tests/test_v4_diagnostics.py
git commit -m "feat: add V4.8 checkpoint fit diagnostics"
```

### Task 2: Configurable masked mean pooling

**Files:**
- Modify: `ml/humanized_detector/model.py`
- Modify: `ml/humanized_detector/v4_train.py`
- Modify: `ml/tests/test_model.py` or create `ml/tests/test_v4_pooling.py`

**Interfaces:**
- `ModelConfig(vocab_size: int, hidden_size: int = 192, heads: int = 6, layers: int = 4, max_tokens: int = 256, dropout: float = 0.15, token_pooling: Literal["first", "mean"] = "first")`
- `TinyTransformerClassifier.encode(input_ids) -> Tensor` honours `token_pooling`.
- `v4_train` accepts `--token-pooling {first,mean}`.

- [ ] **Step 1: Write the failing padding-invariance test**

```python
def test_masked_mean_pooling_ignores_appended_padding() -> None:
    model = TinyTransformerClassifier(ModelConfig(vocab_size=32, hidden_size=8, heads=2, layers=1, max_tokens=8, token_pooling="mean"))
    model.eval()
    assert torch.allclose(model.encode(torch.tensor([[1, 2, 3, 0]])), model.encode(torch.tensor([[1, 2, 3, 0, 0, 0]])), atol=1e-6)
```

- [ ] **Step 2: Run it and observe failure**

Run: `pytest tests/test_v4_pooling.py::test_masked_mean_pooling_ignores_appended_padding -v`

Expected: FAIL because `ModelConfig` has no `token_pooling`.

- [ ] **Step 3: Implement pooling without changing the legacy default**

After encoder masking, return `hidden[:, 0]` for `first`; for `mean`, compute the sum over `~padding` token states divided by the non-padding count clamped to one. Reject any other value in `ModelConfig` construction.

- [ ] **Step 4: Add a failing V4 CLI propagation test and implement it**

Assert `train_v4_transformer(data_dir, artifact_dir, "5m", 1, 2, 1e-4, 0.01, token_pooling="mean")` saves checkpoint configuration with `token_pooling == "mean"`. Add a `--token-pooling` parser argument and use `dataclasses.replace` on the capacity preset.

- [ ] **Step 5: Run pooling tests and commit**

Run: `pytest tests/test_v4_pooling.py tests/test_v4_control.py -v`

Commit:
```bash
git add ml/humanized_detector/model.py ml/humanized_detector/v4_train.py ml/tests/test_v4_pooling.py ml/tests/test_v4_control.py
git commit -m "feat: add V4.8 masked mean pooling ablation"
```

### Task 3: Internal PADBen diagnostic set

**Files:**
- Create: `ml/humanized_detector/v4_padben_diagnostic.py`
- Create: `ml/tests/test_v4_padben_diagnostic.py`

**Interfaces:**
- `build_padben_diagnostic(padben_rows, control_data_dir) -> list[V4Record]`
- The result includes only PADBen records whose `padben:<idx>` ID is absent from the control metadata manifest.

- [ ] **Step 1: Write the failing exclusion test**

```python
def test_padben_diagnostic_excludes_all_control_manifest_ids(tmp_path: Path) -> None:
    (tmp_path / "metadata_manifest.json").write_text(json.dumps({"records": [{"id": "padben:1"}]}), encoding="utf-8")
    records = build_padben_diagnostic([
        {"idx": 1, "sentence": "seen", "label": 0}, {"idx": 2, "sentence": "unseen", "label": 1},
    ], tmp_path)
    assert [record.id for record in records] == ["padben:2"]
```

- [ ] **Step 2: Run it and observe failure**

Run: `pytest tests/test_v4_padben_diagnostic.py::test_padben_diagnostic_excludes_all_control_manifest_ids -v`

Expected: FAIL because `v4_padben_diagnostic` does not exist.

- [ ] **Step 3: Implement diagnostic records and a CLI writer**

Create V4-shaped, non-sealed diagnostic records with `split="padben_diagnostic"` only in its standalone output JSONL. Preserve label/provenance and never include them in the V4 control loader.

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/test_v4_padben_diagnostic.py -v`

Commit:
```bash
git add ml/humanized_detector/v4_padben_diagnostic.py ml/tests/test_v4_padben_diagnostic.py
git commit -m "feat: add unused PADBen diagnostic builder"
```

### Task 4: Group-cross-fitted Platt report

**Files:**
- Modify: `ml/humanized_detector/v4_calibrate.py`
- Modify: `ml/tests/test_v4_calibrate.py`

**Interfaces:**
- `crossfit_platt(rows, probabilities, folds=5) -> np.ndarray`
- Calibration report adds raw and cross-fitted Platt Brier/ECE values, plus final all-row Platt coefficients for deployment after model selection.

- [ ] **Step 1: Write the failing group-isolation test**

```python
def test_crossfit_platt_holds_out_complete_lineages() -> None:
    rows = [{"lineage_id": "a", "label": 0}, {"lineage_id": "a", "label": 1}, {"lineage_id": "b", "label": 0}, {"lineage_id": "b", "label": 1}]
    probabilities = np.asarray([0.2, 0.8, 0.3, 0.7])
    calibrated = crossfit_platt(rows, probabilities, folds=2)
    assert calibrated.shape == probabilities.shape
```

- [ ] **Step 2: Run it and observe failure**

Run: `pytest tests/test_v4_calibrate.py::test_crossfit_platt_holds_out_complete_lineages -v`

Expected: FAIL because `crossfit_platt` does not exist.

- [ ] **Step 3: Implement group-cross-fitted Platt scaling**

Use `GroupKFold`; transform probabilities to clipped logits; fit `LogisticRegression` on each training-fold logit and predict the held-out lineage fold. Reject fewer than two lineages or folds that lack both classes. Fit a final all-row mapper separately and persist only coefficients/intercept.

- [ ] **Step 4: Add report assertions and implement report fields**

Assert `calibration.json` contains `raw_calibration_metrics`, `crossfit_platt_metrics`, and `final_platt_scaler`. Preserve existing operating-point fields exactly.

- [ ] **Step 5: Run calibration tests and commit**

Run: `pytest tests/test_v4_calibrate.py -v`

Commit:
```bash
git add ml/humanized_detector/v4_calibrate.py ml/tests/test_v4_calibrate.py
git commit -m "feat: cross-fit V4.8 Platt calibration"
```

### Task 5: Optimisation controls and Colab guide

**Files:**
- Modify: `ml/humanized_detector/v3_train.py`
- Modify: `ml/humanized_detector/v4_train.py`
- Modify: `ml/tests/test_v3_train.py`
- Modify: `docs/v4-control-runs.md`

**Interfaces:**
- `train_v3_model(train_file: Path, development_file: Path, artifact_dir: Path, config: ModelConfig, variant: str, epochs: int = 6, batch_size: int = 64, learning_rate: float = 3e-5, weight_decay: float = 0.01, label_smoothing: float = 0.1, warmup_steps: int = 0, gradient_clip_norm: float | None = None)` preserves defaults.
- V4 CLI adds `--warmup-steps`, `--grad-clip-norm`, and `--label-smoothing`.

- [ ] **Step 1: Write a failing gradient-clipping test**

```python
def test_train_v3_accepts_zero_warmup_and_optional_gradient_clipping(tmp_path: Path) -> None:
    rows = [
        {"id": "h", "text": "human prose", "label": 0, "source": "test"},
        {"id": "a", "text": "generated prose", "label": 1, "source": "test"},
    ]
    for name in ("train", "development"):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = train_v3_model(tmp_path / "train.jsonl", tmp_path / "development.jsonl", tmp_path / "artifacts", ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=8), "text_mean", epochs=1, batch_size=2, label_smoothing=0.02, warmup_steps=0, gradient_clip_norm=1.0)
    assert result.checkpoint.exists()
```

- [ ] **Step 2: Run it and observe failure**

Run: `pytest tests/test_v3_train.py::test_train_v3_accepts_zero_warmup_and_optional_gradient_clipping -v`

Expected: FAIL because the new arguments are absent.

- [ ] **Step 3: Implement a warm-up-plus-cosine scheduler and clipping**

Use a LambdaLR stepped after every optimiser update. Its multiplier rises linearly from `1 / warmup_steps` to one during warm-up, then follows a cosine decay to the `1e-6 / learning_rate` multiplier by the final update. Invoke `torch.nn.utils.clip_grad_norm_` only when the optional norm is non-null. Preserve default V3 results by using zero warm-up and no clipping.

- [ ] **Step 4: Document exact V4.8 commands**

Document the pooling-only run first, then the one compound recipe run: `--token-pooling mean --lr 1e-4 --warmup-steps 400 --grad-clip-norm 1.0 --label-smoothing 0.02`. State that the latter is a compound ablation.

- [ ] **Step 5: Run the full suite and commit**

Run: `pytest -q`

Commit:
```bash
git add ml/humanized_detector/v3_train.py ml/humanized_detector/v4_train.py ml/tests/test_v3_train.py docs/v4-control-runs.md
git commit -m "feat: add V4.8 optimization controls"
```

## Self-review

- Every V4.8 component reads only its permitted partitions and leaves V4.0/RAID untouched.
- Diagnostics precede optimisation changes; masked pooling is isolated before the compound optimiser trial.
- The plan separates internal PADBen diagnostics from Beemo OOD model selection.
- Cross-fitted calibration evaluates calibration quality without using development or sealed data.
