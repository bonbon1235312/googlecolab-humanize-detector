# Humanized-AI Likelihood Detector — Design Specification

## Purpose

Build a small, from-scratch English-language classifier that estimates whether a passage resembles **deeply paraphrased or humanized AI text** rather than genuine human writing. It is an educational portfolio project and must never claim to prove authorship, academic misconduct, use of a specific tool, or detection of every paraphrase method.

The product wording is **humanized-AI likelihood**. A positive result means only that the text is more similar to the model's humanized-AI training examples.

## Data strategy

### Training and development: PADBen

Use the MIT-licensed [PADBen](https://huggingface.co/datasets/JonathanZha/PADBen) `exhaustive-task5` configuration. Its binary task is deliberately aligned with this project:

- label `0`: original human-written text;
- label `1`: third-iteration LLM paraphrases of generated text (deep paraphrase attack).

The adapter downloads data only into a local or Colab/Drive cache, validates the `text`, `label`, and source-group identifier fields, then writes ignored JSONL splits. It preserves punctuation, casing, and whitespace. It rejects blank passages and exact normalized duplicates.

Rows sharing a PADBen source identifier must stay in the same split. The adapter treats `idx` as the group identifier only after validating that both class variants for a source share it; it fails with an actionable error if this assumption is false. A deterministic group-stratified 70/15/15 split uses seed `20260903`.

### Independent evaluation: Beemo

Use [Beemo](https://github.com/Toloka/beemo) only as an external robustness evaluation, never mixed into PADBen training. It contains genuine human text, machine outputs, expert-edited machine outputs, and LLM-edited/humanized machine outputs. The evaluator will report separate metrics for:

- human-written versus expert-edited AI;
- human-written versus LLM-humanized AI;
- raw-AI versus LLM-humanized AI as a diagnostic, not the product claim.

Beemo licensing includes source and model-specific terms, so the repository stores neither corpus and cites its licensing conditions in the model card.

## Model and training

The production candidate is a compact Transformer encoder trained from random weights:

- byte-level BPE tokenizer trained only on the PADBen training split, vocabulary size `4,000`;
- maximum sequence length `256` tokens;
- four encoder layers, learned positional embeddings, six attention heads, hidden size `192`;
- binary classification head, BCE-with-logits loss, AdamW, and deterministic seeds where supported;
- validation-F1 checkpoint selection, with early stopping and resumable checkpoints;
- ONNX export for CPU-only production inference.

A character n-gram logistic-regression baseline is trained and reported beside the Transformer. The project makes no superiority claim unless the untouched PADBen test metrics support it.

Training writes artifacts to Google Drive in Colab. Raw data, prepared text, tokenizers, checkpoints, ONNX files, and secrets are excluded from Git.

## Evaluation

Report accuracy, precision, recall, F1, ROC-AUC, confusion matrix, calibration bins, and metrics by length bucket for both the PADBen holdout and the separate Beemo evaluation. Report false-positive rate prominently because genuine human text must not be casually flagged.

No metric is invented or copied from a source paper. The model card records the exact dataset configuration, group-split procedure, seed, tokenizer/model configuration, training environment, measured metrics, source licences, and limitations.

## API and web demo

A FastAPI service loads a single ONNX Runtime CPU session and accepts `POST /predict` with one text field. It rejects malformed JSON, unsupported media types, input below `80` characters, and input above `12,000` characters. It stores neither text nor request bodies; logs contain only status, latency, and input length.

The response contains:

- `humanized_ai_likelihood` from 0 to 1;
- a restrained training-distribution label;
- low/moderate/high confidence band based on calibration;
- analysed token count;
- a limitation stating it does not prove authorship or use of a humanizer.

A Vercel frontend provides a paste box, accessible validation/loading/error states, result explanation, and a prominent limitation. It avoids red/green authorship verdicts.

## Deployment and cost safeguards

Cloud Run runs CPU-only under request-based billing with minimum instances `0`, maximum instances `1`, concurrency `1`, memory `1 GiB`, and a finite request timeout. The deployment flow stages the ONNX model and tokenizer as local build-context artifacts after the user downloads them from Drive; they remain ignored by Git.

Vercel hosts only the frontend and receives the API base URL through an environment variable. The API permits only the configured Vercel origin through CORS. The user sets a Google Cloud billing alert; alerts do not guarantee a spending cap.

## Verification and acceptance criteria

Tests cover the PADBen schema and group splitting, tokenizer/model behavior, tiny fixture training, ONNX export, API validation and absence-of-artifact behavior, Cloud Run safeguards, frontend request states, and limitation text. Before deployment, run local API and frontend checks; after deployment, smoke-test `/healthz`, `/predict`, desktop and mobile.

The finished repository can prepare PADBen, train and export a model in Colab, evaluate independently on Beemo, and expose a cautious public demo without committing text datasets or claiming authorship proof.
