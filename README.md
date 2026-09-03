# Humanized-AI Likelihood Detector

A small Transformer classifier trained from random weights to estimate whether English text resembles deeply paraphrased AI examples in PADBen. It is an educational experiment, not authorship proof or an academic-integrity tool.

## What is included

- deterministic PADBen `exhaustive-task5` preparation;
- byte-level BPE plus a compact four-layer Transformer;
- validation-F1 checkpoint selection and ONNX export;
- a Colab notebook that stores only generated artifacts in Drive.

## Fastest path: Google Colab

Open `notebooks/train_humanized_ai_detector.ipynb` in Colab, select **T4 GPU**, and run cells from top to bottom. The notebook downloads PADBen to the runtime, saves the model/tokenizer/metrics under `MyDrive/humanized-ai-likelihood-artifacts`, and never puts a dataset or model in Git.

## Important limitation

The result is a similarity score relative to PADBen's deeply paraphrased-AI examples. It does not prove that a person used AI, a paraphrasing tool, or any particular humanizer. External Beemo evaluation is required before making a robustness claim.

## Sources

- [PADBen](https://huggingface.co/datasets/JonathanZha/PADBen), MIT licence.
- [Beemo](https://github.com/Toloka/beemo), retained only for separate evaluation; consult its source/model licence terms.
