# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 Midterm Project (Fall 2026)

| Role | Name | Responsibilities |
|------|------|------------------|
| Member | Charlie Kim | Data preprocessing, shared training pipeline, RNN (LSTM), RQ1–RQ2 |
| Member | Jonghun Won | CNN, RQ3–RQ4 analysis, report editing |

Project report: `report/report.pdf`

## Overview

Different large language models may leave distinct linguistic, stylistic and structural "fingerprints" in their responses. This project asks whether a classifier can identify which LLM family generated a given response, and what signals make this possible. We implement a Convolutional Neural Network (CNN) and a Recurrent Neural Network (bidirectional LSTM) and compare them under the same training pipeline.

## Research Questions

| RQ | Question | Setup |
|----|----------|-------|
| RQ1 | Can we identify which LLM generated a response? | Input to classifier: model response only |
| RQ2 | Does the user's prompt help identify the LLM? | Compare input only / output only / input + output |
| RQ3 _(extra credit)_ | Do LLM fingerprints generalize across task domains? | Train on some domains (e.g. code, math), test on others (e.g. creative writing) |
| RQ4 _(extra credit)_ | What characteristics distinguish different LLMs? | Response length, vocabulary, lexical diversity, Markdown formatting; ablation by masking or removing signals |

## Dataset

We use the public [LMArena arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) dataset, which contains real user conversations with recent LLMs collected on LMArena between April and July 2025. In each record two models answer the same user prompt, which gives a natural control for RQ2. The dataset also provides domain labels (`is_code`, `category_tag`) for RQ3 and Markdown/length statistics (`conv_metadata`) for RQ4.

User prompts are licensed under CC-BY-4.0; model outputs are subject to the terms of use of each model provider. Raw data is not committed to this repository; the notebooks download it directly from Hugging Face.

### Preprocessing

1. Stream the pinned dataset revision `6322995ab34d7c2693e3f47dd13fa5caa0789a74`, anonymously (`token=False`).
2. Keep English, text-only, single-turn conversations where both sides received the same normalized prompt, matching the existing `Preprocessing.ipynb` protocol.
3. Map `gpt-` / `chatgpt-`, `claude-`, and `gemini-` model prefixes to GPT=0, Claude=1, Gemini=2. Other models are excluded. Responses remain raw; model-name masking is not applied by this pipeline.
4. Remove duplicate (prompt, family, response) examples. Split by normalized prompt SHA-256 using seed 42: 70/15/15 train/validation/test before balancing. Downsample each split independently to equal family counts; final row proportions can differ.
5. Generate `train.csv`, `validation.csv`, `test.csv` and `preparation.json` in `data/`. Validate nonempty text, columns, label mapping, all classes, hashes, duplicates and disjoint prompt groups before training. Valid existing CSVs are reused; missing or invalid splits trigger regeneration.

The pinned source revision and seed make regeneration reproducible with the same dependency versions. `preparation.json` records the source, seed and split counts; every training run records its settings and vocabulary. CSV validation checks structural integrity and leakage, not source provenance: remove custom CSVs if you want to regenerate the canonical dataset.

## Models

Both models share the same tokenizer, vocabulary, data loaders, training loop and evaluation code. Only the model class differs, so performance differences can be attributed to the architecture.

**Common interface:** input is a batch of token-id sequences (padded/truncated to a fixed length); output is one logit per class.

- **CNN (TextCNN):** word embedding → parallel 1D convolutions with several kernel widths (e.g. 3, 4, 5) → max-over-time pooling → dropout → linear classifier.
- **RNN (BiLSTM):** word embedding → bidirectional LSTM → pooled hidden states → dropout → linear classifier.

**Training:** cross-entropy loss, Adam optimizer, early stopping on validation macro-F1. Hyperparameters and the final configuration of each model will be listed in the report.

**Evaluation:** accuracy, macro-F1, per-class precision/recall and confusion matrices on the held-out test set, averaged over multiple random seeds.

## Repository Structure

```text
Training.ipynb       # Colab setup, CSV preparation, six experiments, downloads
Preprocessing.ipynb  # Original standalone preprocessing notebook
prepare_data.py      # Reusable public download, preprocessing and CSV validation
data.py             # Tokenizer, vocabulary and shared data loaders
train.py            # Shared CNN/RNN training and evaluation CLI
src/cnn.py          # TextCNN
src/rnn.py          # BiLSTM
verify_pipeline.py  # Isolated synthetic preparation and training smoke checks
requirements.txt    # Notebook and local dependencies
data/               # Generated CSVs (not committed)
results/            # Run directories and comparison (not committed)
```

## How to Run

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cucu071212/CMPSC-448-Midterm/blob/rnn/Training.ipynb)

1. Open `Training.ipynb` using the Colab link above (the pipeline is on the `rnn` branch).
2. Select **Runtime > Change runtime type > T4 GPU** (or another available GPU).
3. Select **Runtime > Run all**. The notebook clones the repository if absent, installs packages, downloads/preprocesses data if valid CSVs are absent, validates splits, and runs all six CNN/RNN experiments.
4. Download the ZIP from the final cell before the runtime resets. It contains checkpoints, vocabulary, metrics, curves, predictions, and the comparison CSV. Training all six experiments can take hours and may exceed a free Colab session; reduce epochs for an initial smoke run.

No personal paths, pre-generated CSVs, Google Drive mount, Hugging Face account, or token are required. On 2026-10-10, anonymous Hugging Face API access confirmed `private=false` and `gated=false` for the [dataset](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k). If access restrictions change, review its page, obtain any required access, and adapt authentication before retrying. Initial preparation streams about 1.61 GB and can take several minutes.

The first setup cell uses the actual repository remote:

```python
REPO_URL = 'https://github.com/cucu071212/CMPSC-448-Midterm.git'
REPO_REF = 'rnn'
PROJECT_DIR = Path('/content/CMPSC-448-Midterm')
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'results'
```

An existing checkout is used as-is. Set `UPDATE_CHECKOUT = True` to request `git pull --ff-only`; the notebook refuses updates when local changes exist and never resets or forces the checkout. When opening a notebook before updating an existing checkout, reopen the notebook afterward to load its updated cells.

For a manual first checkout in Colab:

```bash
!git clone --branch rnn https://github.com/cucu071212/CMPSC-448-Midterm.git /content/CMPSC-448-Midterm
```

Use `git pull --ff-only` only inside an existing checkout. The notebook adds `PROJECT_DIR` to `sys.path` and calls scripts by absolute path, so imports do not depend on the notebook's current directory.

`/content` data and results are temporary and disappear when the runtime resets. CSVs will be regenerated next time. Drive is an optional backup only: set `SAVE_TO_DRIVE = True` in the final cell to copy CSVs and results to a new timestamped folder. This introduces a mount prompt only for users who enable it. To reuse a backup later, copy its CSVs into `DATA_DIR` before the preparation cell; they are validated normally.

Local CLI (Python 3.10+):

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python train.py --model cnn --mode output
python train.py --model rnn --mode output
```

Validation on 2026-10-10 (local Python 3.11 / CPU):

- Installed `requirements.txt` in a clean virtual environment.
- Confirmed the GitHub repository is public, and accessed the complete pinned Hugging Face dataset with `token=False`.
- Generated and validated real CSVs: train 20,844 rows (6,948 per family), validation 4,779 (1,593 per family), test 4,374 (1,458 per family). Prompt groups are disjoint. A second preparation run reused these CSVs without downloading.
- `python verify_pipeline.py` passed notebook code-cell syntax checks, deterministic preprocessing, CSV reuse, invalid-CSV regeneration, prompt-leakage rejection, and all six CNN/RNN input modes with one CPU epoch on synthetic data. Training was launched from a different working directory and checkpoints, metrics and plots were saved.
- Trained CNN/output on all real CSV rows for one CPU epoch with a reduced smoke configuration (32 tokens, embedding dimension 8, 4 filters); evaluation, checkpoint reload, metrics and plots completed successfully.

Not verified: a complete fresh Colab GPU Run all, the six default experiments on real data with a GPU, browser downloads, or optional Drive mounting/backups. Local smoke metrics are not research results.

## Timeline

| By | Milestone |
|----|-----------|
| Oct 4 | Preprocessing and shared pipeline done; report outline drafted |
| Oct 7 | CNN and LSTM trained and tuned |
| Oct 9 | RQ1–RQ4 experiments finished |
| Oct 11 | Report finalized and repository submitted |

## Results
