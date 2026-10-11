# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 Midterm Project (Fall 2026)

| Role | Name | Responsibilities |
|------|------|------------------|
| Member | Charlie Kim | Data preprocessing, shared training pipeline, RNN (LSTM), RQ1–RQ2 |
| Leader | Jonghun Won | CNN, RQ3–RQ4 analysis, report editing |

Project report: [`report/report.pdf`](report/report.pdf)

## Quick Start (for reproduction)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cucu071212/CMPSC-448-Midterm/blob/main/reproduce.ipynb)

1. Open `reproduce.ipynb` in Colab with the badge above.
2. Set **Runtime → Change runtime type → GPU**.
3. **Runtime → Run all.**

The notebook clones this repository, installs dependencies, downloads the dataset from Hugging Face, builds the train/validation/test splits, trains all six models and prints the comparison table. No Google Drive access or manual setup is required; all folders are created automatically.

Set `QUICK = True` in the first cell to run a short smoke test (small subset, 1 epoch) that checks the full pipeline in a few minutes. Results from our full run are already included in [`results/`](results/), so the numbers in the report can be checked without retraining.

**Expected time (Colab T4 GPU):** preprocessing ≈ 6 min, each CNN run a few minutes, each RNN run noticeably longer.

## Overview

Different large language models may leave distinct linguistic, stylistic and structural "fingerprints" in their responses. This project asks whether a classifier can identify which LLM family (GPT, Claude or Gemini) generated a given response, and what signals make this possible. We implement a Convolutional Neural Network (CNN) and a Recurrent Neural Network (bidirectional LSTM) and compare them under the same data pipeline and training loop.

## Research Questions

| RQ | Question | Setup |
|----|----------|-------|
| RQ1 | Can we identify which LLM generated a response? | Classifier input: model response only |
| RQ2 | Does the user's prompt help identify the LLM? | Compare input only / output only / input + output |
| RQ3 _(extra credit)_ | Do LLM fingerprints generalize across task domains? | Train on some domains, test on others |
| RQ4 _(extra credit)_ | What characteristics distinguish different LLMs? | Response length, formatting habits, opening phrases, error analysis |

## Dataset

We use the public [LMArena arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) dataset: real user conversations with LLMs collected on LMArena between April and July 2025. In each record two anonymous models answer the same user prompt.

User prompts are licensed under CC-BY-4.0; model outputs are subject to the terms of use of each model provider. Raw and processed data are **not** committed to this repository; the pipeline downloads the data directly from Hugging Face.

### Preprocessing (`preprocess.py`)

1. Scan all 135,634 battles and keep English, text-only, single-turn conversations (exactly one user message and one model answer).
2. Keep battles in which both models received the same prompt.
3. Map model IDs to three families by prefix: `gpt-` / `chatgpt-` → **GPT**, `claude-` → **Claude**, `gemini-` → **Gemini**. All other models are discarded.
4. Remove duplicate responses.
5. Split train / validation / test = 70 / 15 / 15, grouped by a hash of the normalized prompt, so no prompt appears in more than one split.
6. Downsample within each split to an equal number of examples per family.

| Split | Rows | Per family |
|-------|------|------------|
| train | 20,844 | 6,948 |
| validation | 4,779 | 1,593 |
| test | 4,374 | 1,458 |

Columns: `prompt_id, LLM_Input, LLM_output, LLM_name, label` with labels GPT = 0, Claude = 1, Gemini = 2. The random seed is fixed, so the splits are identical on every run.

### Tokenization (`data.py`)

- Lowercase; words and symbols become separate tokens. Markdown markers (`**`, `##`, `---`), emoji, curly vs. straight quotes and em dashes are kept as tokens because they differ between model families; newlines become `<nl>`.
- Vocabulary is built from the training split only (minimum frequency 2, at most 30,000 tokens; `<pad>` = 0, `<unk>` = 1).
- Sequences are truncated or padded to 512 tokens.
- Input modes for RQ2: `output` (response), `input` (prompt), `both` (first 128 prompt tokens + `<sep>` + response).

## Models

Both models share the same tokenizer, vocabulary, data loaders, training loop and evaluation code; only the model class differs.

- **CNN (`model/cnn.py`, TextCNN):** embedding (128) → parallel 1D convolutions with kernel widths 3, 4, 5 (100 filters each) → max-over-time pooling that ignores padding → dropout 0.5 → linear (3 classes).
- **RNN (`model/rnn.py`, BiLSTM):** embedding (128) → bidirectional LSTM (hidden 128, 1 layer) over packed sequences so padding is skipped → final hidden states of both directions → dropout 0.5 → linear (3 classes).

**Training (`train.py`):** cross-entropy loss, AdamW (lr 1e-3, weight decay 0.01), batch size 64, gradient clipping 1.0, up to 15 epochs with early stopping (patience 3) on validation macro-F1. The best validation checkpoint is evaluated **once** on the test set. Seed 42.

**Evaluation:** accuracy, macro-F1, per-class precision/recall and confusion matrix on the test set.

## Experiments

Six runs cover RQ1 and RQ2: {CNN, RNN} × {output, input, both}.

```bash
python run_all.py                                  # preprocessing (if needed) + all six runs + comparison
python train.py --model cnn --mode output          # a single run
```

Runs whose results already exist are skipped, so an interrupted run can simply be restarted.

## Repository Structure

```
.
├── reproduce.ipynb        # one-click reproduction in Colab
├── run_all.py             # preprocessing → six training runs → comparison
├── preprocess.py          # Hugging Face dataset → train/validation/test CSV
├── data.py                # tokenization, vocabulary, padding, data loaders
├── train.py               # training, model selection, test evaluation
├── model/
│   ├── cnn.py             # TextCNN
│   └── rnn.py             # BiLSTM
├── scripts/
│   └── explore_data.py    # dataset statistics used in the report (RQ4)
├── results/
│   ├── comparison.csv     # test metrics of all six runs
│   └── {model}_{mode}/    # config, history, metrics, confusion matrix, predictions, plots
├── report/
│   └── report.pdf
└── requirements.txt
```

Each run writes `config.json`, `dataset_info.json`, `history.csv`, `metrics.json`, `summary.csv`, `confusion_matrix.csv`, `predictions.csv` and `training_and_confusion.png`. Model weights (`best_model.pt`) are not committed.

## Running Locally

```bash
pip install -r requirements.txt
python run_all.py --data-dir data/arena_processed --output-dir results
```

A CUDA GPU is used automatically when available.

## Results

| Model | Input | Test accuracy | Test macro-F1 |
|-------|-------|---------------|---------------|
| CNN | output | | |
| CNN | input | | |
| CNN | both | | |
| RNN | output | | |
| RNN | input | | |
| RNN | both | | |

Chance level is 33.3% (three balanced classes). See the report for confusion matrices and analysis.
