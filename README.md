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

1. Keep English conversations at `evaluation_order == 1`, and use only the first user prompt and the first model response.
2. Map model versions (e.g. dated releases, thinking variants) to families: OpenAI, Anthropic, Google, Meta, Qwen, DeepSeek _(final set depends on available counts)_.
3. Mask model and company names in responses (`[MODEL]`) to prevent the classifier from relying on self-identification. Both raw and masked text are kept for ablation.
4. Downsample to an equal number of examples per family.
5. Split train / validation / test = 70 / 10 / 20, grouped by prompt hash so that no prompt appears in more than one split.

## Models

Both models share the same tokenizer, vocabulary, data loaders, training loop and evaluation code. Only the model class differs, so performance differences can be attributed to the architecture.

**Common interface:** input is a batch of token-id sequences (padded/truncated to a fixed length); output is one logit per class.

- **CNN (TextCNN):** word embedding → parallel 1D convolutions with several kernel widths (e.g. 3, 4, 5) → max-over-time pooling → dropout → linear classifier.
- **RNN (BiLSTM):** word embedding → bidirectional LSTM → pooled hidden states → dropout → linear classifier.

**Training:** cross-entropy loss, Adam optimizer, early stopping on validation macro-F1. Hyperparameters and the final configuration of each model will be listed in the report.

**Evaluation:** accuracy, macro-F1, per-class precision/recall and confusion matrices on the held-out test set, averaged over multiple random seeds.

## Repository Structure

```
.
├── notebooks/
│   ├── 01_data_prep.ipynb    # download, filter, label, mask, balance, split
│   ├── 02_train_cnn.ipynb    # CNN training and evaluation
│   ├── 03_train_rnn.ipynb    # LSTM training and evaluation
│   └── 04_analysis.ipynb     # RQ2–RQ4 experiments and figures
├── src/
│   ├── data.py               # tokenization, vocabulary, datasets, loaders
│   ├── models.py             # TextCNN and BiLSTM classes
│   └── train.py              # shared training and evaluation loop
├── results/                  # metrics (CSV/JSON) and figures
├── report/
│   └── report.pdf
└── README.md
```

## How to Run

All experiments run on Google Colab with a GPU runtime.

1. Run `notebooks/01_data_prep.ipynb`. Processed data is saved to Google Drive under `MyDrive/cmpsc448_midterm/`.
2. Run `02_train_cnn.ipynb` and `03_train_rnn.ipynb` (in either order).
3. Run `04_analysis.ipynb` for RQ2–RQ4.

Main dependencies: Python 3, PyTorch, pandas, pyarrow, scikit-learn, huggingface_hub.

## Timeline

| By | Milestone |
|----|-----------|
| Oct 4 | Preprocessing and shared pipeline done; report outline drafted |
| Oct 7 | CNN and LSTM trained and tuned |
| Oct 9 | RQ1–RQ4 experiments finished |
| Oct 11 | Report finalized and repository submitted |

## Results
