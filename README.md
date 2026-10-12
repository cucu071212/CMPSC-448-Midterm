# Who Wrote It? Identifying LLMs from Their Responses

**CMPSC 448 Midterm Project — Fall 2026**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cucu071212/CMPSC-448-Midterm/blob/main/Training.ipynb)

This project studies whether text classifiers can identify which LLM family generated a response. We compare a **TextCNN** and a **bidirectional LSTM** on responses from three model families:

- GPT
- Claude
- Gemini

The project focuses on two main questions:

1. **RQ1:** Can we identify the LLM family from the model response alone?
2. **RQ2:** Does including the user's prompt improve identification?

Extra-credit work also explores cross-domain generalization and model-specific response characteristics.

---

## Team

| Role   | Name        | Responsibilities                                                  |
| ------ | ----------- | ----------------------------------------------------------------- |
| Member | Charlie Kim | Data preprocessing, shared training pipeline, RNN (LSTM), RQ1–RQ2 |
| Leader | Jonghun Won | CNN, RQ3–RQ4 analysis, report editing                             |

---

## Results

All results below are from the balanced test set with **4,374 examples** (**1,458 per class**).

| Model   | Input              | Test Accuracy |   Macro-F1 | Best Epoch |
| ------- | ------------------ | ------------: | ---------: | ---------: |
| CNN     | output only        |        92.96% |     0.9296 |          6 |
| RNN     | output only        |        92.43% |     0.9243 |         10 |
| CNN     | input only         |        34.09% |     0.3396 |          6 |
| RNN     | input only         |        34.50% |     0.3436 |          8 |
| **CNN** | **input + output** |    **93.39%** | **0.9339** |      **8** |
| RNN     | input + output     |        91.06% |     0.9106 |         15 |

Because the dataset contains three balanced classes, random-chance accuracy is approximately **33.3%**.

### Main findings

**RQ1 — Can the response alone identify the LLM?**  
Yes. Both models achieve about **92–93% accuracy** using only the generated response, showing that GPT, Claude, and Gemini outputs contain strong model-specific linguistic and formatting signals.

**RQ2 — Does the prompt help?**  
The prompt alone performs near chance level, so prompts do not reveal the model identity. Adding the prompt to the response gives a small improvement for CNN (**92.96% → 93.39%**) but lowers RNN performance (**92.43% → 91.06%**). This suggests that most of the useful signal comes from the model response itself.

---

## Dataset

We use the public **LMArena `arena-human-preference-140k`** dataset, which contains real user prompts and responses from many LLMs.

### Preprocessing

The pipeline:

1. Scans all 135,634 battles.
2. Keeps English, text-only, single-turn conversations.
3. Keeps examples where both models received the same prompt.
4. Maps model IDs into three families:
   - `gpt-` / `chatgpt-` → **GPT**
   - `claude-` → **Claude**
   - `gemini-` → **Gemini**
5. Removes duplicate responses.
6. Splits by normalized prompt hash so the same prompt cannot appear across train/validation/test.
7. Downsamples each split to keep the three classes balanced.

| Split      |   Rows | Per Family |
| ---------- | -----: | ---------: |
| Train      | 20,844 |      6,948 |
| Validation |  4,779 |      1,593 |
| Test       |  4,374 |      1,458 |

Labels:

```text
GPT    = 0
Claude = 1
Gemini = 2
```

---

## Input Representation

The tokenizer preserves stylistic features that may act as LLM fingerprints.

- Text is lowercased.
- Words and symbols are tokenized separately.
- Markdown markers such as `**`, `##`, and `---` are preserved.
- Emoji and punctuation are preserved.
- Newlines become `<nl>`.
- Vocabulary is built only from the training split.
- Maximum vocabulary size: 30,000.
- Maximum sequence length: 512 tokens.

Three input modes are used:

```text
output : LLM response only
input  : user prompt only
both   : first 128 prompt tokens + <sep> + LLM response
```

---

## Models

### TextCNN

```text
Embedding (128)
    ↓
Parallel Conv1D
kernel sizes = 3, 4, 5
100 filters each
    ↓
Max-over-time pooling
    ↓
Dropout (0.5)
    ↓
Linear classifier (3 classes)
```

The CNN is designed to capture local lexical, stylistic, and formatting patterns.

### Bidirectional LSTM

```text
Embedding (128)
    ↓
Bidirectional LSTM
hidden size = 128
1 layer
    ↓
Final forward + backward hidden states
    ↓
Dropout (0.5)
    ↓
Linear classifier (3 classes)
```

Packed sequences are used so padding is ignored by the LSTM.

---

## Training

Both models use the same training and evaluation pipeline.

| Setting           | Value               |
| ----------------- | ------------------- |
| Loss              | Cross-entropy       |
| Optimizer         | AdamW               |
| Learning rate     | 1e-3                |
| Weight decay      | 0.01                |
| Batch size        | 64                  |
| Gradient clipping | 1.0                 |
| Maximum epochs    | 15                  |
| Early stopping    | Patience = 3        |
| Selection metric  | Validation macro-F1 |
| Random seed       | 42                  |

The best checkpoint is selected using validation macro-F1 and evaluated **once** on the test set.

Evaluation includes:

- Accuracy
- Macro-F1
- Per-class precision
- Per-class recall
- Confusion matrix

---

## Repository Structure

```text
.
├── README.md
├── data.py
├── prepare_data.py
├── train.py
├── requirements.txt
├── Preprocessing.ipynb
├── Training.ipynb
├── verify_pipeline.py
│
├── src/
│   ├── cnn.py
│   ├── rnn.py
│   └── Traing.ipynb
│
├── rq3/
│   ├── create_notebook.py
│   ├── prepare_domains.py
│   ├── RQ3_Training.ipynb
│   └── verify_notebook.py
│
└── explore_data.py
```

---

## Reproduction

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Run:

```bash
python prepare_data.py
```

or use:

```text
Preprocessing.ipynb
```

### 3. Train a model

Example:

```bash
python train.py --model cnn --mode output
```

Available model choices:

```text
cnn
rnn
```

Available input modes:

```text
output
input
both
```

### 4. Run the full experiment

Use `Training.ipynb` to train and compare all six RQ1/RQ2 configurations.

For a quick pipeline check, run:

```bash
python verify_pipeline.py
```

---

## Research Questions

### RQ1: Can we identify which LLM generated a response?

**Yes.**  
CNN reaches **92.96%** accuracy and RNN reaches **92.43%** accuracy using only the generated response.

This strongly suggests that model families leave identifiable response-level fingerprints.

### RQ2: Does the user's prompt help identify the LLM?

Mostly **no**.

Prompt-only accuracy is approximately chance:

```text
CNN: 34.09%
RNN: 34.50%
```

The CNN gains only **0.43 percentage points** when the prompt is added to the response, while the RNN performs worse.

The response itself therefore contains nearly all of the discriminative information.

### RQ3: Do LLM fingerprints generalize across task domains?

The `rq3/` directory contains the cross-domain experiment pipeline for comparing in-domain and cross-domain performance using the same CNN/RNN framework.

### RQ4: What characteristics distinguish different LLMs?

`explore_data.py` examines response-level characteristics such as:

- response length
- markdown formatting
- headers
- lists
- code blocks
- emoji
- em dashes
- common opening phrases

These features help interpret what stylistic signals the classifiers may be learning.

---

## Reproducibility Notes

- The random seed is fixed.
- Train/validation/test splits are prompt-disjoint.
- Vocabulary is built from training data only.
- CNN and RNN use the same tokenizer, splits, data loaders, and evaluation procedure.
- Classes are balanced independently within each split.
- Test performance is measured only after selecting the best validation checkpoint.

---

## Course

**CMPSC 448 — Machine Learning and Algorithmic AI**  
Penn State University, Fall 2026
