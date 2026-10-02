# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 Midterm Project (Fall 2026)

**Team Members:** Charlie Kim, Jonghun Won

## Overview

This project investigates whether machine learning can identify which LLM generated a given response. We implement and compare a CNN and an RNN (LSTM) classifier.

## Research Questions

- **RQ1:** Can we identify which LLM generated a response?
- **RQ2:** Does the user's prompt help identify the LLM?
- **RQ3** _(extra credit)_: Do LLM fingerprints generalize across tasks or domains?
- **RQ4** _(extra credit)_: What characteristics distinguish different LLMs?

## Dataset

Each example has the form `(LLM_name, LLM_input, LLM_output)` and covers at least three LLM families. Data will come from a public dataset (e.g. [LMArena conversations](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)) or a dataset we collect ourselves. Details are in the report.

## Repository Structure

```
.
├── data/        # data scripts / prompts (raw data not committed)
├── notebooks/   # Colab notebooks
├── src/         # model and training code
├── results/     # metrics and figures
└── report/      # project report (PDF)
```

## How to Run

Experiments run on Google Colab. _Instructions to be added._

## Results

_To be added._
