# OmniCap-IF: Benchmarking and Improving Instruction Following Abilities for Omni-Video Captioning

[![Project Page](https://img.shields.io/badge/Project%20Page-OmniCap--IF-1B2838?logo=githubpages&logoColor=white)](https://wang-jiahao.github.io/OmniCap-IF/)
&nbsp;
[![Model 7B](https://img.shields.io/badge/Model-OmniCaptioner--IF--7B-2563eb)](https://huggingface.co/NJU-LINK/OmniCaptioner-IF-7B)
&nbsp;
[![Model 3B](https://img.shields.io/badge/Model-OmniCaptioner--IF--3B-2563eb)](https://huggingface.co/NJU-LINK/OmniCaptioner-IF-3B)
&nbsp;
[![Trainset](https://img.shields.io/badge/Trainset-OmniCap--IF--54K-059669)](https://huggingface.co/datasets/NJU-LINK/OmniCap-IF-54K)
&nbsp;
[![Testset](https://img.shields.io/badge/Testset-OmniCap--IF-d97706)](https://huggingface.co/datasets/NJU-LINK/OmniCap-IF)

## Introduction

This repository contains a Python pipeline for generating per-video `check_result` files from:
- prompt definitions (`annotation/prompts.json`)
- checklist definitions (`annotation/checklists.json`)

A small example set of videos is included under `videos/`

The GitHub Pages source lives in [`docs/`](docs/).

## Folder layout

- `generate_check_result.py`: main multi-threaded pipeline.
- `annotation/`: prompts, checklists, video metadata.
- `llm_judge/`: meta-prompts for format and content judging.
- `utils/`: format checking and helper utilities.
- `videos/`: example video clips.

## Quick start

### 1) Install dependencies

```bash
pip install openai google-genai tqdm pandas openpyxl
```

### 2) Prepare API keys

The pipeline may read local keys from `api.json` or environment variables, depending on the judge/model backend.

### 3) Run the pipeline

Run the default example (uses `response/example_model_response.json` and `annotation/`):

```bash
python generate_check_result.py --models example_model --meta_dir ./annotation --response_dir ./response --output_dir ./check_result
```

## Compute metrics

After `check_result/<MODEL>_check_result.json` files are generated, you can compute summary metrics (CSR/ISR and breakdowns).

### Run

Compute metrics for specific models:

```bash
python metrics.py --models example_model
```
