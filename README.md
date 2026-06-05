# **OmniCap-IF: Benchmarking and Improving Instruction Following Abilities for Omni-Video Captioning**

[![Project Page](https://img.shields.io/badge/Project%20Page-OmniCap--IF-1B2838?logo=githubpages&logoColor=white)](https://nju-link.github.io/OmniCap-IF/)
&nbsp;
[![Model 7B](https://img.shields.io/badge/Model-OmniCaptioner--IF--7B-2563eb)](https://huggingface.co/NJU-LINK/OmniCaptioner-IF-7B)
&nbsp;
[![Model 3B](https://img.shields.io/badge/Model-OmniCaptioner--IF--3B-2563eb)](https://huggingface.co/NJU-LINK/OmniCaptioner-IF-3B)
&nbsp;
[![Trainset](https://img.shields.io/badge/Trainset-OmniCap--IF--54K-059669)](https://huggingface.co/datasets/NJU-LINK/OmniCap-IF-54K)
&nbsp;
[![Testset](https://img.shields.io/badge/Testset-OmniCap--IF-d97706)](https://huggingface.co/datasets/NJU-LINK/OmniCap-IF)

## Overview

**OmniCap-IF** is a benchmark for evaluating instruction-following abilities in omni-modal video captioning. It evaluates both format correctness and content correctness across visual, audio, and audio-visual constraints, with temporal grounding for fine-grained spatio-temporal verification.

<p align="center">
  <img src="docs/static/images/overview_framework.png" width="92%" alt="OmniCap-IF evaluation framework">
</p>

---

## Quick Start

### Clone

```bash
git clone https://github.com/NJU-LINK/OmniCap-IF.git
cd OmniCap-IF
```

### Installation

```bash
pip install openai google-genai tqdm pandas openpyxl
```

### Usage

Generate per-video check results from prompt and checklist annotations:

```bash
python generate_check_result.py \
  --models example_model \
  --meta_dir ./annotation \
  --response_dir ./response \
  --output_dir ./check_result
```

Compute CSR/ISR metrics:

```bash
python metrics.py --models example_model
```

---

## Evaluation on OmniCap-IF

Download the OmniCap-IF testset from Hugging Face:

```bash
hf download NJU-LINK/OmniCap-IF --repo-type dataset --local-dir OmniCap-IF-testset
```

Prepare model responses under `response/`:

```text
response/
  YourModel/
    001.json
    002.json
    ...
```

Run checklist-based evaluation:

```bash
export JUDGE_API_KEY=YOUR_KEY
export JUDGE_MODEL=gpt-5-mini

python generate_check_result.py \
  --models YourModel \
  --meta_dir ./annotation \
  --response_dir ./response \
  --output_dir ./check_result

python metrics.py --models YourModel
```

---

## License

Our dataset is under the CC-BY-NC-SA-4.0 license.

---

## Citation

```bibtex
@article{wang2026omnicapif,
  title   = {OmniCap-IF: Benchmarking and Improving Instruction Following Abilities for Omni-Video Captioning},
  author  = {Wang, Jiahao and Ping, An and Wang, Yanghai and Zhang, Yuanxing and Li, Shihao and Bian, Hanyan and Ren, Yichi and Zhang, Yize and Wang, Han and Chen, Haowen and Li, Junze and Wang, Jiaqi and Hu, Yiyang and Xu, Zhuze and Zhang, Zijie and Liu, Jiaheng},
  journal = {Preprint},
  year    = {2026}
}
```
