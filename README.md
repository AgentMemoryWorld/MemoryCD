<div align="center">

# MemoryCD

### Benchmarking Long-Context User Memory of LLM Agents for Lifelong Cross-Domain Personalization

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-WZDavid%2FMemoryCD-orange)](https://huggingface.co/datasets/WZDavid/MemoryCD)
[![arXiv](https://img.shields.io/badge/arXiv-2603.25973-b31b1b.svg)](https://arxiv.org/abs/2603.25973)
[![Code style: black](https://img.shields.io/badge/code%20style-pep8-000000.svg)](https://peps.python.org/pep-0008/)

**A minimal, end-to-end evaluation harness for lifelong cross-domain personalization with LLM agents.**

[**Dataset**](https://huggingface.co/datasets/WZDavid/MemoryCD) ·
[**Quick Start**](#quick-start) ·
[**Tasks & Metrics**](#tasks-and-metrics) ·
[**Methods**](#methods) ·
[**Citation**](#citation)

</div>

---

## Highlights

- **Lifelong, real user behavior.** Memory is built from authentic Amazon reviews spanning years, not synthetic LLM-generated transcripts.
- **Two evaluation settings.** Single-domain (memory and test from the same domain) and cross-domain (memory pooled from other domains, test on a target domain).
- **Four end-to-end tasks.** Rating prediction, review summarization, review generation, and item ranking — covering both discriminative and generative use cases.
- **Two memory-selection methods out of the box.** A long-context baseline and a BM25-based RAG retriever, both behind a simple plug-in interface.
- **One unified data file.** Both eval settings read the same cross-domain JSONL — no separate per-domain dumps.
- **OpenAI-compatible.** Works with OpenAI, OpenRouter, or any compatible gateway via env vars — no SDK lock-in.

## Comparison with existing memory benchmarks

<p align="center">
  <img src="assets/MemoryCD-Intro.png" alt="Comparison of memory benchmarks" width="320">
</p>

| Benchmark | Time horizon | Domains | User behavior |
|-----------|--------------|---------|---------------|
| **MemoryCD (ours)** | Long | Cross-domain | **Real users** |
| LaMP (Salemi et al., 2024) | Short | Single-domain | Real users |
| LoCoMo (Maharana et al., 2024) | Long | Single-domain | LLM-simulated |

## Overview

<p align="center">
  <img src="assets/MemoryCD.png" alt="MemoryCD overview">
</p>

**Figure 2.** MemoryCD spans 12 real-world domains and evaluates 6 SOTA memory methods.
Unlike benchmarks that probe only one memory stage (mostly retrieval), we design **4 basic
tasks × 2 settings** to provide end-to-end user-satisfaction evaluation grounded in lifelong
real user behaviors.

---

## Quick start

```bash
# 1. Install (Python 3.10+)
pip install -r requirements.txt

# 2. Configure your LLM provider
export OPENAI_API_KEY=...                # or OPENROUTER_API_KEY=... and LLM_PROVIDER=openrouter

# 3. Grab the dataset
pip install -U huggingface_hub
huggingface-cli download WZDavid/MemoryCD --repo-type dataset --local-dir .

# 4. Run a quick single-domain sanity check (first 5 users)
python evaluation_all_4task.py \
    --task rating_prediction \
    --domain Books \
    --input data/cross_domain.jsonl \
    --method long_context \
    --max-users 5
```

You should see per-user progress and a final summary printed to stdout, plus
`logs/single/Books/predictions_*.jsonl` + `summary_*.json` on disk.

---

## Setup

```bash
# Python 3.10+
pip install -r requirements.txt
# or
pip install -e .
```

Configure the LLM client (any OpenAI-compatible endpoint works):

```bash
# OpenAI
export OPENAI_API_KEY=...

# OpenRouter
export OPENROUTER_API_KEY=...
export LLM_PROVIDER=openrouter
```

## Dataset

The unified cross-domain JSONL plus the per-domain metadata files are released on Hugging Face:

> **https://huggingface.co/datasets/WZDavid/MemoryCD**

After downloading, place the files like this (or pass `--input` and `--meta-dir` to point elsewhere):

```
data/cross_domain.jsonl
meta/meta_Beauty_and_Personal_Care.jsonl.gz
meta/meta_Books.jsonl.gz
meta/meta_Electronics.jsonl.gz
meta/meta_Home_and_Kitchen.jsonl.gz
```

Quick download with the Hugging Face CLI:

```bash
pip install -U huggingface_hub
huggingface-cli download WZDavid/MemoryCD --repo-type dataset --local-dir .
```

<details>
<summary><b>Input-data format</b> (click to expand)</summary>

Both evaluation scripts read a single **cross-domain JSONL** file. Each line is one user:

```json
{"user_id": "...", "interactions": {
    "Beauty_and_Personal_Care": [{"parent_asin": "...", "rating": 5, "title": "...",
                                  "text": "...", "timestamp": 1700000000,
                                  "negative_parent_asins": ["...", "..."]}],
    "Books": [...],
    "Electronics": [...],
    "Home_and_Kitchen": [...]
}}
```

Metadata files `meta_<Domain>.jsonl[.gz]` (one item per line, with `parent_asin`, `title`,
`average_rating`, `main_category`, optional `description`) should live in `meta/` by default.

Required fields per interaction:
- `parent_asin`, `rating`, `title`, `text`, `timestamp`
- `negative_parent_asins` (only needed for `item_ranking`)

</details>

## Tasks and metrics

| Task                    | Output             | Metrics                                  |
|-------------------------|--------------------|------------------------------------------|
| `rating_prediction`     | integer 1–5        | MAE, RMSE                                |
| `review_summarization`  | short review title | ROUGE-1/L, BLEU-1/4, BERTScore           |
| `review_generation`     | full review body   | ROUGE-1/L, BLEU-1/4, BERTScore           |
| `item_ranking`          | ranked ASIN list   | NDCG@K, Recall@K (default K = 1, 3, 5)   |

## Methods

| `--method`     | Behavior                                                                                          |
|----------------|---------------------------------------------------------------------------------------------------|
| `long_context` | Keep the most-recent N interactions by recency.                                                   |
| `rag`          | BM25 retrieval over memory; top-K via `--max-memory-items`. Persistent disk cache under `cache/`. |

> Adding a new method is a single file in `methods/` plus one branch in `eval_core.build_method()`.

## Settings

### Single-domain — `evaluation_all_4task.py`

Memory and test both come from the same domain. For each user, the last `--num-test`
interactions in `--domain` are the test set; earlier ones are the memory.

```bash
python evaluation_all_4task.py \
    --task rating_prediction \
    --domain Books \
    --input data/cross_domain.jsonl \
    --llm-model openai/gpt-5

python evaluation_all_4task.py \
    --task item_ranking \
    --domain Electronics \
    --input data/cross_domain.jsonl \
    --method rag --max-memory-items 10 \
    --k-values 1 3 5
```

### Cross-domain — `evaluation_all_4task_cross_domain.py`

Memory comes from one or more **source** domains; test comes from a **target** domain.

```bash
# Default: memory = the 3 non-target domains
python evaluation_all_4task_cross_domain.py \
    --task rating_prediction \
    --target-domain Home_and_Kitchen \
    --input data/cross_domain.jsonl \
    --llm-model openai/gpt-5

# Subset of source domains
python evaluation_all_4task_cross_domain.py \
    --task review_summarization \
    --target-domain Home_and_Kitchen \
    --source-domains Books Electronics

# No-memory baseline
python evaluation_all_4task_cross_domain.py \
    --task rating_prediction \
    --target-domain Books \
    --no-memory
```

## Common flags

| Flag                 | Description                                                   |
|----------------------|---------------------------------------------------------------|
| `--task`             | One of the 4 tasks above.                                     |
| `--input`            | Path to the unified cross-domain JSONL.                       |
| `--meta-dir`         | Directory with `meta_<Domain>.jsonl[.gz]` (default `meta/`).  |
| `--llm-model`        | Model identifier for the OpenAI-compatible client.            |
| `--method`           | `long_context` (default) or `rag`.                            |
| `--max-memory-items` | Cap on memory items kept after selection.                     |
| `--num-test`         | Number of last interactions per user used as test (default 3).|
| `--max-users`        | Limit to first N users for quick smoke tests.                 |
| `--k-values`         | `--k-values 1 3 5` (item_ranking only).                       |
| `--log-dir`          | Output log directory (default `logs/single` or `logs/cross`). |

## Outputs

Each run writes two files:

- `logs/<setting>/<domain>/predictions_<run_id>.jsonl` — one entry per prediction with
  the full prompt, raw LLM response, prediction, ground truth, and per-prediction metrics.
- `logs/<setting>/<domain>/summary_<run_id>.json` — aggregated final metrics plus RAG
  cache statistics (when applicable).

`<run_id>` encodes setting + domain + task + method + model + caps + timestamp, so
every experiment lands in its own pair of files.

## Reproduce the full sweep

Two parallel drivers run the full grid of tasks × methods × models × domains:

```bash
# Single-domain sweep
export API_KEY=$OPENAI_API_KEY
./run_evaluation_pa.sh 4               # 4 parallel jobs

# Cross-domain sweep
./run_evaluation_cross_domain.sh 4
```

Edit the `DOMAINS` / `TARGET_DOMAINS` / `TASKS` / `METHODS` / `MODELS` arrays at the
top of each script to scope the run.

## Repository layout

```
api.py                                   LLM gateway client (OpenAI / OpenRouter)
eval_core.py                             Shared library: loaders, evaluators, predictor, prompts
evaluation_all_4task.py                  Single-domain entry point
evaluation_all_4task_cross_domain.py     Cross-domain entry point
methods/
  __init__.py
  base_method.py                         Abstract memory-selection interface
  long_context.py                        Recency-based selection
  rag.py                                 BM25 retrieval with persistent cache
run_evaluation_pa.sh                     Parallel driver — single domain
run_evaluation_cross_domain.sh           Parallel driver — cross domain
requirements.txt                         Pinned core dependencies
pyproject.toml                           PEP-621 manifest
assets/                                  Figures used in this README
```

## FAQ

<details>
<summary><b>Do I need GPUs?</b></summary>

No. All four tasks call an external LLM API. The only optional GPU usage is BERTScore,
which falls back to CPU automatically (it's a touch slower but still runs end-to-end).
</details>

<details>
<summary><b>Can I evaluate my own model?</b></summary>

Yes — set `OPENAI_BASE_URL` (or `OPENROUTER_BASE_URL`) to any OpenAI-compatible endpoint
and pass the model name via `--llm-model`. Self-hosted servers (vLLM, llama.cpp,
TGI, Ollama with its OpenAI shim) all work out of the box.
</details>

<details>
<summary><b>How do I add a new memory method?</b></summary>

1. Subclass `BaseMethod` in `methods/your_method.py` and implement `select_memory(...)`.
2. Register it in `methods/__init__.py`.
3. Add one `elif` branch in `eval_core.build_method()` and a CLI choice in both eval entry points.

The shared evaluator, logger, and prompt builders in `eval_core.py` don't need to change.
</details>

<details>
<summary><b>Where is the data-prep pipeline?</b></summary>

Out of scope for this repository. The released dataset is a single self-contained JSONL on
Hugging Face. If you want to rebuild from raw Amazon-reviews dumps, see the dataset card.
</details>

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use MemoryCD in your research, please cite:

```bibtex
@article{zhang2026memorycd,
  title={Memorycd: Benchmarking long-context user memory of llm agents for lifelong cross-domain personalization},
  author={Zhang, Weizhi and Wei, Xiaokai and Huang, Wei-Chieh and Hui, Zheng and Wang, Chen and Gong, Michelle and Yu, Philip S},
  journal={arXiv preprint arXiv:2603.25973},
  year={2026}
}
```

## Acknowledgements

The benchmark builds on the public Amazon Reviews dataset and is inspired by prior memory
benchmarks including LaMP and LoCoMo. We thank the authors of those works for releasing
their resources.

---

<div align="center">
  Made with research-grade duct tape. PRs welcome.
</div>
