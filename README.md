# Abstractive + Reinforcement Learning PDF Summarizer

This project summarizes research papers (PDFs) by combining **abstractive summarization** (a T5 transformer model) with a **reinforcement learning** refinement step, and evaluates the resulting summaries with **ROUGE** scores.

Given a PDF, the pipeline:
1. Extracts text page-by-page with PyMuPDF (`fitz`).
2. Generates an abstractive summary of the text using a T5 (`t5-small`) model from Hugging Face `transformers`.
3. Refines/selects sentences with a reinforcement learning agent (Stable-Baselines3 PPO) whose reward is based on ROUGE overlap.
4. Scores the final summary against the source text (or a query) using ROUGE-1 / ROUGE-2 / ROUGE-L.

## Two ways to use this repo

- **Run the notebook**  [`project2.ipynb`](project2.ipynb) has everything in one place, in the order it was developed, with real outputs and plots already saved. This is the easiest way to see the project working end-to-end. Open it in Google Colab or Jupyter, run the cells top to bottom, and upload a PDF when prompted.
- **Run the Python scripts**  the same code, split into standalone `.py` files under [`scripts/`](scripts/), for anyone who'd rather read or run it outside a notebook (e.g. straight from GitHub, or in a local Python environment/IDE).

Both are kept in sync with each other, pick whichever is more convenient for you.

## Project history (why there are 7 scripts)

This started as an exploratory notebook, and each script in `scripts/` is a real stage of that evolution rather than a single "final" pipeline, the notebook's own section headers name each stage:

| Script | Notebook stage | What it does |
|---|---|---|
| [`scripts/01_initial_work.py`](scripts/01_initial_work.py) | *Initial work* | Extractive, query-based summarization with BM25, plus author/formula extraction, and a first draft RL environment. |
| [`scripts/02_pre_final_code.py`](scripts/02_pre_final_code.py) | *pre-final code* | Adds T5 abstractive summarization, then refines it with a trained RL (PPO) agent scored against a query. |
| [`scripts/03_final_code_summarizer.py`](scripts/03_final_code_summarizer.py) | *final code summarizer* | Per-page abstractive summarization, ROUGE evaluation vs. the source document, and a simplified keep/modify/remove RL loop with training-reward plots. |
| [`scripts/04_reworked.py`](scripts/04_reworked.py) | *reworked* | Iterative shorten/expand/reword RL loop over 5 epochs, keeping the best-scoring iteration, with ROUGE and accuracy/loss plots. |
| [`scripts/05_improvement.py`](scripts/05_improvement.py) | *improvement* | Strips boilerplate sections (references, conclusion, etc.) before summarizing, and sweeps `max_length` across 5 iterations to find the best ROUGE-L score. |
| [`scripts/06_more_refined.py`](scripts/06_more_refined.py) | *more more refined* | Same as above with a slightly larger sentence window and added randomness in `max_length` and the reported ROUGE-L score. |
| [`scripts/07_chatgpt_comparison.py`](scripts/07_chatgpt_comparison.py) | *chatgpt comparision* | Scores a hand-written summary against a source document with ROUGE-L, for comparing against another model's output. |

If you just want to see the project work, start with `04_reworked.py` or `06_more_refined.py` those are the most complete end-to-end pipelines. The earlier scripts are kept for reference and to show how the approach evolved.

## Running the scripts

The notebook cells relied on Google Colab's file upload widget (`google.colab.files.upload()`), which doesn't exist outside Colab. The scripts in `scripts/` replace that with a plain command-line argument (or a prompt if you don't pass one), so they run in any local Python environment:

```bash
pip install -r requirements.txt
python scripts/06_more_refined.py path/to/paper.pdf
```

or, without an argument, it will just ask for the path:

```bash
python scripts/06_more_refined.py
# Enter path to PDF file: path/to/paper.pdf
```

Each script has its own docstring at the top listing exactly which packages it needs (they differ slightly stage to stage, e.g. some use the older `gym` library, later ones use `gymnasium`, and `05`/`06`/`07` use `rouge`/`rouge-score` respectively). `requirements.txt` at the repo root installs what you need for the recommended stage (`06_more_refined.py`).

## Repo contents

- [`project2.ipynb`](project2.ipynb) — the original notebook, unchanged, with saved outputs.
- [`scripts/`](scripts/) — the notebook's code cells as standalone, locally-runnable Python scripts (see table above).
- [`requirements.txt`](requirements.txt) — dependencies for the recommended script.
- [`data.xlsx`](data.xlsx) — sample/supporting data used during development.
- [`best_summary (12).txt`](<best_summary (12).txt>) — an example summary produced by the pipeline.

## Evaluation

Summary quality is measured with [ROUGE](https://pypi.org/project/rouge-score/) (ROUGE-1, ROUGE-2, ROUGE-L), comparing generated summaries against the source document text (or, in the earliest stage, against a query).
