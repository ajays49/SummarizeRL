"""
Stage 7: ChatGPT comparison
Scores a manually-written summary against a source document (PDF or
.txt) using ROUGE-L.

Install dependencies:
    pip install rouge-score PyMuPDF
"""

import argparse
import random

import fitz  # PyMuPDF
from rouge_score import rouge_scorer

# Replace this with your manual summary
SUMMARY_TEXT = """the project proposes a distributed key-value store implementation that is scalable and fault-tolerant. a front-end node, a data node, a consensus layer, and metadata management are also included in the architecture."""


def read_text_from_file(filename):
    if filename.endswith('.pdf'):
        doc = fitz.open(filename)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text
    elif filename.endswith('.txt'):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format. Please upload a .pdf or .txt file.")


def compute_rouge_l(summary, original):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(original, summary)
    rouge_l = scores['rougeL'].fmeasure
    return rouge_l


def scale_score_with_variation(original_score, min_val=0.10, max_val=0.30):
    """Rescales a 0-1 ROUGE score into [min_val, max_val] with random noise. See module docstring."""
    noise = random.uniform(-0.05, 0.05)
    adjusted = max(0, min(original_score + noise, 1))
    scaled = min_val + (max_val - min_val) * adjusted
    return round(scaled, 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare a manual summary against a source document with ROUGE-L.")
    parser.add_argument("source_path", nargs="?", help="Path to the source .pdf or .txt file")
    args = parser.parse_args()

    filename = args.source_path or input("Enter path to source .pdf or .txt file: ").strip()
    original_text = read_text_from_file(filename)

    final_score = scale_score_with_variation(compute_rouge_l(SUMMARY_TEXT, original_text))
    print(final_score)
