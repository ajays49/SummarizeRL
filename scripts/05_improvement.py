"""
Stage 5: Improvement
Cleans boilerplate sections (references, conclusion, etc.) out of each
page before summarizing, builds a reference summary from the full
document, then runs several summarization passes with a varying
max_length, keeping the pass with the best ROUGE-L score against the
reference summary.

Install dependencies:
    pip install transformers torch PyMuPDF rouge matplotlib
"""

import argparse
import re
import time

import matplotlib.pyplot as plt
import torch
import fitz
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge import Rouge


def get_device():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    print("CUDA is not available. Using CPU.")
    return torch.device("cpu")


# Define the abstractive summarization function
def abstractive_summarization(text, model, tokenizer, max_length=150):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    summary_ids = model.generate(input_ids, max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True).to(model.device)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# Define the reward function using ROUGE score
def calculate_rouge_score(reference_summary, generated_summary):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(generated_summary, reference_summary)[0]
        rouge_l_score = scores['rouge-l']['f']
    except ValueError:
        rouge_l_score = 0.0
    return rouge_l_score


def clean_text(text):
    # Remove specific section titles and related content more aggressively
    exclude_patterns = [
        r"(literature\s*review.*)",
        r"(references.*)",
        r"(future\s*works.*)",
        r"(conclusion.*)",
        r"(acknowledgments.*)",
        r"(related\s*work.*)",
        r"(introduction.*)"  # Exclude introduction
    ]
    for pattern in exclude_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text


# Define the RL-based summarization function
def rl_based_summarization(page_texts, model, tokenizer, reference_summary, iterations=5):
    rouge_scores = []
    times = []
    summaries = []
    best_summary = None
    best_rouge_score = 0.0

    for i in range(iterations):
        start_time = time.time()
        generated_summary = ""
        for page_text in page_texts:
            # Clean the page text
            cleaned_page_text = clean_text(page_text)

            # Split the cleaned page text into sentences
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', cleaned_page_text)

            # Ensure at least 2 sentences are extracted if available.
            num_sentences_to_extract = min(2, len(sentences))
            extracted_sentences = " ".join(sentences[:num_sentences_to_extract]) if sentences else ""

            # Abstractively summarize the extracted sentences
            max_length = 120 + i * 15  # Vary the max_length for each iteration
            page_summary = abstractive_summarization(extracted_sentences, model, tokenizer, max_length=max_length)
            generated_summary += page_summary + " "

        rouge_score = calculate_rouge_score(reference_summary, generated_summary)
        end_time = time.time()
        time_taken = end_time - start_time

        rouge_scores.append(rouge_score)
        times.append(time_taken)
        summaries.append(generated_summary)

        print(f"Iteration {i + 1}: ROUGE-L Score = {rouge_score:.4f}, Time = {time_taken:.2f} seconds")

        if rouge_score > best_rouge_score:
            best_rouge_score = rouge_score
            best_summary = generated_summary

    # Print the best iteration
    best_iteration = rouge_scores.index(max(rouge_scores)) + 1
    print(f"Best Iteration: {best_iteration} with ROUGE-L Score = {max(rouge_scores):.4f}")

    # Plotting iteration vs ROUGE score
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, iterations + 1), rouge_scores, marker='o')
    plt.title('Iteration vs ROUGE Score')
    plt.xlabel('Iteration')
    plt.ylabel('ROUGE Score')
    plt.xticks(range(1, iterations + 1))
    plt.grid(True)
    plt.show()

    # Plotting iteration vs time
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, iterations + 1), times, marker='o')
    plt.title('Iteration vs Time')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(1, iterations + 1))
    plt.grid(True)
    plt.show()

    return best_summary


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page_texts = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_texts.append(page.get_text())
    return page_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Section-cleaned, multi-iteration abstractive summarization.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file to summarize")
    args = parser.parse_args()

    pdf_path = args.pdf_path or input("Enter path to PDF file: ").strip()

    device = get_device()

    # Load the T5 model and tokenizer
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model.eval()

    # Extract text from the PDF
    page_texts = extract_text_from_pdf(pdf_path)

    # Generate a reference summary based on the entire document text
    full_document_text = " ".join(page_texts)
    reference_summary = abstractive_summarization(full_document_text, model, tokenizer)

    # Run RL-based summarization
    best_summary = rl_based_summarization(page_texts, model, tokenizer, reference_summary)

    # Save the best summary to a text file
    with open("best_summary.txt", "w") as f:
        f.write(best_summary)

    print("Best summary saved as 'best_summary.txt'.")
