"""
Stage 4: Reworked
Iterative abstractive summarization + RL "editing" loop (shorten /
expand / reword) run over 5 epochs, tracking the ROUGE-1 score of
each iteration and keeping the best one. Plots ROUGE vs. iteration
and a simulated accuracy/loss curve.

Install dependencies:
    pip install transformers torch PyMuPDF rouge-score stable-baselines3 matplotlib gym
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env, spaces
import fitz


# -----------------------------
# Function: Abstractive Summarization (T5)
# -----------------------------
def abstractive_summarization(text):
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


# -----------------------------
# Function: Extract Text from Each Page
# -----------------------------
def extract_page_texts(pdf_path):
    doc = fitz.open(pdf_path)
    page_texts = [page.get_text() for page in doc]
    return page_texts


# -----------------------------
# Function: Compute ROUGE Scores
# -----------------------------
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }


# -----------------------------
# Reinforcement Learning Environment
# -----------------------------
class SummarizationEnv(Env):
    def __init__(self, initial_summary):
        super(SummarizationEnv, self).__init__()
        self.initial_summary = initial_summary
        self.current_summary = initial_summary
        self.action_space = spaces.Discrete(3)  # Actions: [Shorten, Expand, Reword]
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.current_summary = self.current_summary[:int(len(self.current_summary) * 0.9)]
        elif action == 1:
            self.current_summary += " Additional details."
        elif action == 2:
            self.current_summary = " ".join(self.current_summary.split()[::-1])  # Reverse words (dummy rewording)

        reward = compute_rouge(self.initial_summary, self.current_summary)['rouge1']
        return np.array([reward]), reward, False, {}

    def reset(self):
        self.current_summary = self.initial_summary
        return np.array([0.0])


# -----------------------------
# Function: Train RL Model
# -----------------------------
def train_rl_agent(env, episodes=50):
    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=episodes)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative abstractive summarization refined over 5 RL epochs.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file to summarize")
    args = parser.parse_args()

    pdf_path = args.pdf_path or input("Enter path to PDF file: ").strip()

    # Extract text from PDF
    page_texts = extract_page_texts(pdf_path)
    full_text = " ".join(page_texts)

    # -----------------------------
    # Abstractive Summarization
    # -----------------------------
    print("\nPerforming Abstractive Summarization...")
    start_time = time.time()
    initial_summary = abstractive_summarization(full_text)
    end_time = time.time()
    print(f"Initial Summary Generated in {end_time - start_time:.2f} seconds.")

    # -----------------------------
    # Reinforcement Learning Iterations
    # -----------------------------
    rouge_scores_per_iteration = []
    env = SummarizationEnv(initial_summary)

    print("\nRefining summary with reinforcement learning over 5 iterations...")
    best_summary = initial_summary
    best_rouge = 0

    for i in range(5):
        print(f"\nEpoch {i+1}: Training RL model...")
        rl_model = train_rl_agent(env, episodes=50)

        state = env.reset()
        action = env.action_space.sample()  # Sample an action
        new_state, reward, _, _ = env.step(action)

        refined_summary = env.current_summary
        rouge_scores = compute_rouge(initial_summary, refined_summary)

        print(f"ROUGE Scores for Iteration {i+1}: {rouge_scores}")
        rouge_scores_per_iteration.append(rouge_scores['rouge1'])

        if rouge_scores['rouge1'] > best_rouge:
            best_rouge = rouge_scores['rouge1']
            best_summary = refined_summary

    # -----------------------------
    # Declare Best Iteration
    # -----------------------------
    best_iteration = np.argmax(rouge_scores_per_iteration) + 1
    print(f"\nBest Iteration: {best_iteration} with ROUGE-1 Score: {best_rouge:.4f}")
    print("Final Refined Summary:\n", best_summary)

    # -----------------------------
    # Save Outputs
    # -----------------------------
    with open("final_summary.txt", "w") as f:
        f.write(best_summary)

    # -----------------------------
    # Plot Graphs
    # -----------------------------
    plt.figure(figsize=(12, 5))

    # ROUGE Score per Iteration
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 6), rouge_scores_per_iteration, marker='o', color='b', label="ROUGE-1 Score")
    plt.xlabel("Iteration")
    plt.ylabel("ROUGE Score")
    plt.title("ROUGE Score vs. Iteration")
    plt.legend()
    plt.grid(True)

    # Accuracy vs. Loss (Simulated)
    epochs = np.arange(1, 6)
    accuracy = np.array(rouge_scores_per_iteration) * 100  # Simulated accuracy
    loss = 100 - accuracy  # Simulated loss

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, marker='o', color='g', label="Accuracy")
    plt.plot(epochs, loss, marker='s', color='r', label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage")
    plt.title("Learning Accuracy vs. Loss")
    plt.legend()
    plt.grid(True)

    plt.show()
