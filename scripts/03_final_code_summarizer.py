"""
Stage 3: Final code summarizer
Per-page T5 abstractive summarization, ROUGE evaluation against the
source document, and a simplified single-step RL environment (keep /
modify / remove) with reward curves plotted over training episodes.

Install dependencies:
    pip install transformers torch PyMuPDF stable-baselines3 gymnasium shimmy rouge-score matplotlib
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
from stable_baselines3 import PPO
from gymnasium import Env, spaces


# -----------------------------
# Abstractive Summarization
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
# Extract Text from Each Page
# -----------------------------
def extract_page_texts(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]


# -----------------------------
# ROUGE Score Evaluation
# -----------------------------
def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores


# -----------------------------
# Reinforcement Learning for Summary Refinement (No Query Needed)
# -----------------------------
class SummarizationEnv(Env):
    def __init__(self, summary):
        super(SummarizationEnv, self).__init__()
        self.summary = summary
        self.reward = 0
        self.action_space = spaces.Discrete(3)  # Actions: Keep, Modify, Remove
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(summary.split()),), dtype=np.float32)

    def step(self, action):
        if action == 0:  # Keep
            self.reward += 1
        elif action == 1:  # Modify (simulate improvement)
            self.reward += 2
        elif action == 2:  # Remove
            self.reward -= 1

        done = True  # Single-step environment
        return np.array([0] * len(self.summary.split()), dtype=np.float32), self.reward, done, False, {}

    def reset(self, seed=None, options=None):
        return np.array([0] * len(self.summary.split()), dtype=np.float32), {}


def train_rl_agent(env, episodes=50):
    model = PPO("MlpPolicy", env, verbose=0)
    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

    return model, rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-page abstractive summarization with RL refinement.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file to summarize")
    args = parser.parse_args()

    pdf_path = args.pdf_path or input("Enter path to PDF file: ").strip()
    page_texts = extract_page_texts(pdf_path)

    # -----------------------------
    # Summarize Each Page
    # -----------------------------
    start_time = time.time()
    page_summaries = [abstractive_summarization(page) for page in page_texts]
    full_summary = " ".join(page_summaries)
    end_time = time.time()
    print(f"Summarization Time: {end_time - start_time:.2f} seconds")

    # Save the summary
    with open("summary.txt", "w") as f:
        f.write(full_summary)
    print("Generated summary saved as 'summary.txt'.")

    # Compute ROUGE Score
    reference_text = " ".join(page_texts)
    rouge_scores = compute_rouge(reference_text, full_summary)

    print("\nROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1={score.fmeasure:.4f}")

    # Train RL model
    env = SummarizationEnv(full_summary)
    rl_model, rewards = train_rl_agent(env, episodes=50)

    # -----------------------------
    # Plot Training Rewards
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward per Episode", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Reinforcement Learning Training Progress")
    plt.legend()
    plt.grid()
    plt.show()
