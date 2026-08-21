"""
Stage 2: Pre-final code
Adds T5-based abstractive summarization, then refines the abstractive
summary with an RL agent (PPO) trained to pick sentences that maximize
ROUGE overlap with a query.

Install dependencies:
    pip install transformers torch PyMuPDF rouge-score gym stable-baselines3 "shimmy>=2.0" gymnasium
"""

import argparse

import fitz
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


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
# RL Environment for Refinement
# -----------------------------
class SummarizationEnv(gym.Env):
    def __init__(self, text, query, reward_function):
        super(SummarizationEnv, self).__init__()
        self.sentences = text.split('. ')
        self.query = query
        self.state = []
        self.reward_function = reward_function
        self.action_space = gym.spaces.Discrete(len(self.sentences))  # Choose a sentence
        self.observation_space = gym.spaces.MultiDiscrete([2] * len(self.sentences))  # Binary state (included/excluded)

    def reset(self):
        self.state = [0] * len(self.sentences)
        return np.array(self.state)

    def step(self, action):
        self.state[action] = 1  # Mark sentence as included
        included_sentences = [self.sentences[i] for i in range(len(self.state)) if self.state[i] == 1]
        summary = ' '.join(included_sentences)
        reward = self.reward_function(summary, self.query)
        done = sum(self.state) == len(self.sentences)
        return np.array(self.state), reward, done, {}


def reward_function(summary, query):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(query, summary)
    reward = scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure
    return reward


def train_rl_agent(env, episodes=50):
    vec_env = make_vec_env(lambda: env, n_envs=1)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=episodes * len(env.sentences))
    return model


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abstractive summarization refined with RL.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file to summarize")
    parser.add_argument("--query", help="Query to guide the refinement")
    args = parser.parse_args()

    pdf_path = args.pdf_path or input("Enter path to PDF file: ").strip()

    # Extract text from PDF
    doc = fitz.open(pdf_path)
    full_text = ' '.join([page.get_text() for page in doc])

    # Step 1: Abstractive Summarization
    print("Performing abstractive summarization...")
    abstractive_summary = abstractive_summarization(full_text)
    print("Abstractive Summary:\n", abstractive_summary)

    # Step 2: RL-based Refinement
    print("\nRefining summary with reinforcement learning...")
    query = args.query or input("Enter the query for refinement: ")

    env = SummarizationEnv(abstractive_summary, query, reward_function)
    rl_model = train_rl_agent(env, episodes=50)
    state = env.reset()

    refined_summary = []
    for _ in range(len(env.sentences)):
        action, _states = rl_model.predict(state, deterministic=True)
        state, reward, done, _ = env.step(action)
        if done:
            break

    refined_summary = ' '.join([env.sentences[i] for i in range(len(state)) if state[i] == 1])
    print("Refined Summary:\n", refined_summary)

    # Step 3: Evaluate with ROUGE
    print("\nEvaluating summary with ROUGE...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(query, refined_summary)
    print("ROUGE Scores:", scores)

    # Save Results
    with open("abstractive_summary.txt", "w") as f:
        f.write(abstractive_summary)
    with open("refined_summary.txt", "w") as f:
        f.write(refined_summary)

    print("Summaries saved as 'abstractive_summary.txt' and 'refined_summary.txt'.")
