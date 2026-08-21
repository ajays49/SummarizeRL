"""
Stage 1: Initial work
Query-based extractive summarization using BM25, plus author/formula
extraction from a PDF, and a first draft of an RL environment for
sentence selection (defined here but not yet trained in this stage).

Install dependencies:
    pip install PyMuPDF rank_bm25 sentence-transformers rouge-score transformers gym stable-baselines3 "shimmy>=2.0" gymnasium
"""

import argparse
import re

import fitz
import numpy as np
import gym
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer


# -----------------------------
# Section 1: Text Extraction
# -----------------------------
def extract_page_texts(pdf_path):
    """
    Extracts text from each page of the PDF.
    """
    doc = fitz.open(pdf_path)
    page_texts = [page.get_text() for page in doc]
    return page_texts


def extract_authors(page_texts):
    """
    Extracts author names, assuming they appear under the title of the paper.
    """
    # Assume the first page contains the title and author names
    first_page = page_texts[0]

    # Split by lines and find potential author names
    lines = first_page.split("\n")
    for i, line in enumerate(lines):
        # Titles are often in all caps, authors typically follow
        if line.isupper() and i + 1 < len(lines):
            # Author names are likely in the next line(s) (heuristic)
            potential_authors = lines[i + 1]
            # Filter out email or affiliation-like entries
            author_names = re.split(r',| and ', potential_authors)
            return [name.strip() for name in author_names if '@' not in name]
    return ["Authors not found"]


# -----------------------------
# Section 2: Query-based Summarization with BM25
# -----------------------------
def query_based_summarization_bm25(text, query, top_n=5):
    """
    Generates a query-based summary using BM25 ranking.
    """
    sentences = text.split('. ')
    tokenized_sentences = [sentence.split() for sentence in sentences]

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_sentences)

    # Get query results
    query_tokens = query.split()
    ranked_sentences = bm25.get_top_n(query_tokens, sentences, n=top_n)

    return ' '.join(ranked_sentences)


# -----------------------------
# Section 3: Formula Extraction
# -----------------------------
def extract_formulas(page_texts):
    """
    Extracts mathematical formulas from the text.
    """
    formulas = []
    formula_pattern = r'(\$.*?\$|\\\[(.*?)\\\]|\\\((.*?)\\\))'
    for page in page_texts:
        matches = re.findall(formula_pattern, page)
        for match in matches:
            # Combine LaTeX formula patterns
            formulas.append(match[0])
    return formulas if formulas else ["No formulas found"]


# -----------------------------
# Section 4: Reinforcement Learning Environment
# -----------------------------
class SummarizationEnv(gym.Env):
    """
    Custom RL environment for summarization.
    """

    def __init__(self, text, query, reward_function):
        super(SummarizationEnv, self).__init__()
        self.sentences = text.split('. ')
        self.query = query
        self.state = []
        self.reward_function = reward_function
        self.action_space = gym.spaces.Discrete(len(self.sentences))  # Choose a sentence
        self.observation_space = gym.spaces.MultiDiscrete([2] * len(self.sentences))  # Binary state (included/excluded)

    def reset(self):
        """
        Resets the environment.
        """
        self.state = [0] * len(self.sentences)
        return np.array(self.state)

    def step(self, action):
        """
        Performs an action (include/exclude a sentence).
        """
        self.state[action] = 1  # Mark sentence as included
        included_sentences = [self.sentences[i] for i in range(len(self.state)) if self.state[i] == 1]
        summary = ' '.join(included_sentences)

        # Compute reward
        reward = self.reward_function(summary, self.query)

        # Check if all sentences are processed
        done = sum(self.state) == len(self.sentences)
        return np.array(self.state), reward, done, {}


def reward_function(summary, query):
    """
    Computes a reward based on ROUGE scores for a summary.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(query, summary)
    reward = scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure
    return reward


# -----------------------------
# Section 5: RL Agent Training
# -----------------------------
def train_rl_agent(env, episodes=100):
    """
    Trains an RL agent on the summarization environment.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    # Create a vectorized environment
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Initialize PPO agent
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # Train the model
    model.learn(total_timesteps=episodes * len(env.sentences))
    return model


# -----------------------------
# Section 6: Main Execution
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query-based BM25 summarization of a PDF.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file to summarize")
    parser.add_argument("--query", help="Query to guide the summary")
    args = parser.parse_args()

    pdf_path = args.pdf_path or input("Enter path to PDF file: ").strip()
    query = args.query or input("Enter the query for summarization: ")

    # Extract text from the PDF
    page_texts = extract_page_texts(pdf_path)
    full_text = ' '.join(page_texts)

    # Extract authors
    authors = extract_authors(page_texts)
    print("Authors:", authors)

    # Extract formulas
    formulas = extract_formulas(page_texts)
    print("Formulas:", formulas)

    # Hybrid Summarization
    print("Generating query-based summary...")
    summary = query_based_summarization_bm25(full_text, query)
    print("Query-based Summary:", summary)

    # Save the summary
    with open("summary.txt", "w") as f:
        f.write(summary)
    print("Summary saved as 'summary.txt'.")
