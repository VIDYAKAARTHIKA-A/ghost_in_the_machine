"""
Task 1: The Fingerprint Analysis (PARAGRAPH LEVEL VERSION)
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import spacy
from textstat import flesch_kincaid_grade


# ------------------ LOGGER ------------------

class OutputLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ------------------ SPACY ------------------

print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ------------------ TEXT FINGERPRINT ------------------

class TextFingerprint:

    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.words = self._tokenize_words()
        self.sentences = self._tokenize_sentences()

    def _tokenize_words(self):
        return re.findall(r'\b[a-z]+\b', self.text.lower())

    def _tokenize_sentences(self):
        sentences = re.split(r'[.!?]+', self.text)
        return [s.strip() for s in sentences if s.strip()]

    def type_token_ratio(self, sample_size=5000):
        sample = self.words[:sample_size]
        if not sample:
            return 0
        return len(set(sample)) / len(sample)

    def hapax_legomena_count(self, sample_size=5000):
        sample = self.words[:sample_size]
        counts = Counter(sample)
        return sum(1 for c in counts.values() if c == 1)

    def hapax_percentage(self, sample_size=5000):
        sample = self.words[:sample_size]
        if not sample:
            return 0
        counts = Counter(sample)
        hapax = sum(1 for c in counts.values() if c == 1)
        return hapax / len(set(sample)) * 100

    def pos_distribution(self):
        doc = nlp(self.text[:100000])
        pos_counts = Counter([t.pos_ for t in doc])
        total = sum(pos_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in pos_counts.items()}

    def adjective_to_noun_ratio(self):
        pos = self.pos_distribution()
        return pos.get("ADJ", 0) / max(pos.get("NOUN", 1e-6), 1e-6)

    def dependency_tree_depth(self):
        doc = nlp(self.text[:100000])
        depths = []

        for sent in doc.sents:
            for token in sent:
                depth = 0
                head = token
                while head.head != head:
                    depth += 1
                    head = head.head
                depths.append(depth)

        return np.mean(depths) if depths else 0

    def average_sentence_length(self):
        if not self.sentences:
            return 0
        return len(self.words) / len(self.sentences)

    def punctuation_density(self):
        marks = {',': 'comma', '.': 'period', ';': 'semicolon', ':': 'colon',
                 '!': 'exclamation', '?': 'question'}

        total = len(self.text)
        densities = {}

        for m, name in marks.items():
            densities[f'punct_{name}'] = self.text.count(m) / max(total, 1) * 1000

        return densities

    def flesch_kincaid(self):
        try:
            return flesch_kincaid_grade(self.text)
        except:
            return None

    def analyze_all(self):

        res = {
            "label": self.label,
            "ttr": self.type_token_ratio(),
            "hapax_count": self.hapax_legomena_count(),
            "hapax_percentage": self.hapax_percentage(),
            "adj_noun_ratio": self.adjective_to_noun_ratio(),
            "dep_tree_depth": self.dependency_tree_depth(),
            "avg_sentence_length": self.average_sentence_length(),
            "flesch_kincaid": self.flesch_kincaid(),
        }

        res.update(self.punctuation_density())
        return res

# ------------------ JSON PARAGRAPH LOADER ------------------

def load_json_paragraphs(filepath):

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    paragraphs = []

    for item in data:
        if isinstance(item, dict) and "text" in item:
            paragraphs.append(item)

    return paragraphs


# ------------------ PARAGRAPH ANALYSIS ------------------

def analyze_class(filepath, label):

    items = load_json_paragraphs(filepath)
    results = []

    print(f"Analyzing {label} ({len(items)} paragraphs)")

    for idx, item in enumerate(items):

        text = item["text"]
        fp = TextFingerprint(text, label)

        res = fp.analyze_all()

        # ---- Metadata ----
        res["paragraph_id"] = idx
        res["author"] = item.get("author")
        res["book"] = item.get("book")
        res["class"] = item.get("class")
        res["word_count"] = item.get("word_count")

        results.append(res)

    return results


# ------------------ MAIN ANALYSIS ------------------

def analyze_three_classes(class1_path, class2_path, class3_path):

    results = []

    results += analyze_class(class1_path, "Human")
    results += analyze_class(class2_path, "AI Neutral")
    results += analyze_class(class3_path, "AI Styled")

    df = pd.DataFrame(results)

    return df


# ------------------ VISUALIZATION ------------------

def create_boxplots(df, output_dir):

    numeric_cols = ["ttr", "hapax_percentage", "adj_noun_ratio",
                    "dep_tree_depth", "avg_sentence_length", "flesch_kincaid"]

    for col in numeric_cols:
        plt.figure()
        sns.boxplot(data=df, x="label", y=col)
        plt.title(col)
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_boxplot1.png")
        plt.close()


# ------------------ MAIN ------------------

if __name__ == "__main__":

    output_dir = Path.cwd()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"log_{timestamp}.txt"

    logger = OutputLogger(log_file)
    sys.stdout = logger

    class1_path = r"D:\precog_task\class1_human_paragraphs.json"
    class2_path = r"D:\precog_task\class2_ai_neutral1.json"
    class3_path = r"D:\precog_task\class3_ai_styled1.json"

    try:

        df = analyze_three_classes(class1_path, class2_path, class3_path)

        print("\nSummary (Class Mean Values):")
        print(df.groupby("label").mean(numeric_only=True))

        df.to_csv(output_dir / "paragraph_fingerprint_results1.csv", index=False)

        create_boxplots(df, output_dir)

        print("\nFiles Generated:")
        print("- paragraph_fingerprint_results1.csv")
        print("- metric boxplots")

    finally:
        sys.stdout = logger.terminal
        logger.close()

        print(f"Logs saved to {log_file}")
