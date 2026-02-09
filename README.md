# ghost_in_the_machine

The Ghost in the Machine
Stylometric Detection of AI vs Human Authorship

“Le style, c'est l'homme même” — Georges-Louis Leclerc
(Style is the man himself.)

📌 Project Overview

This project investigates whether machine-generated text can be reliably distinguished from human-authored text using stylometric analysis, classical machine learning, deep learning, and adversarial text evolution.

The pipeline explores:

Statistical authorship fingerprinting

Multi-tier AI detection models

Model interpretability

Genetic algorithm-based adversarial text evolution

🎯 Objectives

Build an authorship-controlled dataset.

Prove that AI and human texts are mathematically distinct.

Train multi-tier detection models.

Explain model decision-making.

Attempt to evolve AI text that bypasses detection.

📚 Task 0 — Dataset Construction
“The Library of Babel”
👩‍💻 Human Corpus

Texts were collected from Project Gutenberg.

Selected Authors
Charles Dickens

David Copperfield

Great Expectations

Hard Times

Oliver Twist

A Tale of Two Cities

Jane Austen

Emma

Northanger Abbey

Persuasion

Pride and Prejudice

Sense and Sensibility

🧹 Data Cleaning

The following preprocessing steps were applied:

Removed Gutenberg headers and footers

Normalized whitespace

Extracted paragraph-level text

Filtered paragraphs between 100–200 words

Final Human Dataset

3328 paragraphs

🧠 Topic Control

To ensure classification relies on style rather than topic, thematic extraction was performed.

Identified Core Themes

Social Class and Economic Inequality

Marriage and Courtship

Moral Character and Virtue

Pursuit of Happiness

Urban Social Transformation

Reputation and Honor

Education and Self-Improvement

Friendship and Human Relationships

🤖 AI Dataset Generation
Class 2 — AI Neutral

Generated using Gemini 2.5 Flash Lite

500 paragraphs

Topic-controlled

No stylistic constraints

🎭 Class 3 — AI Styled

AI was prompted to mimic author-specific styles.

Austen Style Characteristics

Free indirect discourse

Social irony

Third-person narration

Regency vocabulary

Dickens Style Characteristics

Narrative storytelling

Emotional vividness

Mixed sentence structures

Victorian vocabulary

Total:

500 styled AI paragraphs

✍️ Stylistic Differences Between Austen and Dickens
Feature	Austen	Dickens
Narrative Perspective	Third-person	Often First-person
Tone	Ironic, restrained	Emotional, dramatic
Sentence Structure	Balanced, polished	Variable and expressive
Focus	Social psychology	Social realism
Vocabulary	Elegant, subtle	Descriptive, vivid
Characterization	Internal thought driven	Plot-driven storytelling
🔬 Task 1 — Stylometric Fingerprint Analysis

The goal was to prove that the three text classes are mathematically distinguishable.

📊 Lexical Richness Metrics
1. Type Token Ratio (TTR)
Measures vocabulary diversity.

2. Hapax Legomena
Words appearing exactly once in a sample.

Higher hapax usage typically indicates:
Greater lexical spontaneity
Reduced repetitive phrasing

3. Hapax Percentage:
Hapax%=
UniqueWords
HapaxWords×100

🧩 Syntactic Complexity Metrics
POS Adjective-Noun Ratio

Measures descriptive density:
Adj/NounRatio=
Nouns
Adjectives
	
Dependency Tree Depth

Calculated using SpaCy.

Higher values indicate:

Nested grammatical complexity

Longer hierarchical sentence structures

Average Sentence Length

Captures rhythmic variation and structural complexity.

📖 Readability
Flesch-Kincaid Grade Level

Estimates required education level to understand text.

✒️ Punctuation Density

Tracked frequency of:

Commas

Periods

Semicolons

Colons

Exclamation Marks

Question Marks

📈 Mathematical Distinctness Evidence
Key Observations
Feature	Human	AI Neutral	AI Styled
TTR	Lower	Highest	Moderate
Hapax Usage	Lower	Higher	Highest
Syntax Depth	Moderate	Lowest	Highest
Sentence Length	Variable	Short	Long
Readability	Mixed	Simple	Complex

These differences confirm statistically separable class distributions.

🕵️ Task 2 — Multi-Tier AI Detector
Tier A — Statistical Detector
Models Used

Random Forest

XGBoost

Input Features

Stylometric numerical metrics from Task 1.

Results
Model	Accuracy
XGBoost	95.9%
Feature Importance Findings

Most predictive features:

Hapax Percentage

Semicolon Usage

Readability Scores

Lexical Diversity

Tier B — Semantic Detector
Method

GloVe word embeddings

Feedforward neural network

What are GloVe Embeddings?

GloVe learns vector representations of words using global word co-occurrence statistics.

It captures:

Semantic similarity

Contextual relationships

Narrative tone

Results

Accuracy: 99%

This indicates AI struggles to perfectly replicate deeper semantic narrative patterns.

Tier C — Transformer Detector
DistilBERT

A compressed transformer model retaining most of BERT’s language understanding capability while reducing computational cost.

LoRA (Low Rank Adaptation)

Efficient fine-tuning method where:

Only a small fraction of model parameters are trained

Preserves pretrained knowledge

Reduces memory and training cost

Training Summary

Only ~1.09% parameters trained

GPU acceleration used

Results

Accuracy: 100%

This suggests transformers capture:

Sentence rhythm

Contextual coherence

Narrative voice

🔍 Task 3 — Explainability

Due to near-perfect classification performance, interpretability was theoretically analyzed.

Models likely detect AI-specific linguistic signals such as:

Over-structured phrasing

Excess lexical novelty

Uniform narrative rhythm

🧬 Task 4 — Genetic Algorithm Adversarial Attack
“The Turing Test”
Objective

Attempt to evolve AI text until the detector classifies it as Human (>90%).

Genetic Algorithm Workflow
Initial Population

10 AI-generated paragraphs

Fitness Function

Human classification probability from Tier C model.

Selection

Top 3 highest-scoring paragraphs retained.

Mutation Strategies
Rhythm Mutation

Alters sentence flow and pacing.

Vocabulary Mutation

Replaces formal vocabulary with natural alternatives.

Inconsistency Injection

Introduces subtle human-like imperfections.

Archaic Vocabulary

Adds rare but natural lexical choices.

Punctuation Variation

Introduces expressive punctuation diversity.

Structural Complexity

Adds subordinate clauses and narrative detail.

Evolution Results

Generations executed: 10

Best Human Score Achieved: 0.51

Target Score: 0.90

Interpretation

The plateau suggests:

The detector captures deep linguistic signals

Simple stylistic perturbations are insufficient to bypass detection

📊 Key Findings

Human and AI text are mathematically separable.

Statistical stylometry alone achieves high accuracy.

Semantic embeddings dramatically improve detection.

Transformer-based detectors nearly eliminate classification errors.

Adversarial text evolution remains challenging.

🛠 Technology Stack

Python

SpaCy

Scikit-Learn

XGBoost

PyTorch

HuggingFace Transformers

LoRA (PEFT)

Gemini API

Matplotlib / Seaborn
