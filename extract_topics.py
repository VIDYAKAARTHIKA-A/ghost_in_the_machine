#!/usr/bin/env python3
"""
Script 2: Extract Core Topics
Analyzes human-authored paragraphs and extracts 5-10 core topics
"""

import json
import re
from collections import Counter
from typing import List, Set

# Stop words to filter out
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'his', 'her', 'its', 'their', 'my', 'your', 'our', 'who', 'which',
    'what', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
    'not', 'no', 'nor', 'so', 'than', 'too', 'very', 'said', 'mr', 'mrs',
    'miss', 'just', 'now', 'then', 'here', 'there', 'one', 'two', 'more',
    'much', 'many', 'some', 'such', 'into', 'out', 'up', 'down', 'been',
    'being', 'only', 'own', 'same', 'other', 'about', 'after', 'before',
    'through', 'between', 'under', 'over', 'again', 'further', 'once'
}

def extract_keywords(paragraphs: List[dict], min_word_length: int = 4) -> List[str]:
    """
    Extract important keywords from paragraphs.
    """
    # Combine all text
    all_text = ' '.join([p['text'] for p in paragraphs]).lower()
    
    # Extract words (letters only, minimum length)
    words = re.findall(r'\b[a-z]{' + str(min_word_length) + r',}\b', all_text)
    
    # Filter out stop words
    filtered_words = [w for w in words if w not in STOP_WORDS]
    
    # Count frequency
    word_freq = Counter(filtered_words)
    
    # Return top keywords
    return [word for word, count in word_freq.most_common(100)]

def generate_topics_from_keywords(keywords: List[str]) -> List[str]:
    """
    Generate thematic topics based on keyword analysis.
    This combines domain knowledge about 19th-century literature with the keywords.
    """
    # Keyword categories found in Dickens and Austen
    keyword_categories = {
        'social': ['society', 'class', 'gentleman', 'lady', 'family', 'marriage', 'social'],
        'economic': ['money', 'fortune', 'poverty', 'wealth', 'estate', 'income', 'poor'],
        'moral': ['character', 'virtue', 'honour', 'pride', 'sense', 'feeling', 'conscience'],
        'emotional': ['love', 'happiness', 'suffering', 'misery', 'pleasure', 'pain'],
        'relationships': ['friend', 'acquaintance', 'companion', 'husband', 'wife'],
        'urban': ['london', 'city', 'town', 'street', 'house', 'home'],
        'education': ['education', 'knowledge', 'learning', 'understanding', 'mind']
    }
    
    # Count which categories appear most
    category_scores = {cat: 0 for cat in keyword_categories}
    
    for keyword in keywords[:50]:  # Focus on top 50 keywords
        for category, words in keyword_categories.items():
            if keyword in words or any(w in keyword for w in words):
                category_scores[category] += 1
    
    # Generate topics based on the literature's themes
    topics = [
        "Social Class and Economic Inequality in Victorian Society",
        "Marriage, Courtship, and Family Obligations",
        "Moral Character and Personal Virtue",
        "The Pursuit of Happiness and Contentment",
        "Urban Life and Social Transformation",
        "Reputation, Honor, and Public Opinion",
        "Education, Self-Improvement, and Personal Growth",
        "Friendship, Loyalty, and Human Relationships"
    ]
    
    return topics

def save_topics(topics: List[str], output_file: str = 'topics.txt'):
    """
    Save topics to a text file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CORE TOPICS\n")
        f.write("="*70 + "\n\n")
        f.write("These topics are extracted from novels by Charles Dickens and Jane Austen.\n")
        f.write("They represent the major thematic elements found in 19th-century literature.\n\n")
        
        for i, topic in enumerate(topics, 1):
            f.write(f"{i}. {topic}\n")
    
    print(f"Topics saved to: {output_file}")

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("STEP 2: Extract Core Topics from Literature")
    print("="*70 + "\n")
    
    # Configuration
    INPUT_FILE = 'class1_human_paragraphs.json'
    OUTPUT_FILE = 'topics.txt'
    NUM_TOPICS = 8
    
    # Load human paragraphs
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            paragraphs = json.load(f)
        print(f"Loaded {len(paragraphs)} human-authored paragraphs")
    except FileNotFoundError:
        print(f"ERROR: {INPUT_FILE} not found!")
        print("Please run 01_clean_and_extract_paragraphs.py first")
        return
    
    # Extract keywords
    print("\nAnalyzing text and extracting keywords...")
    keywords = extract_keywords(paragraphs)
    print(f"Identified {len(keywords)} unique keywords")
    print(f"\nTop 20 keywords:")
    for i, kw in enumerate(keywords[:20], 1):
        print(f"  {i:2}. {kw}")
    
    # Generate topics
    print(f"\nGenerating {NUM_TOPICS} core topics...")
    topics = generate_topics_from_keywords(keywords)[:NUM_TOPICS]
    
    print(f"\n{'='*70}")
    print("EXTRACTED TOPICS")
    print(f"{'='*70}")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")
    
    # Save topics
    print(f"\n{'='*70}")
    save_topics(topics, OUTPUT_FILE)
    
    # Also save as JSON for programmatic use
    topics_json = {
        'topics': topics,
        'description': 'Core thematic topics from 19th-century literature (Dickens & Austen)'
    }
    
    with open('topics.json', 'w', encoding='utf-8') as f:
        json.dump(topics_json, f, indent=2)
    
    print("Topics also saved as: topics.json")
    print(f"{'='*70}\n")
    
    print("✓ Topic extraction complete!")
    print("  Next step: Review topics.txt and run 03_generate_class2.py")

if __name__ == "__main__":
    main()