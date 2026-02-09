"""
Script 4: Generate Class 3 Paragraphs (AI - Author-Mimicking Style)
Uses Gemini Flash 2.5 Lite to generate 500 paragraphs
"""

import os
import json
import time
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash-lite"
model = genai.GenerativeModel(MODEL_NAME)



def load_topics(topics_file: str = "topics.json") -> List[str]:
    try:
        with open(topics_file, "r", encoding="utf-8") as f:
            return json.load(f)["topics"]
    except FileNotFoundError:
        print(f"ERROR: {topics_file} not found! Run 02_extract_topics.py first.")
        return []


def call_gemini(prompt: str, temperature: float = 1.0) -> str:
    """Generate text using Gemini Flash 2.5 Lite."""
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=400,
            top_p=0.95,
            top_k=40,
        ),
    )

    if response and response.text:
        return response.text.strip()

    return ""


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

DICKENS_TEMPLATE = load_prompt_template("prompts/class_3_dickens.py")
AUSTEN_TEMPLATE  = load_prompt_template("prompts/class_3_austen.py")

def create_dickens_prompt(topic: str) -> str:
    return DICKENS_TEMPLATE.replace("{TOPIC}", topic)


def create_austen_prompt(topic: str) -> str:
    return AUSTEN_TEMPLATE.replace("{TOPIC}", topic)


def generate_class3_paragraphs(
    topics: List[str],
    total_paragraphs: int = 500,
    output_file: str = "class3_ai_styled1.json",
):
    dickens_target = total_paragraphs // 2
    austen_target = total_paragraphs - dickens_target

    authors = [
        ("Charles Dickens", "dickens", dickens_target, create_dickens_prompt),
        ("Jane Austen", "austen", austen_target, create_austen_prompt),
    ]

    all_paragraphs = []
    total_generated = 0

    for author_name, style_code, target_count, prompt_fn in authors:
        print(f"\n{'='*70}")
        print(f"GENERATING {author_name.upper()} STYLE")
        print(f"{'='*70}") 

        per_topic = target_count // len(topics)
        remainder = target_count % len(topics)

        generated_for_author = 0

        for idx, topic in enumerate(topics):
            needed = per_topic + (1 if idx < remainder else 0)
            print(f"\nTopic: {topic} → {needed} paragraphs")

            count = 0
            retries = 0

            while count < needed:
                try:
                    text = call_gemini(prompt_fn(topic), temperature=1.0)

                    if not text or len(text.split()) < 50:
                        retries += 1
                        if retries >= 3:
                            print("  Skipping after retries")
                            break
                        time.sleep(2)
                        continue

                    retries = 0

                    all_paragraphs.append({
                        "text": text,
                        "topic": topic,
                        "word_count": len(text.split()),
                        "author_style": author_name,
                        "style_code": style_code,
                        "class": "ai_styled",
                        "temperature": 1.0,
                        "model": MODEL_NAME,
                    })

                    count += 1
                    generated_for_author += 1
                    total_generated += 1

                    if count % 5 == 0:
                        print(f"  {count}/{needed} generated")

                    time.sleep(1.5)

                except Exception as e:
                    print(f"  Error: {e}")
                    time.sleep(5)

            print(f"  ✓ Completed {count} paragraphs")

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_paragraphs, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Finished {author_name}: {generated_for_author} paragraphs")

    return all_paragraphs



def main():
    print("\nSTEP 4: Generate Class 3 (AI Author-Styled) Paragraphs\n")

    topics = load_topics()
    if not topics:
        return

    print(f"Loaded {len(topics)} topics")

    start = time.time()
    paragraphs = generate_class3_paragraphs(topics)
    elapsed = (time.time() - start) / 60

    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total paragraphs: {len(paragraphs)}")
    print(f"Time elapsed: {elapsed:.1f} minutes")
    print("Saved to: class3_ai_styled.json")

if __name__ == "__main__":
    main()
