#!/usr/bin/env python3
"""
Script 3: Generate Class 2 Paragraphs (AI - Neutral Style)
Uses Gemini API with batch processing to generate 500 paragraphs in neutral, informative style
"""

import os
import json
import time
from typing import List, Dict
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

def load_topics(topics_file: str = 'topics.json') -> List[str]:
    """Load topics from JSON file."""
    try:
        with open(topics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['topics']
    except FileNotFoundError:
        print(f"ERROR: {topics_file} not found!")
        print("Please run 02_extract_topics.py first")
        return []

def create_neutral_prompt(topic: str) -> str:
    """
    Create a prompt for neutral-style generation.
    Based on prompt_template_class2.txt
    """
    prompt = f"""Write a narrative paragraph about {topic} in natural, contemporary storytelling style.

Requirements:
- 100-150 words
- Tell a story, don't analyze or explain concepts
- Use modern, conversational narrative voice
- Focus on characters, events, and descriptions
- Write as if continuing a novel, not writing an essay
- Use simple, direct language
- First or third person perspective (choose what fits)

NO academic language. NO abstract concepts. NO analytical tone.
Just tell a story about the topic.

Output only the paragraph, no preamble or title.

"""
    
    return prompt

def generate_single_paragraph(model, topic: str, batch_id: int, para_id: int, 
                              max_retries: int = 3) -> Dict:
    """
    Generate a single paragraph with retries.
    Returns the paragraph dict or None if failed.
    """
    for attempt in range(max_retries):
        try:
            # Create prompt
            prompt = create_neutral_prompt(topic)
            
            # Generate content
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    max_output_tokens=350,
                    top_p=0.95,
                    top_k=40,
                )
            )
            
            # Extract text
            text = response.text.strip() if response.text else ""
            
            if not text:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            # Validate word count
            word_count = len(text.split())
            
            if word_count < 50:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None
            
            # Return successful paragraph
            return {
                'text': text,
                'topic': topic,
                'word_count': word_count,
                'class': 'ai_neutral',
                'temperature': 0.8,
                'batch_id': batch_id,
                'para_id': para_id
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle rate limiting
            if "429" in error_msg or "quota" in error_msg.lower():
                wait_time = 60 * (attempt + 1)
                print(f"    [Batch {batch_id}, Para {para_id}] Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif "503" in error_msg or "unavailable" in error_msg.lower():
                wait_time = 30 * (attempt + 1)
                print(f"    [Batch {batch_id}, Para {para_id}] Service unavailable. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                time.sleep(5)
                
            if attempt == max_retries - 1:
                print(f"    [Batch {batch_id}, Para {para_id}] Failed after {max_retries} attempts: {e}")
                return None
    
    return None

def generate_batch(model, topics: List[str], batch_id: int, 
                   batch_size: int, max_workers: int = 5) -> List[Dict]:
    """
    Generate a batch of paragraphs using parallel processing.
    """
    print(f"\n  Batch {batch_id}: Generating {batch_size} paragraphs with {max_workers} workers...")
    
    batch_paragraphs = []
    
    # Create tasks for this batch
    tasks = []
    for i in range(batch_size):
        topic = topics[i % len(topics)]
        tasks.append((model, topic, batch_id, i))
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(generate_single_paragraph, *task): task 
            for task in tasks
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_task):
            result = future.result()
            if result:
                batch_paragraphs.append(result)
                completed += 1
                if completed % 10 == 0:
                    print(f"    Progress: {completed}/{batch_size} completed")
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.2)
    
    print(f"  ✓ Batch {batch_id} complete: {len(batch_paragraphs)}/{batch_size} paragraphs generated")
    return batch_paragraphs

def generate_class2_paragraphs(model, topics: List[str],
                               total_paragraphs: int = 500,
                               batch_size: int = 50,
                               max_workers: int = 5,
                               output_file: str = 'class2_ai_neutral1.json'):
    """
    Generate Class 2 paragraphs using batch processing.
    """
    print(f"\nGenerating {total_paragraphs} neutral-style AI paragraphs...")
    print(f"  Topics: {len(topics)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max workers: {max_workers}")
    print(f"  Temperature: 0.8")
    print(f"  Model: {model._model_name}")
    
    all_paragraphs = []
    num_batches = (total_paragraphs + batch_size - 1) // batch_size
    
    print(f"\n{'='*70}")
    print(f"Processing {num_batches} batches...")
    print(f"{'='*70}")
    
    for batch_num in range(num_batches):
        # Calculate size for this batch
        remaining = total_paragraphs - len(all_paragraphs)
        current_batch_size = min(batch_size, remaining)
        
        # Generate batch
        batch_paragraphs = generate_batch(model, topics, batch_num + 1, 
                                         current_batch_size, max_workers)
        
        # Add to collection
        all_paragraphs.extend(batch_paragraphs)
        
        # Save progress after each batch
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_paragraphs, f, indent=2, ensure_ascii=False)
        
        print(f"  Progress saved: {len(all_paragraphs)}/{total_paragraphs} total paragraphs")
        
        # Brief pause between batches
        if batch_num < num_batches - 1:
            print(f"  Waiting 5 seconds before next batch...")
            time.sleep(5)
    
    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_paragraphs, f, indent=2, ensure_ascii=False)
    
    return all_paragraphs

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("STEP 3: Generate Class 2 (AI - Neutral Style) Paragraphs")
    print("="*70 + "\n")
    
    # Get API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("\nPlease set your Gemini API key in .env file:")
        print("  GEMINI_API_KEY=your-api-key-here")
        print("\nGet your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # Initialize Gemini Flash model
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    # Configuration
    TARGET_PARAGRAPHS = 500
    BATCH_SIZE = 50  # Process 50 paragraphs per batch
    MAX_WORKERS = 5  # Number of parallel requests
    OUTPUT_FILE = 'class2_ai_neutral1.json'
    
    # Load topics
    topics = load_topics()
    if not topics:
        return
    
    print(f"Loaded {len(topics)} topics:")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic}")
    
    # Generate paragraphs
    print(f"\n{'='*70}")
    print("STARTING BATCH GENERATION")
    print(f"{'='*70}")
    print(f"Target: {TARGET_PARAGRAPHS} paragraphs")
    estimated_time = (TARGET_PARAGRAPHS / MAX_WORKERS) * 0.5 / 60
    print(f"Estimated time: ~{estimated_time:.1f} minutes")
    print(f"(with parallel processing)")
    
    start_time = time.time()
    
    paragraphs = generate_class2_paragraphs(model, topics, TARGET_PARAGRAPHS, 
                                           BATCH_SIZE, MAX_WORKERS, OUTPUT_FILE)
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total paragraphs generated: {len(paragraphs)}")
    print(f"Time elapsed: {elapsed_time / 60:.1f} minutes")
    print(f"Average time per paragraph: {elapsed_time / len(paragraphs):.2f} seconds")
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Word count statistics
    word_counts = [p['word_count'] for p in paragraphs]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    print(f"\nWord count statistics:")
    print(f"  Average: {avg_words:.1f} words")
    print(f"  Min: {min(word_counts) if word_counts else 0} words")
    print(f"  Max: {max(word_counts) if word_counts else 0} words")
    
    # Topics distribution
    topic_counts = {}
    for p in paragraphs:
        topic = p['topic']
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print(f"\nParagraphs per topic:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count}")
    
    print(f"\n✓ Class 2 generation complete!")
    print("  Next step: Run 04_generate_class3.py")

if __name__ == "__main__":
    main()