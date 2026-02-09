#!/usr/bin/env python3
"""
Script 1: Clean and Extract Paragraphs
Processes Project Gutenberg texts and extracts clean paragraphs (100-200 words)
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict

def clean_gutenberg_text(text: str) -> str:
    """
    Clean Project Gutenberg text by removing headers, footers, and formatting.
    """
    # Remove the Project Gutenberg header (everything before "*** START OF")
    start_patterns = [
        r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
    ]
    
    for pattern in start_patterns:
        start_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if start_match:
            text = text[start_match.end():]
            break
    
    # Remove the Project Gutenberg footer (everything after "*** END OF")
    end_patterns = [
        r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
    ]
    
    for pattern in end_patterns:
        end_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if end_match:
            text = text[:end_match.start()]
            break
    
    # Remove chapter headings (various formats)
    text = re.sub(r'CHAPTER [IVXLCDM]+\.?\s*\n', '', text)
    text = re.sub(r'Chapter [0-9]+\.?\s*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CHAPTER [0-9]+\.?\s*\n', '', text)
    
    # Remove section markers
    text = re.sub(r'VOLUME [IVXLCDM]+\.?\s*\n', '', text)
    text = re.sub(r'PART [IVXLCDM]+\.?\s*\n', '', text)
    
    # Remove standalone Roman numerals (chapter numbers)
    text = re.sub(r'\n\s*[IVXLCDM]+\.?\s*\n', '\n\n', text)
    
    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r'\n\n\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove lines that are all caps (often titles/headers)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that are very short and all caps
        if len(line.strip()) < 50 and line.strip().isupper():
            continue
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    return text.strip()

def extract_paragraphs(text: str, min_words: int = 100, max_words: int = 200) -> List[str]:
    """
    Extract paragraphs within the specified word count range.
    """
    # Split by double newlines (paragraph breaks)
    paragraphs = re.split(r'\n\n+', text)
    
    filtered = []
    for para in paragraphs:
        # Clean up the paragraph
        para = para.replace('\n', ' ')  # Remove line breaks within paragraph
        para = re.sub(r'\s+', ' ', para)  # Normalize whitespace
        para = para.strip()
        
        # Skip if empty
        if not para:
            continue
        
        # Skip if it looks like metadata or headers
        if para.isupper() and len(para) < 100:
            continue
        
        # Check word count
        words = para.split()
        word_count = len(words)
        
        if min_words <= word_count <= max_words:
            filtered.append(para)
    
    return filtered

def process_books(data_dir: str, output_file: str):
    """
    Process all books in the data directory and extract paragraphs.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"ERROR: Directory '{data_dir}' not found!")
        print("Please make sure the 'data' folder exists with subfolders:")
        print("  - data/charles_dickens/")
        print("  - data/jane_austen/")
        return
    
    all_paragraphs = []
    
    # Define the authors and their folders
    authors = {
        'charles_dickens': 'Charles Dickens',
        'jane_austen': 'Jane Austen'
    }
    
    for folder_name, author_name in authors.items():
        author_dir = data_path / folder_name
        
        if not author_dir.exists():
            print(f"WARNING: Folder '{author_dir}' not found, skipping...")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {author_name}")
        print(f"{'='*70}")
        
        # Get all .txt files in the author's directory
        txt_files = list(author_dir.glob('*.txt'))
        
        if not txt_files:
            print(f"  No .txt files found in {author_dir}")
            continue
        
        for txt_file in txt_files:
            print(f"\n  Reading: {txt_file.name}")
            
            try:
                # Read the file
                with open(txt_file, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                
                # Clean the text
                cleaned_text = clean_gutenberg_text(raw_text)
                
                # Extract paragraphs
                paragraphs = extract_paragraphs(cleaned_text)
                
                print(f"    Extracted: {len(paragraphs)} paragraphs")
                
                # Add to dataset with metadata
                for para in paragraphs:
                    all_paragraphs.append({
                        'text': para,
                        'author': author_name,
                        'book': txt_file.stem,  # filename without extension
                        'word_count': len(para.split()),
                        'class': 'human'
                    })
                
            except Exception as e:
                print(f"    ERROR processing {txt_file.name}: {e}")
                continue
    
    # Save the results
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total paragraphs extracted: {len(all_paragraphs)}")
    
    # Count by author
    from collections import Counter
    author_counts = Counter(p['author'] for p in all_paragraphs)
    for author, count in author_counts.items():
        print(f"  {author}: {count} paragraphs")
    
    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_paragraphs, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to: {output_file}")
    print(f"{'='*70}\n")

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("STEP 1: Clean and Extract Paragraphs from Project Gutenberg Books")
    print("="*70 + "\n")
    
    # Configuration
    DATA_DIR = 'data'
    OUTPUT_FILE = 'class1_human_paragraphs.json'
    
    # Process the books
    process_books(DATA_DIR, OUTPUT_FILE)
    
    print("\n✓ Paragraph extraction complete!")
    print("  Next step: Run 02_extract_topics.py")

if __name__ == "__main__":
    main()