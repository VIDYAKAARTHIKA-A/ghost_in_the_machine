import json
import csv
import pandas as pd

def rebuild_csv_verified():
    """Rebuild CSV with verification"""
    
    print("Rebuilding CSV with verification...")
    
    # Load JSON files
    print("\n1. Loading JSON files...")
    with open(r'D:\precog_task\class1_human_paragraphs.json', 'r', encoding='utf-8') as f:
        human_data = json.load(f)
    
    with open(r'D:\precog_task\class2_ai_neutral1.json', 'r', encoding='utf-8') as f:
        ai_neutral_data = json.load(f)
    
    with open(r'D:\precog_task\class3_ai_styled1.json', 'r', encoding='utf-8') as f:
        ai_styled_data = json.load(f)
    
    print(f"   Human: {len(human_data)} samples")
    print(f"   AI Neutral: {len(ai_neutral_data)} samples")
    print(f"   AI Styled: {len(ai_styled_data)} samples")
    
    # Verify the data
    print("\n2. Verifying data...")
    print("\n   HUMAN samples (first 2):")
    for i, item in enumerate(human_data[:2]):
        print(f"      {i+1}. {item['text'][:100]}...")
        print(f"         Class in JSON: {item.get('class', 'NOT FOUND')}")
    
    print("\n   AI_NEUTRAL samples (first 2):")
    for i, item in enumerate(ai_neutral_data[:2]):
        print(f"      {i+1}. {item['text'][:100]}...")
        print(f"         Class in JSON: {item.get('class', 'NOT FOUND')}")
    
    print("\n   AI_STYLED samples (first 2):")
    for i, item in enumerate(ai_styled_data[:2]):
        print(f"      {i+1}. {item['text'][:100]}...")
        print(f"         Class in JSON: {item.get('class', 'NOT FOUND')}")
    
    # Create rows
    print("\n3. Creating CSV rows...")
    rows = []
    
    # Add human paragraphs - FORCE the class label
    for item in human_data:
        rows.append({
            'text': item['text'],
            'class': 'human'  # Force it
        })
    
    # Add AI neutral paragraphs - FORCE the class label
    for item in ai_neutral_data:
        rows.append({
            'text': item['text'],
            'class': 'ai_neutral'  # Force it
        })
    
    # Add AI styled paragraphs - FORCE the class label
    for item in ai_styled_data:
        rows.append({
            'text': item['text'],
            'class': 'ai_styled'  # Force it
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Verify
    print("\n4. Verification:")
    print(f"   Total rows: {len(df)}")
    print(f"\n   Class distribution:")
    print(df['class'].value_counts())
    
    print("\n   Sample from each class in final CSV:")
    for cls in ['human', 'ai_neutral', 'ai_styled']:
        sample = df[df['class'] == cls].iloc[0]['text']
        print(f"\n   {cls}: {sample[:150]}...")
    
    # Save
    output_path = r'D:\precog_task\paragraphs_dataset_FIXED1.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n5. ✓ Saved to {output_path}")
    
    return df

# Run it
df = rebuild_csv_verified()