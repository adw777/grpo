import pandas as pd
import numpy as np
import openai
from datasets import load_dataset
import random
from tqdm import tqdm
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")    

def load_datasets():
    """Load the two datasets"""
    qna_dataset = load_dataset("axondendriteplus/legal-qna-dataset")
    rag_dataset = load_dataset("axondendriteplus/legal-rag-embedding-dataset")
    
    return qna_dataset, rag_dataset

def create_positive_examples(qna_data, rag_data):
    """Create positive examples by matching queries between datasets"""
    positive_examples = []
    
    # Convert to pandas for easier manipulation
    qna_df = pd.DataFrame(qna_data['train'])
    rag_df = pd.DataFrame(rag_data['train'])
    
    # Match queries between datasets
    for _, qna_row in qna_df.iterrows():
        question = qna_row['question']
        answer = qna_row['answer']
        
        # Find matching context in RAG dataset
        matching_context = rag_df[rag_df['question'] == question]
        
        if len(matching_context) > 0:
            context = matching_context.iloc[0]['context']
            positive_examples.append({
                'question': question,
                'answer': answer,
                'context': context,
                'label': 1
            })
    
    return positive_examples

def generate_negative_answer(context, original_answer):
    """Generate a negative answer using GPT-4o-mini"""
    prompt = f"""Given the following context and original answer, generate a plausible but incorrect answer that would NOT be derivable from the given context. The answer should be related to legal topics but contain information not present in the context.

Context: {context}

Original Answer: {original_answer}

Generate a different answer that sounds plausible but is NOT supported by the given context:"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates plausible but incorrect answers for training data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating negative answer: {e}")
        return None

def create_negative_examples_type1(positive_examples, num_negatives):
    """Create negative examples by generating wrong answers for correct contexts"""
    negative_examples = []
    
    selected_positives = random.sample(positive_examples, min(num_negatives, len(positive_examples)))
    
    for example in tqdm(selected_positives, desc="Generating negative answers"):
        negative_answer = generate_negative_answer(example['context'], example['answer'])
        
        if negative_answer:
            negative_examples.append({
                'question': example['question'],
                'answer': negative_answer,
                'context': example['context'],
                'label': 0
            })
        
        # Add delay to avoid rate limiting
        time.sleep(0.1)
    
    return negative_examples

def create_negative_examples_type2(positive_examples, num_negatives):
    """Create negative examples by pairing correct answers with wrong contexts"""
    negative_examples = []
    
    for i in range(num_negatives):
        # Randomly select an answer and a different context
        answer_example = random.choice(positive_examples)
        context_example = random.choice(positive_examples)
        
        # Make sure they're different
        while answer_example['question'] == context_example['question']:
            context_example = random.choice(positive_examples)
        
        negative_examples.append({
            'question': answer_example['question'],
            'answer': answer_example['answer'],
            'context': context_example['context'],
            'label': 0
        })
    
    return negative_examples

def balance_dataset(positive_examples, negative_ratio=1.0):
    """Create a balanced dataset with positive and negative examples"""
    num_positives = len(positive_examples)
    num_negatives = int(num_positives * negative_ratio)
    
    print(f"Creating {num_positives} positive examples and {num_negatives} negative examples")
    
    # Create two types of negative examples
    neg_type1_count = num_negatives // 2
    neg_type2_count = num_negatives - neg_type1_count
    
    # Type 1: Wrong answers with correct contexts
    negative_examples_type1 = create_negative_examples_type1(positive_examples, neg_type1_count)
    
    # Type 2: Correct answers with wrong contexts
    negative_examples_type2 = create_negative_examples_type2(positive_examples, neg_type2_count)
    
    # Combine all examples
    all_examples = positive_examples + negative_examples_type1 + negative_examples_type2
    
    # Shuffle the dataset
    random.shuffle(all_examples)
    
    return all_examples

def save_dataset(dataset, filename):
    """Save the dataset to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {filename}")

    # Save the dataset to a JSONL file
    jsonl_filename = filename.replace('.json', '.jsonl')
    with open(jsonl_filename, 'w', encoding='utf-8') as f_jsonl:
        for example in dataset:
            json.dump(example, f_jsonl, ensure_ascii=False)
            f_jsonl.write('\n')

    print(f"Dataset also saved to {jsonl_filename}")
    print(f"Total examples: {len(dataset)}")
    print(f"Positive examples: {sum(1 for ex in dataset if ex['label'] == 1)}")
    print(f"Negative examples: {sum(1 for ex in dataset if ex['label'] == 0)}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load datasets
    print("Loading datasets...")
    qna_dataset, rag_dataset = load_datasets()
    
    # Create positive examples
    print("Creating positive examples...")
    positive_examples = create_positive_examples(qna_dataset, rag_dataset)
    
    print(f"Created {len(positive_examples)} positive examples")
    
    # Create balanced dataset with negative examples
    print("Creating negative examples...")
    final_dataset = balance_dataset(positive_examples, negative_ratio=1.0)
    
    # Save the dataset
    save_dataset(final_dataset, 'legal_context_answer_dataset.json')
    
    # Also save as CSV for easy inspection
    df = pd.DataFrame(final_dataset)
    df.to_csv('legal_context_answer_dataset.csv', index=False)
    print("Dataset also saved as CSV file")

if __name__ == "__main__":
    main()