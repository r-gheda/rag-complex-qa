import json
import pandas as pd
import os
import random

# change to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def read_json(file_path: str) -> dict:
    """Read a json file and return a dict."""
    with open(file_path) as f:
        data = json.load(f)
    return data

# read the data
dev = read_json('data/dev.json')[:1200]
# faster way to index the data
dev_df = pd.DataFrame(dev)
dev_df.set_index('_id', inplace=True)
corpus = read_json('data/wiki_musique_corpus.json')
hard_negatives = read_json('data/hard_negatives.json')

def get_ground_truth(idx, supporting_facts=False):
    """Extract the question, oracle contexts and answer from dev.
    The oracle contexts are tuples (title, text).
    """
    row = dev_df.loc[idx]
    if supporting_facts:
        sfs = [t for t, _ in row['supporting_facts']]
        oracle = [(title, ' '.join(texts)) for title, texts in row['context'] if title in sfs]
    else:
        oracle = [(title, ' '.join(texts)) for title, texts in row['context']]
    return row['question'], oracle, row['answer']

def get_hard_negatives(hard_neg_ids, k=5):
    """Get top k hard negatives from the corpus."""
    k_hard_negatives = []
    for i, id in enumerate(hard_neg_ids):
        if i < k:
            title = corpus[id]['title']
            text = corpus[id]['text']
            k_hard_negatives.append((title, text))
    return k_hard_negatives

def create_prompt(contexts, question):
    """Create a prompt for LLM."""
    prompt = "Documents:\n"
    for i, (title, text) in enumerate(contexts):
        prompt += f"Document[{i+1}](Title: {title}) {text}\n"
    prompt += f"\nQuestion: {question}"
    return prompt

def create_oracle_and_hard_negative_prompts(k=3):
    """Generate prompts wirh oracle and k hard negatives 
    context for each question in dev."""

    oracle_hard_negative = {}
    
    for q_id, hard_neg_dict in hard_negatives.items():
        # get ids of hard negatives
        hard_neg_ids = hard_neg_dict.keys()
        # get question, oracle contexts and answer
        question, oracle_ctxs, answer = get_ground_truth(q_id, supporting_facts=True)
        # get k hard negatives
        hard_negs = get_hard_negatives(hard_neg_ids, k=k)
        # combine and shuffle hard negatives and oracle contexts
        contexts = hard_negs + oracle_ctxs
        random.shuffle(contexts)
        # create prompt for LLM
        prompt = create_prompt(contexts, question)
        oracle_hard_negative[q_id] = {"prompt": prompt, "answer": answer}
    
    # save prompts as json with given name
    with open(f"results/oracle_hard_negatives_{k}.json", 'w') as f:
        json.dump(oracle_hard_negative, f, indent=4)

ks = [1, 2, 3]
for k in ks:
    create_oracle_and_hard_negative_prompts(k=k)
    print(f"Prompts with oracle and {k} hard negatives created.")