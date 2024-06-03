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

dev = read_json('data/dev.json')[:1200]
# faster way to index the data
dev_df = pd.DataFrame(dev)
dev_df.set_index('_id', inplace=True)

corpus = read_json('data/wiki_musique_corpus.json')
random_contexts = read_json('data/random_contexts.json')

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

def get_random_documents(random_ctx_ids, k=5):
    k_contexts = []
    for i, id in enumerate(random_ctx_ids):
        if i < k:
            title = corpus[id]['title']
            text = corpus[id]['text']
            k_contexts.append((title, text))
    return k_contexts

def create_prompt(contexts, question):
    """Create a prompt for the model."""
    prompt = "Documents:\n"
    for i, (title, text) in enumerate(contexts):
        prompt += f"Document[{i+1}](Title: {title}) {text}\n"
    prompt += f"\nQuestion: {question}"
    return prompt

oracle_random = {}
k = 3
for q_id, random_ctx_ids in random_contexts.items():
    question, oracle_ctxs, answer = get_ground_truth(q_id, supporting_facts=True)
    random_ctxs = get_random_documents(random_ctx_ids, k=k)

    # combine random and oracle contexts
    contexts = random_ctxs + oracle_ctxs
    random.shuffle(contexts)

    prompt = create_prompt(contexts, question)

    oracle_random[q_id] = {"prompt": prompt, "answer": answer}


# save as json
with open(f"results/oracle_random_{k}.json", 'w') as f:
    json.dump(oracle_random, f, indent=4)