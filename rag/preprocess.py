import json
import pandas as pd
import os

# change to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def read_json(file_path: str) -> dict:
    """Read a json file and return a dict."""
    with open(file_path) as f:
        data = json.load(f)
    return data

response = read_json('data/response.json')
corpus = read_json('data/wiki_musique_corpus.json')
dev = read_json('data/dev.json')[:1200]

# faster way to index the data
dev_df = pd.DataFrame(dev)
dev_df.set_index('_id', inplace=True)

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

def get_top_k_contexts(responses, k=5):
    """Extract the top k similar contexts from the corpus. Each
    context is a tuple (title, text).
    """
    top_k_contexts = []
    for i, (idx, score) in enumerate(responses.items()):
        if i < k:
            tittle = corpus[idx]['title']
            text = corpus[idx]['text']
            top_k_contexts.append((tittle, text))
    return top_k_contexts

def create_prompt(contexts, question):
    """Create a prompt for the model without task instruction."""
    prompt = "Documents:"
    for i, (title, text) in enumerate(contexts):
        prompt += f"Document[{i+1}](Title: {title}) {text}"
    prompt += f"Question: {question}"
    return prompt

results = []
for _id, similar_ctx_ids in response.items():
    question, oracle_ctxs, answer = get_ground_truth(_id, supporting_facts=True)
    similar_top1_ctxs = get_top_k_contexts(similar_ctx_ids, k=1)
    similar_top3_ctxs = get_top_k_contexts(similar_ctx_ids, k=3)
    similar_top5_ctxs = get_top_k_contexts(similar_ctx_ids, k=5)
    
    # create prompts
    prompt_oracle = create_prompt(oracle_ctxs, question)
    prompt_top1_similar = create_prompt(similar_top1_ctxs, question)
    prompt_top3_similar = create_prompt(similar_top3_ctxs, question)
    prompt_top5_similar = create_prompt(similar_top5_ctxs, question)
    
    # save prompts in a list
    results.append({
        "prompt_oracle": prompt_oracle,
        "prompt_top1_similar": prompt_top1_similar,
        "prompt_top3_similar": prompt_top3_similar,
        "prompt_top5_similar": prompt_top5_similar,
        "answer": answer
    })

# convert results to a DataFrame
results_df = pd.DataFrame(results)

# save results to a csv file
results_df.to_csv('data/prompts.csv', index=False)