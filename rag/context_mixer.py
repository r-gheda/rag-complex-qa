"""
This file creates LLM prompts for each question in the dev set, mixing oracle 
contexts with random, hard negative, and ADORE contexts. The prompts are saved as 
a json file.
"""
import json
import pandas as pd
import random

def read_json(file_path: str) -> dict:
    """Read a json file and return a dict."""
    with open(file_path) as f:
        data = json.load(f)
    return data

# read the data
dev = read_json('data/dev.json')[:1200]
dev_df = pd.DataFrame(dev)
dev_df.set_index('_id', inplace=True) # faster way to index by _id
corpus = read_json('data/wiki_musique_corpus.json')
random_ctxs = read_json('data/random_contexts.json')
adore_ctxs = read_json('data/adore_top_100_response.json')
hard_negative_ctxs = read_json('data/hard_negatives.json')

def get_ground_truth(idx, supporting_facts=False):
    """Extract the question, oracle contexts and answer from dev.
    The oracle contexts is a list of tuples (title, text).
    """
    row = dev_df.loc[idx]
    if supporting_facts:
        sfs = [t for t, _ in row['supporting_facts']]
        oracle = [(title, ' '.join(texts)) for title, texts in row['context'] if title in sfs]
    else:
        oracle = [(title, ' '.join(texts)) for title, texts in row['context']]
    return row['question'], oracle, row['answer']

def is_text_similar(text1, text2, sim_threshold=0.95):
    """Check if two texts are similar based on Jaccard similarity."""
    words1 = set(text1.split())
    words2 = set(text2.split())
    jac_sim = len(words1.intersection(words2)) / len(words1.union(words2))
    return True if jac_sim > sim_threshold else False

def get_top_k_contexts(corpus_ids, k, oracle_ctxs, filter_oracle):
    """Get the top k contexts from the corpus excluding the oracle contexts 
    if filter_oracle is True. Returns a list of tuples (title, text)"""
    k_contexts = []
    i = 0
    for id in corpus_ids:
        if i == k:
            break
        title = corpus[id]['title']
        text = corpus[id]['text']
        if filter_oracle:
            # check if the text is similar to any oracle context
            if any(is_text_similar(text, oracle_text) for _, oracle_text in oracle_ctxs):
                continue
            else:
                i += 1
                k_contexts.append((title, text))
        else:
            i += 1
            k_contexts.append((title, text))
    return k_contexts

def create_prompt(contexts, question):
    """Create prompt for LLM."""
    prompt = "Documents:\n"
    for i, (title, text) in enumerate(contexts):
        prompt += f"Document[{i+1}](Title: {title}) {text}\n"
    prompt += f"\nQuestion: {question}"
    return prompt

def create_oracle_and_k_other_prompts(other_ctxs, k, filter_oracle=False, file_name=None):
    """Generate LLM prompts for each question in dev, mixing oracle contexts with 
    top-k other contexts from the corpus.

    Args:
        other_ctxs: the other contexts to mix with oracle contexts.
        k: the number of other contexts to mix with oracle contexts.
        file_name: the name of the file to save the prompts.
    """
    
    oracle_k_other = {}
    
    for q_id, ctx_ids in other_ctxs.items():
        # get question, oracle contexts (only evidences) and answer
        question, oracle_ctxs, answer = get_ground_truth(q_id, supporting_facts=True)
        # get top k other contexts
        other_ctxs = get_top_k_contexts(corpus_ids=ctx_ids, k=k, oracle_ctxs=oracle_ctxs, filter_oracle=filter_oracle)
        # combine and shuffle other contexts with oracle contexts
        contexts = other_ctxs + oracle_ctxs
        random.shuffle(contexts)
        prompt = create_prompt(contexts, question)
        oracle_k_other[q_id] = {"prompt": prompt, "answer": answer}
    
    # save as json if file_name is provided
    if file_name:
        with open(f"results/{file_name}_{k}.json", 'w') as f:
            json.dump(oracle_k_other, f, indent=4)

create_oracle_and_k_other_prompts(adore_ctxs, k=1, filter_oracle=True, file_name='oracle_adore')
create_oracle_and_k_other_prompts(adore_ctxs, k=2, filter_oracle=True, file_name='oracle_adore')
create_oracle_and_k_other_prompts(adore_ctxs, k=3, filter_oracle=True, file_name='oracle_adore')