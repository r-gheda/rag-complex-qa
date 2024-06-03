import json
import os
from langchain.evaluation import ExactMatchStringEvaluator

# change to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def read_json(file_path: str) -> dict:
    """Read a json file and return a dict."""
    with open(file_path) as f:
        data = json.load(f)
    flattened_list = [el for batch in data for el in batch]
    return flattened_list

oracle_answers = read_json("results/mistral_oracle_answers.json")
oracle2_answers = read_json("results/mistral_oracle2_answers.json")
top1_answers = read_json("results/mistral_top1_answers.json")
top3_answers = read_json("results/mistral_top3_answers.json")
top5_answers = read_json("results/mistral_top5_answers.json")

def compute_exact_match_score(answers):
    evaluator = ExactMatchStringEvaluator(
        ignore_case=True,
        ignore_punctuation=True,
    )
    no_exact_match = 0
    no_res = 0
    for pred, ref in answers:
        pred = pred.strip() # remove trailing whitespaces
        pred = pred.replace("Answer: ", "") # remove "Answer: " prefix
        # count number of times prediction is "NO-RES"
        if "NO-RES" in pred:
            no_res += 1
        # compute exact match score
        score = evaluator.evaluate_strings(prediction=pred, reference=ref)['score']
        print(f"{pred} | {ref} | {score}")
        no_exact_match += score
    return no_exact_match / len(answers) * 100, no_res

# exact_match_score_oracle, no_res_oracle = compute_exact_match_score(oracle_answers)
# print(f"ORACLE - Exact match score: {exact_match_score_oracle:.1f}%, NO-RES: {no_res_oracle}")
exact_match_score_oracle2, no_res_oracle2 = compute_exact_match_score(oracle2_answers)
print(f"ORACLE2 - Exact match score: {exact_match_score_oracle2:.1f}%, NO-RES: {no_res_oracle2}")

# exact_match_score_top1, no_res_top1 = compute_exact_match_score(top1_answers)
# print(f"TOP1 - Exact match score: {exact_match_score_top1:.1f}%, NO-RES: {no_res_top1}")
# exact_match_score_top3, no_res_top3 = compute_exact_match_score(top3_answers)
# print(f"TOP3 - Exact match score: {exact_match_score_top3:.1f}%, NO-RES: {no_res_top3}")
# exact_match_score_top5, no_res_top5 = compute_exact_match_score(top5_answers)
# print(f"TOP5 - Exact match score: {exact_match_score_top5:.1f}%, NO-RES: {no_res_top5}")