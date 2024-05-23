from transformers import pipeline
import json

# cd ~/.cache/huggingface/hub -> downloaded models are stored here

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_name = "/Users/rodrigoalvarezlucendo/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# pipe = pipeline(task="text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

# # load the prompts
# prompts = json.load(open('data/prompts.json'))

# # generate responses
# responses = []
# for i, p in enumerate(prompts):
#     if i < 2:
#         response_oracle = pipe(p["prompt_oracle"])
#         response_top1 = pipe(p["prompt_top1_similar"])
#         response_top3 = pipe(p["prompt_top3_similar"])
#         response_top5 = pipe(p["prompt_top5_similar"])

#         print(i, response_oracle)

    # response = pipe(prompt)
    # responses.append(response[0]['generated_text'])

