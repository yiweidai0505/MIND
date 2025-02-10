
import argparse
import torch
import os
import json
from huggingface_hub import login

from vllm import LLM

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-c", "--category", type=str, default="talent",choices=["decision","family","role"])
parser.add_argument("--huggingface_token", type=str,default=None)
args = parser.parse_args()


token = args.huggingface_token
login(token)

print(torch.cuda.device_count())
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
max_length = model.llm_engine.model_config.max_model_len

model.llm_engine.tokenizer.pad_token = model.llm_engine.tokenizer.eos_token

for lang in ['en','fr','de']:
    file = f'../data/prompt/{args.category}/gender_prompt_{lang}.json'
    with open(file, 'r', encoding='utf-8') as file:
        prompts = json.load(file)
        prompts_female = [sentence[1] for sentence in prompts]
        prompts_male = [sentence[0] for sentence in prompts]

    tokenized_texts = [model.llm_engine.tokenizer(text, return_tensors='pt')['input_ids'].squeeze(0) for text in prompts_female]

    save_path = f'../data/prompt/{args.category}/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_name = save_path+f'gender_prompt_{lang}_female'
    torch.save(tokenized_texts, file_name)

    tokenized_texts = [model.llm_engine.tokenizer(text, return_tensors='pt')['input_ids'].squeeze(0) for text in prompts_male]

    file_name = save_path+f'gender_prompt_{lang}_male'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(tokenized_texts, file_name)