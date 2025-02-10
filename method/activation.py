import argparse
from types import MethodType
import os
import torch
from vllm import LLM, SamplingParams 

from huggingface_hub import login


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--prompt_dir", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--huggingface_token", type=str,default=None)
args = parser.parse_args()


token = args.huggingface_token
login(token)


is_llama = bool(args.model.lower().find('llama') >= 0)

print(torch.cuda.device_count())

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True,dtype='bfloat16')

max_length = model.llm_engine.model_config.max_model_len
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size if is_llama else model.llm_engine.model_config.hf_config.hidden_size * 4



def factory(idx):
    def llama_forward(self, x): 
        gate_up, _ = self.gate_up_proj(x) 
        i = gate_up.size(-1) # [11008]
        gate_up[:, :, : i // 2] = torch.nn.SiLU()(gate_up[:, :, : i // 2])
        activation = gate_up[:, :, : i // 2].float() 
        sum1[idx, :] += activation.sum(dim=(0,1)) 
        sum2[idx, :] += activation.pow(2).sum(dim=(0,1))
        sum3[idx, :] += activation.pow(3).sum(dim=(0,1))
        sum4[idx, :] += activation.pow(4).sum(dim=(0,1))
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x = gate_up[:, :, : i // 2] * gate_up[:, :, i // 2 :]
        x, _ = self.down_proj(x)
        return x

    if is_llama:
        return llama_forward
    else:
        print("Model not found!")

for i in range(num_layers):
    if is_llama:
        obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
    else:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
    obj.forward = MethodType(factory(i), obj)

for lang in ['fr','en','de']:
    
    for gender in ['male','female']:
        
        sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
        sum2 = torch.zeros(num_layers, intermediate_size).to('cuda')
        sum3 = torch.zeros(num_layers, intermediate_size).to('cuda')
        sum4 = torch.zeros(num_layers, intermediate_size).to('cuda')
        over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
        
        if is_llama:
            ids = torch.load(f'{args.prompt_dir}/gender_prompt_{lang}_{gender}')
        else:
            ids = torch.load(f'{args.prompt_dir}/gender_prompt_{lang}_{gender}')

        l = sum([sublist.size(0) for sublist in ids])

        input_ids = ids

        output = [model.generate(prompt_token_ids=[text.tolist()], sampling_params=SamplingParams(max_tokens=1)) for text in input_ids]
        
        output = dict(n=l, sum1=sum1.to('cpu'), sum2=sum2.to('cpu'), sum3=sum3.to('cpu'), sum4=sum4.to('cpu'), over_zero=over_zero.to('cpu'))

        os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
        torch.save(output, f'{args.save_dir}/activation.{lang}.{gender}.train.llama-7b')
