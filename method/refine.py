import argparse
import json
import os
from types import MethodType

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
import numpy as np




parser = argparse.ArgumentParser()
parser.add_argument("-a", "--mask_dir", type=str, default=None)
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument('--prompts', type=str, nargs='+', default=['role', 'family','decision'],
                    help="List of prompts to use (e.g., 'role family decision')")
parser.add_argument("-lang", "--language", type=str, default="fr",choices=["en","de","fr"])
parser.add_argument("--bias", type=str, default="0.007")
args = parser.parse_args()
print(args)

os.makedirs(args.output, exist_ok=True)

def intersect1d(tensor1, tensor2):

    set1 = set(tensor1.tolist())
    set2 = set(tensor2.tolist())
    intersect = set1 & set2
    return torch.tensor(sorted(list(intersect)))

intersect_mask  = None

for prompt in args.prompts:

        mask_file = args.mask_dir + '/' + prompt + '/' + args.language + '_' + 'activation_mask.llama-7b.' + args.bias


        if args.mask_dir:
            activation_masks = torch.load(mask_file)
            activation_mask_name = mask_file.split("/")[-1].split(".")
            activation_mask_name = ".".join(activation_mask_name[1:])
        else:
            activation_masks = [None]
            print("error: mask is null!")
            
        if intersect_mask is None:
            intersect_mask  = activation_masks
        else:
            intersect_mask[0] = [intersect1d(t1, t2).values for t1, t2 in zip(activation_masks[0], intersect_mask[0])]
            intersect_mask[1] = [intersect1d(t1, t2).values for t1, t2 in zip(activation_masks[1], intersect_mask[1])]
            
    
# save_mask
torch.save(intersect_mask, f"{args.output}/{args.language}_activation_mask.llama-7b.{args.bias}")  
print(f"Save mask into {args.output} successfully!") 


     
    