import torch
import os
import torch.nn.functional as F
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-p",'--percentage',type=float,default=0.001)
parser.add_argument("-a","--activation_bar_ratio",type=float,default=0.95)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--activation_dir", type=str, default=None)
parser.add_argument("--filter_rate",type=float,default=0.95)
args = parser.parse_args()

print("args:",args)

os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)

def activation(lang):
    top_rate = args.percentage
    filter_rate = args.filter_rate

    activation_bar_ratio = args.activation_bar_ratio
    activation_probs = over_zero / n 
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False
    
    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError
    
    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
    print("top_prob_value:",top_prob_value)
    
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index] 

    print("bitcount:",selected_probs.size(0), torch.bincount(selected_probs.argmax(dim=-1)))
    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
    print((selected_probs > activation_bar).sum(dim=1).tolist())
    gender, indice = torch.where(selected_probs > activation_bar)

    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    for _, index in enumerate(indice.split(torch.bincount(gender).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
        final_indice.append(layer_index)
    formatted_activation_bar_ratio = f"{args.activation_bar_ratio:.2f}"
    formatted_percentage = f"{args.percentage:.3f}"
    
    
    torch.save(final_indice, f"{args.save_dir}/{lang}_activation_mask.llama-7b.{formatted_percentage}")  
    


for lang in ['en','fr','de']:

    print("lang:",lang)
    n, over_zero = [], []
    for gender in ['female', 'male']:
        data = torch.load(f'{args.activation_dir}/activation.{lang}.{gender}.train.llama-7b')
        n.append(data['n'])
        over_zero.append(data['over_zero'])
        print("data['over_zero'].size():",data['over_zero'].size())


    n = torch.tensor(n)
    over_zero = torch.stack(over_zero, dim=-1) 

    num_layers, intermediate_size, lang_num = over_zero.size()

    activation(lang)
