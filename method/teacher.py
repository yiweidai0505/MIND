import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
import argparse
import re
import time  # 用于监控训练时长
from utils import load_sentence_pairs, SentencePairDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  
import numpy as np
import random
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--language", type=str, default="en", choices=["en", "fr", "de"])
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--mask_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--activation_bar_ratio", type=float, default=0.95)
    parser.add_argument("--mask_rate", type=float, default=0.007)
    parser.add_argument("--learning_rate", type=float, default=0.01)  
    parser.add_argument("--accumulation_steps", type=int, default=16)  
    parser.add_argument("--training_epoch", type=int, default=1) 
    parser.add_argument("--sentence_pair_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=64, help="32")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--weight_decay", type=int, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--loss_type", type=str, default="JSD")
    parser.add_argument("--alpha", type=float, default=1)
    return parser.parse_args()


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_mask(mask_dir, mask_file):
    mask_path = mask_dir + mask_file
    if mask_dir:
        activation_masks = torch.load(mask_path)
    else:
        activation_masks = [None]
        print("path error!")
    mask = [torch.unique(torch.cat((t1, t2))) for t1, t2 in zip(activation_masks[0], activation_masks[1])]
    return mask

def load_model_with_mask(model_name, mask, device):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  
    ).to(device)

    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'mlp' in name:  
            layer = int(re.findall(r'\d+', name)[0])
            if 'down_proj' in name:  
                param.requires_grad = True

    print(f"Total trainable parameters: {sum(p.requires_grad for p in model.parameters())}")
    return model


def cos_loss(rep1,rep2):
    cos_sim = F.cosine_similarity(rep1, rep2, dim=-1)  
    return 1 - cos_sim.mean() 

def MSE_loss(rep1,rep2):
    res = F.mse_loss(rep1, rep2)  
    return res


# 计算损失
def compute_loss(rep1, rep2, loss_type='cos', temperature=0.5):
    if loss_type == 'cos':
        loss = cos_loss(rep1,rep2)
    elif loss_type == 'MSE':
        loss = MSE_loss(rep1, rep2)
    else:
        print('loss type error')
    loss = torch.nan_to_num(loss, nan=0.0)  
    return loss


def save_model(model, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"Model weights saved at {save_dir}")


def monitor_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)  
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Reserved Memory: {reserved_memory:.2f} GB")

def check_model_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():  
            print(f"Warning: NaN detected in {name}")
            return True
    return False


def train_model_with_mask(
    model,
    train_dataloader,
    optimizer,
    device,
    mask,
    accumulation_steps=1,
    patience=5,
    max_epochs=50,
    step_loss_file="step_losses.txt",
    epoch_loss_file="epoch_losses.txt",
    plot_image="loss_plot.png",
    output_dir=None,
    temperature=0.5,
    loss_type='JSD',
    tokenizer=None,
    alpha=None,
):
    best_loss = float("inf")
    patience_counter = 0
    optimizer.zero_grad()

    step_losses = []
    epoch_losses = []

    start_time = time.time()
    step_loss_file = output_dir + '/' + step_loss_file
    epoch_loss_file = output_dir + '/' + epoch_loss_file

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        accumulated_loss = 0.0
        step_count = 0.0 
        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
        ):
            input_ids1 = batch["input_ids1"].to(device)
            attention_mask1 = batch["attention_mask1"].to(device)
            input_ids2 = batch["input_ids2"].to(device)
            attention_mask2 = batch["attention_mask2"].to(device)
            
            label1 = input_ids1.clone()
            label2 = input_ids2.clone()
            label1[input_ids1==tokenizer.pad_token_id]=-100
            label2[input_ids2==tokenizer.pad_token_id]=-100


            outputs1 = model(
                input_ids=input_ids1, attention_mask=attention_mask1, output_hidden_states=True, labels=label1
            )
            outputs2 = model(
                input_ids=input_ids2, attention_mask=attention_mask2, output_hidden_states=True, labels=label2
            )

            outputs1_hidden_states = outputs1.hidden_states
            

            last_hidden_state = outputs1_hidden_states[-1]  


            rep1 = torch.mean(last_hidden_state.squeeze(0), dim=0) 

            
            outputs2_hidden_states = outputs2.hidden_states

            last_hidden_state = outputs2_hidden_states[-1]  


            rep2 = torch.mean(last_hidden_state.squeeze(0), dim=0) 

            
            loss = alpha*compute_loss(rep1, rep2, loss_type= loss_type, temperature=temperature) + outputs1.loss + outputs2.loss
            loss = loss / accumulation_steps
            loss.backward()


            for name, param in model.named_parameters():
                if param.grad is not None:
                    layer = int(re.findall(r"\d+", name)[0])
                    if "down_proj" in name:
                        
                        mask_tensor = torch.zeros(param.grad.shape[1], dtype=torch.bool, device=param.device)
                        mask_tensor[mask[layer]] = 1


                        mask_tensor_expanded = mask_tensor[None, :].expand(param.grad.shape[0], -1)

                        param.grad *= mask_tensor_expanded
            

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                

                step_loss = accumulated_loss / accumulation_steps
                step_losses.append(step_loss)
                print(f"Update Step {step_count + 1}, Loss: {step_loss:.6f}")
                
                with open(step_loss_file, "w") as f:
                    f.write("\n".join(map(str, step_losses)))
                

                accumulated_loss = 0.0
                step_count += 1
                
            else:

                accumulated_loss += loss.item() * accumulation_steps


        epoch_loss = accumulated_loss / len(train_dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss}")

        with open(epoch_loss_file, "w") as f:
            f.write("\n".join(map(str, epoch_losses)))
        torch.cuda.empty_cache()

    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration // 60:.0f} minutes and {training_duration % 60:.0f} seconds.")
    print(f"Saving Model...")
    save_model(model, save_dir=output_dir)



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(args.seed)

    if os.path.exists(args.output_dir):
        print(f"Output directory already exists: {args.output_dir}")
    else:
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")


    original_sentence_file = args.sentence_pair_dir  + '/' + 'data.txt'
    swapped_sentence_file = args.sentence_pair_dir + '/' + 'gender_swapped_data.txt'
    sentence_pairs = load_sentence_pairs(original_sentence_file, swapped_sentence_file)

    mask = load_mask(args.mask_dir, args.mask_file)
    model = load_model_with_mask(args.model_name, mask, device)
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = "<pad>"

    training_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = torch.optim.Adam(training_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    train_dataset = SentencePairDataset(sentence_pairs, tokenizer,max_length=args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    train_model_with_mask(model, train_dataloader, optimizer, device, mask, accumulation_steps=args.accumulation_steps, max_epochs=args.training_epoch,output_dir=args.output_dir,temperature=args.temperature, loss_type=args.loss_type, tokenizer=tokenizer,alpha=args.alpha)

if __name__ == "__main__":
    main()
