import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
import argparse
import re
import time  
from mllm.MIND.method.utils import load_sentence_pairs, EnglishSentencePairDataset, load_english_vocab, load_sentence_triplets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  
from mllm.MIND.method.teacher import set_seed


import torch.nn.functional as F


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--teacher', type=str,default=None)
    parser.add_argument("--language", type=str, default="fr", choices=["en", "es", "fr", "de"])
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--mask_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--activation_bar_ratio", type=float, default=0.95)
    parser.add_argument("--learning_rate", type=float, default=5e-5)  
    parser.add_argument("--training_step", type=int, default=1)  
    parser.add_argument("--training_epoch", type=int, default=1) 
    parser.add_argument("--english_file", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="opus_books")
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--c", type=float, default=0.3)
    parser.add_argument("--accumulation_steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--weight_decay", type=int, default=1e-5)
    return parser.parse_args()


def load_mask(mask_dir, mask_file):
    mask_path = mask_dir + mask_file
    if mask_dir:
        activation_masks = torch.load(mask_path)
    else:
        activation_masks = [None]
        print("path error!")
    mask = [torch.unique(torch.cat((t1, t2))) for t1, t2 in zip(activation_masks[0], activation_masks[1])]
    return mask

def load_model_with_mask(model_name, is_student, device):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, 
    ).to(device)


    for name, param in model.named_parameters():
        param.requires_grad = False
    
    if is_student is True:


        for name, param in model.named_parameters():
            if 'mlp' in name: 
                layer = int(re.findall(r'\d+', name)[0])

                if 'down_proj' in name:  
                    param.requires_grad = True

        print(f"Total trainable parameters: {sum(p.requires_grad for p in model.parameters())}")
        
    return model


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



def compute_word_level_loss(rep1, rep2, c):
    
    rep1_norm = F.normalize(rep1, p=2, dim=-1)  
    rep2_norm = F.normalize(rep2, p=2, dim=-1)  

    similarity_matrix = torch.mm(rep1_norm, rep2_norm.T)  


    positive_mask = similarity_matrix > c  
    negative_mask = ~positive_mask  


    positive_sim = similarity_matrix[positive_mask]
    positive_loss = torch.mean((1 - positive_sim) ** 2)  

    total_loss = positive_loss 
    
    total_loss = torch.nan_to_num(total_loss, nan=0.0)  

    return total_loss



def compute_sentence_level_loss(rep1, rep2, rep3, margin=0.5):

    rep1_norm = F.normalize(rep1, p=2, dim=-1)  
    rep2_norm = F.normalize(rep2, p=2, dim=-1)  
    rep3_norm = F.normalize(rep3, p=2, dim=-1)  

   
    positive_similarity = F.cosine_similarity(rep1_norm.unsqueeze(0), rep2_norm.unsqueeze(0))  


    negative_similarity = F.cosine_similarity(rep1_norm.unsqueeze(0), rep3_norm.unsqueeze(0)) 


    positive_loss = torch.mean((1 - positive_similarity) ** 2)  
    

    negative_loss = torch.mean(torch.clamp(margin - (1 - negative_similarity), min=0) ** 2)


    total_loss = positive_loss + negative_loss
    
    total_loss = torch.nan_to_num(total_loss, nan=0.0)  

    return total_loss




def train_model(teacher_model, student_model, train_dataloader, optimizer, 
                device, mask, c, alpha, beta, accumulation_steps=1, patience=5, 
                max_epochs=50, loss_file="loss.txt",aggregate_method='mean', 
                output_dir=None, step_loss_file=None, epoch_loss_file=None, plot_image=None,tokenizer=None):

    start_time = time.time()
    
    best_loss = float("inf")
    patience_counter = 0
    optimizer.zero_grad()

    step_losses = []
    epoch_losses = []

    start_time = time.time()
    step_loss_file = output_dir + '/' + step_loss_file
    epoch_loss_file = output_dir + '/' + epoch_loss_file

    for epoch in range(max_epochs):
        student_model.train()
        epoch_loss = 0.0
        accumulated_loss = 0.0
        step_count = 0.0  

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{max_epochs}", dynamic_ncols=True)):
            input_ids1 = batch['input_ids1'].to(device[0])  
            attention_mask1 = batch['attention_mask1'].to(device[0])  
            
            input_ids2 = batch['input_ids2'].to(device[0])  
            attention_mask2 = batch['attention_mask2'].to(device[0])  
            input_ids3 = batch['input_ids3'].to(device[0])  
            attention_mask3 = batch['attention_mask3'].to(device[0])  
            
            english_token_ids = torch.tensor(batch['english_token_ids']).to(device[0])
            
            label = input_ids2.clone()
            label[input_ids2==tokenizer.pad_token_id]=-100
            
            
            

            with torch.no_grad():

                outputs1 = teacher_model(input_ids=input_ids1, attention_mask=attention_mask1, output_hidden_states=True)


                outputs1_hidden_states = outputs1.hidden_states
                
                outputs1_aggregated_hidden_states = outputs1_hidden_states[-1].squeeze(0)
                
                outputs1_selected_rep = torch.index_select(outputs1_aggregated_hidden_states, dim=0, index=english_token_ids)
                
                outputs1_representation = torch.mean(outputs1_aggregated_hidden_states.squeeze(0), dim=0)  
            


            
            outputs2 = student_model(input_ids=input_ids2, attention_mask=attention_mask2, output_hidden_states=True,labels=label)
            outputs2_hidden_states = outputs2.hidden_states  
            outputs3 = student_model(input_ids=input_ids3, attention_mask=attention_mask3, output_hidden_states=True)
            outputs3_hidden_states = outputs3.hidden_states  
            

            outputs2_aggregated_hidden_states = outputs2_hidden_states[-1].squeeze(0)
            outputs3_aggregated_hidden_states = outputs3_hidden_states[-1].squeeze(0)

            outputs2_selected_rep = outputs2_aggregated_hidden_states  
            

            outputs2_representation = torch.mean(outputs2_aggregated_hidden_states.squeeze(0), dim=0)  
            outputs3_representation = torch.mean(outputs3_aggregated_hidden_states.squeeze(0), dim=0)  
            
            
            word_level_loss = compute_word_level_loss(outputs1_selected_rep, outputs2_selected_rep, c)
            
            
            sentence_level_loss = compute_sentence_level_loss(outputs2_representation,outputs1_representation,outputs3_representation)
            


            loss = alpha * word_level_loss + beta * sentence_level_loss + outputs2.loss

            
            loss = loss / accumulation_steps
            loss.backward()
            
            
            for name, param in student_model.named_parameters():
                if param.grad is not None:
                    layer = int(re.findall(r"\d+", name)[0])
                    if "down_proj" in name:
                        
                        mask_tensor = torch.zeros(param.grad.shape[1], dtype=torch.bool, device=param.device)
                        mask_tensor[mask[layer]] = 1

                        
                        mask_tensor_expanded = mask_tensor[None, :].expand(param.grad.shape[0], -1)

                        
                        param.grad *= mask_tensor_expanded
            
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                
                step_loss = accumulated_loss / accumulation_steps
                step_losses.append(step_loss)
                print(f"Update Step {step_count + 1}, Loss: {step_loss:.6f}")
                
                
                accumulated_loss = 0.0
                step_count += 1
                
                with open(step_loss_file, "w") as f:
                    f.write("\n".join(map(str, step_losses)))
                
                
            else:
                
                accumulated_loss += loss.item() * accumulation_steps
                
                
        
            
        epoch_loss = accumulated_loss / len(train_dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss}")

        # Save epoch losses to file
        with open(epoch_loss_file, "w") as f:
            f.write("\n".join(map(str, epoch_losses)))

        torch.cuda.empty_cache()


    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration // 60:.0f} minutes and {training_duration % 60:.0f} seconds.")
    print(f"Saving Model...")
    save_model(student_model, save_dir=output_dir)

# 主函数
def main():
    args = parse_args()
    
    device = [torch.device("cuda:0")]
    
    set_seed(args.seed)

    if os.path.exists(args.output_dir):
        print(f"Output directory already exists: {args.output_dir}")
    else:
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    en_file = args.dataset_dir + '/' + args.language + '/' + args.dataset + '/' + 'maxlength_64_en.txt'
    tar_file = args.dataset_dir + '/' + args.language + '/' + args.dataset + '/' + 'maxlength_64_' + args.language + '.txt'
    tar_neg_file = args.dataset_dir + '/' + args.language + '/' + args.dataset + '/' + 'maxlength_64_' + args.language + '_neg.txt'
    
    sentence_pairs = load_sentence_triplets(en_file, tar_file, tar_neg_file)
    
    student_mask = load_mask(args.mask_dir, args.mask_file)
    
    
    teacher_model = load_model_with_mask(args.teacher, False, device[0])

    
    student_model = load_model_with_mask(args.model_name, True, device[0])

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = "<pad>"
    
    
    english_vocab_ids = load_english_vocab(args.english_file,tokenizer)
    

    training_parameters = list(filter(lambda p: p.requires_grad, student_model.parameters()))

    
    optimizer = torch.optim.Adam(training_parameters, lr=args.learning_rate,weight_decay=args.weight_decay)

    train_dataset = EnglishSentencePairDataset(sentence_pairs, tokenizer, english_vocab_ids)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=8)

    train_model(teacher_model, student_model, train_dataloader, optimizer, device, student_mask, 
                args.c, args.alpha, args.beta, accumulation_steps=args.accumulation_steps, max_epochs=args.training_epoch,
                output_dir=args.output_dir, step_loss_file="step_losses.txt", epoch_loss_file="epoch_losses.txt", 
                plot_image="loss_plot.png",tokenizer=tokenizer)


if __name__ == "__main__":
    main()