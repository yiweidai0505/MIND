import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import Dataset, DataLoader
import json

def load_english_vocab(vocab_file, tokenizer):
    with open(vocab_file, 'r') as f:
        english_vocab = json.load(f)  
        
    flattened_english_vocab = [item for sublist in english_vocab for item in sublist]
    

    english_vocab_ids = set()
    for word in flattened_english_vocab:
        token_id = tokenizer.encode(word, add_special_tokens=False)  
        english_vocab_ids.update(token_id) 
    
    return english_vocab_ids


class SentencePairDataset(Dataset):
    def __init__(self, sentence_pairs, tokenizer, max_length=32):

        self.sentence_pairs = sentence_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sentence1, sentence2 = self.sentence_pairs[idx]


        encoding1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        encoding2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        
        return {
            'input_ids1': encoding1['input_ids'].squeeze(0),
            'attention_mask1': encoding1['attention_mask'].squeeze(0),
            'input_ids2': encoding2['input_ids'].squeeze(0),
            'attention_mask2': encoding2['attention_mask'].squeeze(0)
        }


class EnglishSentencePairDataset(Dataset):
    def __init__(self, sentence_pairs, tokenizer, english_vocab, max_length=64):

        self.sentence_pairs = sentence_pairs
        self.tokenizer = tokenizer
        self.english_vocab = english_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sentence1, sentence2, sentence3 = self.sentence_pairs[idx]


        encoding1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        encoding2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        encoding3 = self.tokenizer(sentence3, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        input_ids1 = encoding1['input_ids'].squeeze(0)
        

        english_token_ids = [idx for idx, token in enumerate(input_ids1) if token.item() in self.english_vocab]
        

        assert len(english_token_ids) > 0, "No matching token IDs found in the input sentence."

        return {
            'input_ids1': input_ids1,
            'attention_mask1': encoding1['attention_mask'].squeeze(0),
            'input_ids2': encoding2['input_ids'].squeeze(0),
            'attention_mask2': encoding2['attention_mask'].squeeze(0),
            'input_ids3': encoding3['input_ids'].squeeze(0),
            'attention_mask3': encoding3['attention_mask'].squeeze(0),
            'english_token_ids': english_token_ids  
        }


def load_sentence_pairs(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        sentences1 = [line.strip() for line in f1.readlines()]
        sentences2 = [line.strip() for line in f2.readlines()]
    assert len(sentences1) == len(sentences2), "The number of lines in both files should be the same."
    return list(zip(sentences1, sentences2))


def load_sentence_triplets(file1, file2,file3):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(file3, 'r') as f3:
        sentences1 = [line.strip() for line in f1.readlines()]
        sentences2 = [line.strip() for line in f2.readlines()]
        sentences3 = [line.strip() for line in f3.readlines()]
    assert len(sentences1) == len(sentences2), "The number of lines in both files should be the same."
    assert len(sentences1) == len(sentences3), "The number of lines in both files should be the same."
    return list(zip(sentences1, sentences2, sentences3))

