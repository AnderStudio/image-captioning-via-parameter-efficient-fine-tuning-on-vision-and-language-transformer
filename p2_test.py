import torch
from torch.utils.data import DataLoader

import numpy as np
import sys
import os

import p2_model 
import p2_dataset

from tqdm import tqdm
import torch.nn.functional as F

import tokenizer

import json

path_of_testing_img_folder = sys.argv[1]
path_of_output_json_file = sys.argv[2]
path_of_decoder_weights = sys.argv[3]


class Config(object):
    def __init__(self, dir):

        self.max_position_embeddings = 30

        self.dir = dir

        self.peft = "adapter"

def beam_search(imgs):

    vocab_size = 50257

    k = 7

    max_length = 30

    k_prev_words = torch.full((k, 1), 50256, dtype=torch.long).to(device) # (k, 1)

    seqs = k_prev_words #(k, 1)

    top_k_scores = torch.ones(k, 1)

    complete_seqs = list()
    complete_seqs_scores = list()

    step = 1

    while True:
        
        with torch.no_grad():
            
            outputs = transformer(imgs, k_prev_words) # outputs: (k, seq_len, vocab_size)
        
        next_token_logits = outputs[:,-1,:] # (k, vocab_size)
        
        if step == 1:
            next_token_logits = F.softmax(next_token_logits, dim=-1)
            top_k_scores, top_k_words = next_token_logits[0].topk(k, dim=0, largest=True, sorted=True)
        
        else:
            next_token_logits = F.softmax(next_token_logits, dim=-1)
            cumulative_probs = top_k_scores.unsqueeze(1) * next_token_logits  # (k, vocab_size)
            cumulative_log_probs = cumulative_probs.view(-1)
            top_k_scores, top_k_words = cumulative_log_probs.topk(k, dim=0, largest=True, sorted=True)
        
        prev_word_inds = top_k_words // vocab_size  
        next_word_inds = top_k_words % vocab_size  
        
        # seqs: (k, step) ==> (k, step+1)
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != 50256]

        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
 
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds]) 

        k -= len(complete_inds) 
  
        if k == 0: 
           break
  
        seqs = seqs[incomplete_inds]
        
        top_k_scores = top_k_scores[incomplete_inds]   

        k_prev_words = seqs
       
        if step > max_length: 
            complete_seqs.extend(seqs.tolist()) 
            complete_seqs_scores.extend(top_k_scores) 
            break

        step += 1
    
    i = complete_seqs_scores.index(max(complete_seqs_scores)) 

    seq = complete_seqs[i] 

    caption = seq[1:-1]

    T = tokenizer.BPETokenizer('encoder.json', 'vocab.bpe') 

    return T.decode(caption)
    


device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 612
torch.manual_seed(seed)
np.random.seed(seed)

config = Config(dir = path_of_testing_img_folder)

transformer = p2_model.Transformer(peft = config.peft, path_of_decoder_weights = path_of_decoder_weights).to(device)
transformer.load_state_dict(torch.load("freeze_large_clip_model_adapter_lr1e4_step30_dm_7.pth"), strict = False)

print(transformer)

batch_size = 1

dataset_val = p2_dataset.build_dataset(config, mode='validation')
data_loader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=4, shuffle = True)

res_dict = {}

transformer.eval()

for batch in tqdm(data_loader_val):
    
    imgs, file_name = batch

    file_name = file_name[0]

    imgs = imgs.to(device)

    caption = beam_search(imgs)
    
    res_dict[file_name.split(".")[0]] = caption


json_data = json.dumps(res_dict, indent=2)

with open(path_of_output_json_file, 'w') as file:
    file.write(json_data)


 
