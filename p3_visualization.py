import os
import torch
from PIL import Image
import numpy as np
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
import torch.nn.functional as F

import p2_model 
import p2_dataset
import tokenizer

from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def visualization(attention, ids, feature_size, image_fn, image_path):
    
    print(len(ids))

    if len(ids) % 5 == 0:
        nrows = len(ids) // 5 
    else:
        nrows = len(ids) // 5 + 1
    ncols = 5

    _, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 8))
      
    for idx, text in enumerate(ids):

        im = Image.open(image_path)
        
        attn_vector = attention[:, idx - 1, 1:]
        attn_map = torch.reshape(attn_vector, (16, 16))
        
        m = np.uint8(resize(attn_map.unsqueeze(0), [im.size[1], im.size[0]]).squeeze(0) * 255)

        row = idx // 5
        col = idx % 5

        ax[row][col].imshow(im)

        if idx == 0 or idx == len(text) - 1:
            ax[row][col].set_title('<|endoftext|>')
            if idx == 0:
                continue
        else:
            ax[row][col].set_title(text)
        
        ax[row][col].imshow(m, alpha=0.7, cmap='jet')
        ax[row][col].axis('off')
    
    plt.savefig(image_fn)


class Config(object):
    def __init__(self):

        self.max_position_embeddings = 30

        self.dir = '/home/yuhongzhou/Desktop/dlcv-fall-2023-hw3-AnderStudio/hw3_data/p2_data'

        self.limit = -1

        self.peft = "adapter"


def greedy_search(imgs, transformer, predict_len):
    
    caption = torch.zeros((1, predict_len), dtype=torch.long).to(device)

    caption[:, 0] = 50256

    early_stop = False

    for i in range(predict_len - 1):
        
        predictions = transformer(imgs, caption)

        predictions = predictions[:, i, :]

        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 50256:
            early_stop = True
            caption = caption[0,1:i+1]
            break

        caption[:, i+1] = predicted_id[0] 
    
    if(early_stop == False):
        caption = caption[0,1:]
    
    T = tokenizer.BPETokenizer('encoder.json', 'vocab.bpe') 

    caption = caption.tolist()
    
    return T.decode(caption)

def beam_search(imgs, transformer):

    vocab_size = 50257

    k = 3

    max_length = 65
    
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

def register_attention_hook(model, features, feature_sizes):
    def hook_decoder(module, ins, outs):
        print(module.att.size())
        features.append(torch.sum(module.att, dim=1).detach().cpu())
    handle_decoder = model.decoder.decoder.block_layer[-2].cross_attn.register_forward_hook(
        hook_decoder)
    return [handle_decoder]

seed = 612
torch.manual_seed(seed)
np.random.seed(seed)

config = Config()

dataset_val = p2_dataset.build_dataset(config, mode='visualization')
dataset_val = DataLoader(dataset_val, batch_size=1, num_workers=4, shuffle = True)

transformer = p2_model.Transformer(peft = config.peft).to(device)
transformer.load_state_dict(torch.load("freeze_large_clip_model_adapter_lr1e4_step30_dm_7.pth"), strict = False)

output_dir = './p3_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


transformer.eval()

for data, name in dataset_val:
    if name[0] not in ['000000179758.jpg', '000000006393.jpg']:
        continue
    features, feature_sizes = [], []
    to_rm_l = register_attention_hook(transformer, features, feature_sizes)
    output_ids = greedy_search(data.to(device), transformer, 20)
    output_tokens = ['<|endoftext|>'] + output_ids.split(" ") + ['<|endoftext|>']

    # visualize
    attention_matrix = features[-1]
    image_path = os.path.join(config.dir, 'images', 'val', name[0])
    output_path = os.path.join(output_dir, name[0])
    visualization(attention_matrix, output_tokens, feature_sizes, output_path, image_path)

    for handle in to_rm_l:
        handle.remove()