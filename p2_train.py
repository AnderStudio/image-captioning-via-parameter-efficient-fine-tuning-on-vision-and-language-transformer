import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable 

import numpy as np
import sys
import os

import p2_model 
import p2_dataset

from tqdm import tqdm
import torch.nn.functional as F

import tokenizer
import p2_evaluate as eval

import loralib as lora

from torch.cuda import amp


class Config(object):
    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_position_embeddings = 64

        self.predicted_max_position_embeddings = 30

        self.dir = '/home/yuhongzhou/Desktop/dlcv-fall-2023-hw3-AnderStudio/hw3_data/p2_data'

        self.limit = -1

        self.start_epoch = 0

        self.learning_rate = 1e-4 # 3e-5

        self.weight_decay = 1e-5

        self.epochs = 20

        self.load_model_name = "freeze_large_clip_model_lora_lr1e4_step30"

        self.save_model_name = "freeze_large_clip_openai_model_lora_lr1e4_step30_dm_lt"

        self.training_batch_size = 14

        self.validation_batch_size = 1

        self.annotation_file = "/home/yuhongzhou/Desktop/dlcv-fall-2023-hw3-AnderStudio/hw3_data/p2_data/val.json"

        self.images_root = "/home/yuhongzhou/Desktop/dlcv-fall-2023-hw3-AnderStudio/hw3_data/p2_data/images/val"

        self.amp = False

        self.peft = "lora"
    
config = Config()

print(f'Initializing Device: {config.device}')


seed = 612 
torch.manual_seed(seed)
np.random.seed(seed)

transformer = p2_model.Transformer(peft = config.peft).to(config.device)

if(config.start_epoch != 0):
    transformer.load_state_dict(torch.load(f"{config.load_model_name}_{config.start_epoch}.pth"), strict = False)

transformer.freeze()

print("Total params:", sum(p.numel() for p in transformer.parameters() if p.requires_grad))

optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

dataset_train = p2_dataset.build_dataset(config, mode='training')
dataset_val = p2_dataset.build_dataset(config, mode='validation')

criterion = torch.nn.CrossEntropyLoss()

data_loader_train = DataLoader(dataset_train, batch_size = config.training_batch_size, num_workers=4, shuffle = True)
data_loader_val = DataLoader(dataset_val, batch_size = config.validation_batch_size, num_workers=4, shuffle = True)

scaler = amp.GradScaler(enabled = config.amp)

with open(f"./{config.save_model_name}_log.txt","a") as f:
    f.write(f" [ {config.save_model_name} ] Starting Training !\n")

for epoch in range(config.epochs):
        
    print(f"Epoch: {config.start_epoch + epoch + 1}")

    transformer.train()

    epoch_loss = 0.0
    total = len(data_loader_train)

    for batch in tqdm(data_loader_train):
        
        imgs, masks, model_input_captions, cross_entropy_captions, file_name = batch

        imgs = imgs.to(config.device)
        model_input_captions = model_input_captions.to(config.device)
        cross_entropy_captions = cross_entropy_captions.to(config.device)

        with amp.autocast(enabled=config.amp):
                
            logits = transformer(imgs, model_input_captions[:,:config.max_position_embeddings - 1])
            
            loss = criterion(logits.permute(0, 2, 1), cross_entropy_captions[:,1:])
        
        loss_value = loss.item()
        
        epoch_loss += loss_value

        optimizer.zero_grad()

        if config.amp:

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:
        
            loss.backward()

            optimizer.step()
    
    print(f' [Train] | epoch : {config.start_epoch + epoch + 1} | epoch loss : {epoch_loss / len(data_loader_train)}')

    with open(f"./{config.save_model_name}_log.txt","a") as f:
        f.write(f' [Train] | epoch : {config.start_epoch + epoch + 1} | epoch loss : {epoch_loss / len(data_loader_train)}\n')


    trainable_weights = [name for name, param in transformer.named_parameters() if param.requires_grad == True]

    save_weights = { k : v for k, v in transformer.state_dict().items() if k in trainable_weights}

    torch.save(save_weights, f"./{config.save_model_name}_{config.start_epoch + epoch + 1}.pth")

    # inference 

    print("Total params:", sum(p.numel() for p in transformer.parameters() if p.requires_grad))

    print(f'Inference Device: {config.device}')

    res_dict = {}

    transformer.eval()

    for batch in tqdm(data_loader_val):
    
        imgs, file_name = batch

        file_name = file_name[0]

        imgs = imgs.to(config.device)

        caption = torch.zeros((1, config.predicted_max_position_embeddings), dtype=torch.long).to(config.device)

        caption[:, 0] = 50256

        early_stop = False

        for i in range(config.predicted_max_position_embeddings - 1):
            
            with torch.no_grad():

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
        
        # print(f' [ {file_name} ] : {T.decode(caption)}')
        
        res_dict[file_name.split(".")[0]] = T.decode(caption)
    
    # eval

    # Read data
    predictions = res_dict

    annotations = eval.readJSON(config.annotation_file)

    # Preprocess annotation file
    gts = eval.getGTCaptions(annotations)

    # Check predictions content is correct
    assert type(predictions) is dict
    assert set(predictions.keys()) == set(gts.keys())
    assert all([type(pred) is str for pred in predictions.values()])

    # CIDErScore
    cider_score = eval.CIDERScore()(predictions, gts)

    # CLIPScore
    clip_score = eval.CLIPScore()(predictions, config.images_root)
    
    print(f" [Val] | epoch : {config.start_epoch + epoch + 1} | CIDEr : {cider_score} | CLIPScore : {clip_score}")
    with open(f"./{config.save_model_name}_log.txt","a") as f:
        f.write(f" [Val] | epoch : {config.start_epoch + epoch + 1} | CIDEr : {cider_score} | CLIPScore : {clip_score}\n")




