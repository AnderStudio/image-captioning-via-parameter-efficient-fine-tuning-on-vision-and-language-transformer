import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv
from typing import Optional, List
from torch import Tensor
from PIL import Image
import numpy as np
import random
import os
import json

import tokenizer

MAX_DIM = 224 

def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = [3, MAX_DIM, MAX_DIM]
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def under_max(image): 
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float64)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image

class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class TrainImageCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform):
        super().__init__()

        self.root = root
        self.transform = transform

        self.ann = ann

        self.annot = [(self._process(val['image_id']), val['caption'])
                      for val in ann['annotations']]

        self.annot = self.annot[: limit]

        self.tokenizer = tokenizer.BPETokenizer('encoder.json', 'vocab.bpe')

        self.max_length = 64

    def _process(self, image_id):

        images = self.ann["images"]

        for image in images:
            if image["id"] == image_id:
                return image["file_name"]

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        
        image_id, caption = self.annot[idx]

        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption = self.tokenizer.encode(caption)

        caption = [50256] + caption + [50256]

        model_input_caption = caption
        cross_entropy_caption = caption
        
        model_input_caption += [50256] * (self.max_length - len(caption))

        cross_entropy_caption += [-100] * (self.max_length - len(caption))

        model_input_caption = torch.tensor(model_input_caption, dtype=torch.long)
        cross_entropy_caption = torch.tensor(cross_entropy_caption, dtype=torch.long)

        return image.tensors.squeeze(0), image.mask.squeeze(0), model_input_caption, cross_entropy_caption, image_id


class ValImageCaption(Dataset):
    def __init__(self, root, transform):
        super().__init__()

        self.root = root

        self.transform = transform

        self.annot = [x for x in os.listdir(self.root) if x.endswith(".jpg")]

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        
        image_id = self.annot[idx]

        image = Image.open(os.path.join(self.root, image_id))

        image = self.transform(image)

        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        return image.tensors.squeeze(0), image_id

class VisualImageCaption(Dataset):
    def __init__(self, root, max_length, limit, transform=train_transform):
        super().__init__()

        print("hello")

        self.root = root
        self.transform = transform

        self.annot = os.listdir(self.root)

        self.tokenizer = tokenizer.BPETokenizer('encoder.json', 'vocab.bpe')

        self.max_length = 64

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        
        image_id = self.annot[idx]

        image = Image.open(os.path.join(self.root, image_id)) #TODO: 直接讀檔

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        return image.tensors.squeeze(0), image_id

def build_dataset(config, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'images', 'train')
        train_file = os.path.join(
            config.dir, 'train.json')
        data = TrainImageCaption(train_dir, read_json(
            train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform)
        return data

    elif mode == 'validation':
        data = ValImageCaption(root = config.dir, transform = val_transform)
        return data
    
    elif mode == 'visualization':
        val_dir = os.path.join(config.dir, 'images', 'val')
        data = VisualImageCaption(val_dir, max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")


