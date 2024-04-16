import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora

class Config:

    def __init__(self, checkpoint=None, peft=""):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.peft = peft
        self.scale_down_of_n_embd = 4
        self.prefix_tuning_length = 64

class Adapter(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.down_proj = nn.Linear(cfg.n_embd, cfg.n_embd // cfg.scale_down_of_n_embd)

        self.non_linear = nn.GELU(approximate='tanh')

        self.up_proj = nn.Linear(cfg.n_embd // cfg.scale_down_of_n_embd, cfg.n_embd)
    
    def forward(self, decoder_embd):
        
        original_decoder_embd = decoder_embd

        decoder_embd = self.up_proj(self.non_linear(self.down_proj(decoder_embd)))

        return decoder_embd + original_decoder_embd

class CrossAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)

        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

        self.att = None

    def forward(self, encoder_embd, decoder_embd):
        
        B_dec, T_dec, C_dec = decoder_embd.size() # batch, context, embedding
        B_enc, T_enc, C_enc = encoder_embd.size() # batch, image, embedding

        q_dec, k_dec, v_dec = self.c_attn(decoder_embd).split(self.n_embd, dim=2)
        q_enc, k_enc, v_enc = self.c_attn(encoder_embd).split(self.n_embd, dim=2)

        k_enc = k_enc.view(B_enc, T_enc, self.n_head, C_enc // self.n_head).transpose(1, 2)
        q_dec = q_dec.view(B_dec, T_dec, self.n_head, C_dec // self.n_head).transpose(1, 2)
        v_enc = v_enc.view(B_enc, T_enc, self.n_head, C_enc // self.n_head).transpose(1, 2)

        att = (q_dec @ k_enc.transpose(-2, -1)) * (1.0 / math.sqrt(k_enc.size(-1)))
        
        att = F.softmax(att, dim=-1)
        
        self.att = att 

        return self.c_proj((att @ v_enc).transpose(1, 2).contiguous().view(B_dec, T_dec, C_dec))

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        if cfg.peft == "lora":
            self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=4)
        else:
            self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
         
        if cfg.peft == "lora":
            self.c_proj = lora.Linear(cfg.n_embd,  cfg.n_embd, r=4)
        else:
            self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        if cfg.peft == "prefix_tuning":
            self.prefix_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size

        self.cfg = cfg

        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x, prefix_embd = None):
        
        B, T, C = x.size() # batch, context, embedding

        if self.cfg.peft == "prefix_tuning":
            B_prefix, T_prefix, C_prefix = prefix_embd.size()
            q_prefix, k_prefix, v_prefix = self.prefix_attn(prefix_embd).split(self.n_embd, dim=2)
            k_prefix = k_prefix.view(B_prefix, T_prefix, self.n_head, C_prefix // self.n_head).transpose(1, 2)
            v_prefix = v_prefix.view(B_prefix, T_prefix, self.n_head, C_prefix // self.n_head).transpose(1, 2)

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        if self.cfg.peft == "prefix_tuning":

            att_prefix = (q @ k_prefix.transpose(-2, -1)) * (1.0 / math.sqrt(k_prefix.size(-1)))
            att_prefix = att_prefix.masked_fill(self.bias[:,:,:T,:T_prefix] == 0, float('-inf'))
            att_prefix = F.softmax(att_prefix, dim=-1)

            return self.c_proj(((att_prefix @ v_prefix) + (att @ v)).transpose(1, 2).contiguous().view(B, T, C))
        
        else:

            return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cross_attn = CrossAttention(cfg)
        
        if cfg.peft == "lora":
            self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=4)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=4))
            ]))
        else:
            self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
            ]))
        
        self.adapter = Adapter(cfg)
        

    def forward(self, x, encoder_embd, prefix_embd):
        
        if self.cfg.peft == "prefix_tuning":
            x = x + self.attn(self.ln_1(x), prefix_embd)
        else:
            x = x + self.attn(self.ln_1(x))
        
        x = x + self.cross_attn(encoder_embd, self.ln_2(x))

        if self.cfg.peft == "adapter":
            x = x + self.adapter(self.mlp(self.ln_3(x)))
        else:
            x = x + self.mlp(self.ln_3(x))

        return x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        print(f'PEFT type of decoder: {self.cfg.peft}')
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.pe = nn.Embedding(self.cfg.prefix_tuning_length, cfg.n_embd)
        
        self.transformer.wte.weight = self.lm_head.weight

        self.block_layer = list(self.transformer.h.children())

        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()

            self.transformer.load_state_dict(state_dict, strict=False)


    def forward(self, x: Tensor, encoder_embd: Tensor):
        
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)

        prefix = torch.arange(self.cfg.prefix_tuning_length, dtype=torch.long, device=x.device)
        prefix = prefix.squeeze(0).repeat(x.size()[0], 1)
        prefix = self.pe(prefix)

        x = self.transformer.wte(x) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            x = block(x, encoder_embd, prefix) 
        
        x = self.lm_head(self.transformer.ln_f(x))

        return x
