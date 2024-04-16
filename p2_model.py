import timm
import torch.nn as nn
import torch
import loralib as lora

import decoder

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.model_name = "vit_large_patch14_clip_224" # "vit_large_patch16_224"
        
        self.encoder = timm.create_model(self.model_name, pretrained=True)

        self.linear_layer = nn.Linear(1024, 768)
        
    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.linear_layer.parameters():
            param.requires_grad = True

    def forward(self, x):

        x = self.encoder.forward_features(x)

        if(self.model_name == "vit_large_patch16_224" or self.model_name == "vit_large_patch14_clip_224" or self.model_name == "deit3_large_patch16_384_in21ft1k" or self.model_name == "vit_large_patch14_clip_224.openai_ft_in12k_in1k"):
            x = self.linear_layer(x)

        return x

class Decoder(nn.Module):

    def __init__(self, peft = "", path_of_decoder_weights = None):
        super().__init__()

        self.config = decoder.Config(checkpoint = path_of_decoder_weights, peft = peft)

        self.decoder = decoder.Decoder(self.config)

    def freeze(self):

        for param in self.decoder.parameters():
            param.requires_grad = False

        for _, param in enumerate(self.decoder.named_parameters()):
            
            if( "cross_attn" in param[0]):
                param[1].requires_grad = True
                

    def freeze_lora(self):

        lora.mark_only_lora_as_trainable(self.decoder)

        for _, param in enumerate(self.decoder.named_parameters()):
            
            # TODO: 加有的沒的！！
            if( "cross_attn" in param[0]):
                param[1].requires_grad = True
    
    def freeze_adapter(self):

        for param in self.decoder.parameters():
            param.requires_grad = False

        for _, param in enumerate(self.decoder.named_parameters()):
            
            if( "adapter" in param[0] ):
                param[1].requires_grad = True
            
            if( "cross_attn" in param[0] ):
                param[1].requires_grad = True
    
    def freeze_prefix_tuning(self):

        for param in self.decoder.parameters():
            param.requires_grad = False

        for _, param in enumerate(self.decoder.named_parameters()):
            
            if( "cross_attn" in param[0]):
                param[1].requires_grad = True
            
            if( "pe" in param[0]):
                param[1].requires_grad = True


    def forward(self, x, encoder_embd):

        x = self.decoder(x, encoder_embd) 

        return x
    
    def getDecoder(self):
        return self.decoder

class Transformer(nn.Module):

    def __init__(self, peft = "", path_of_decoder_weights = None):
        super().__init__()

        self.peft = peft

        self.encoder = Encoder()

        self.decoder = Decoder(peft = self.peft, path_of_decoder_weights = path_of_decoder_weights)
    
    def freeze(self):

        self.encoder.freeze()

        # self.decoder.freeze()

        if self.peft == "lora":
            self.decoder.freeze_lora()
        
        elif self.peft == "adapter":
            self.decoder.freeze_adapter()
        
        elif self.peft == "prefix_tuning":
            self.decoder.freeze_prefix_tuning()
        
        else:
            self.decoder.freeze()
        

    def forward(self, images, captions):

        encoder_embd = self.encoder(images)

        logits = self.decoder(captions, encoder_embd)

        return logits
    
    def getDecoder(self):
        return self.decoder.getDecoder()
