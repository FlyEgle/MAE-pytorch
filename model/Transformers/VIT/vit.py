""" Vision Transformer (ViT) in PyTorch
"""
import sys 
import math
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Transformers.VIT.layers.patch_embd import PatchEmbed, PositionEmbed
from model.Transformers.VIT.layers.mlp import Mlp
from model.Transformers.VIT.layers.drop import DropPath
from model.Transformers.VIT.layers.weight_init import trunc_normal_
from model.Transformers.VIT.utils.mask_embeeding import MaskEmbeeding, UnMaskEmbeeding


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=PatchEmbed, pos_embed="cosine", norm_layer=nn.LayerNorm, act_layer=nn.GELU, pool='mean',
                 classification=False, vit_type="encoder", mask_ratio=0.75, MAE=True
                 
                 ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            pos_embed (nn.Module): position embeeding layer cosine or learnable parameters
            norm_layer: (nn.Module): normalization layer
            pool: 'mean' or 'cls' for classification
            classification: True or False 
            vit_type: "encoder" or "decoder" for MAE
            mask_ratio: a ratio for mask patch numbers
            MAE: Use MAE for trainig 
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1  
        self.classification = classification 
        self.mask_ratio = mask_ratio 
        self.vit_type = vit_type 
        self.MAE = MAE 
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
    
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if pos_embed == "cosine":
            self.pos_embed = PositionEmbed(num_patches, embed_dim, self.num_tokens)()
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
       
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.vit_type == "decoder":
            self.unmask_embed = UnMaskEmbeeding(img_size, 
                                           embed_dim,
                                           in_chans,
                                           patch_size,
                                           num_patches
                                           )
        
        # for MAE dropout is not use
        if self.MAE:
            dpr = [0.0 for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.pool = pool
        
        if self.classification:
            self.class_head = nn.Linear(self.num_features, self.num_classes)
        
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, module):
        """ ViT weight initialization
        """
        if isinstance(module, nn.Linear):
            if module.out_features == self.num_classes:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def autoencoder(self, x):
        """encoder the no mask patch embeeding with position embeeding
        Returns:
            norm_embeeding: encoder embeeding 
            sample_index:   a list of token used for encoder
            mask_index      a list of token mask 
        """

        x = self.patch_embed(x)
        # add cls token for classification
        dummpy_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((dummpy_token, x), dim=1) 
        x = x + self.pos_embed
        
        # mask the patchemb&posemb
        mask_patch_embeeding, sample_index, mask_index = MaskEmbeeding(x, self.mask_ratio)
        
        x = self.blocks(mask_patch_embeeding)
        norm_embeeding = self.norm(x)
        return norm_embeeding, sample_index, mask_index
    
    
    def decoder(self, x, sample_index, mask_index):
        """decoder the all patch embeeding with the mask and position embeeding 
        """
        # unmask the patch embeeidng with the encoder embeeding 
        decoder_embed = self.unmask_embed(x, sample_index, mask_index)
        x = decoder_embed + self.pos_embed 
        
        # decoder
        decoder_embeeding = self.blocks(x)
        return decoder_embeeding
    
    
    def forward_features(self, x):
        """Return the layernormalization features
        """
        x = self.patch_embed(x)
        # add cls token for classification
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        
        return x 

    def forward(self, x):
        x = self.forward_features(x)
        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "cls":
            x = x[:, 0]  # cls token
        else:
            raise ValueError("pool must be 'cls' or 'mean' ")
        
        assert x.shape[1] == self.num_features, "outputs must be same with the features"
        if self.classification:
            x = self.class_head(x)
        return x



def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = VisionTransformer(img_size=224, 
                              in_chans=3,
                              num_classes=1000,
                              mlp_ratio=4., 
                              qkv_bias=True,
                              embed_layer=PatchEmbed, 
                              pos_embed="cosine", 
                              norm_layer=nn.LayerNorm, 
                              act_layer=nn.GELU, 
                              pool='mean',
                              **model_kwargs
                              )
    return model


def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = VisionTransformer(img_size=224, 
                              in_chans=3,
                              num_classes=1000,
                              mlp_ratio=4., 
                              qkv_bias=True,
                              embed_layer=PatchEmbed, 
                              pos_embed="cosine", 
                              norm_layer=nn.LayerNorm, 
                              act_layer=nn.GELU, 
                              pool='mean',
                              **model_kwargs
                              )
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = VisionTransformer(img_size=224, 
                              in_chans=3,
                              num_classes=1000,
                              mlp_ratio=4., 
                              qkv_bias=True,
                              embed_layer=PatchEmbed, 
                              pos_embed="cosine", 
                              norm_layer=nn.LayerNorm, 
                              act_layer=nn.GELU, 
                              pool='mean',
                              **model_kwargs
                              )
    return model


def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = VisionTransformer(img_size=224, 
                              in_chans=3,
                              num_classes=1000,
                              mlp_ratio=4., 
                              qkv_bias=True,
                              embed_layer=PatchEmbed, 
                              pos_embed="cosine", 
                              norm_layer=nn.LayerNorm, 
                              act_layer=nn.GELU, 
                              pool='mean',
                              **model_kwargs
                              )
    return model


def vit_large_patch16_224_decoder(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=512, depth=8, num_heads=16, **kwargs)
    model = VisionTransformer(img_size=224, 
                              in_chans=3,
                              num_classes=1000,
                              mlp_ratio=4., 
                              qkv_bias=True,
                              embed_layer=PatchEmbed, 
                              pos_embed="cosine", 
                              norm_layer=nn.LayerNorm, 
                              act_layer=nn.GELU, 
                              pool='mean',
                              **model_kwargs
                              )
    return model



if __name__ == '__main__':
    model = vit_large_patch16_224()
    print(model)
    inputs = torch.randn(1, 3, 224, 224)
    outputs = model(inputs)
    print(outputs.shape)
    