# current implementation: only support numerical values
from functools import partial
from tkinter import E

import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from timm.models.vision_transformer import Block
from utils import MaskEmbed, get_1d_sincos_pos_embed, ActiveEmbed
eps = 1e-6

from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

from torch.jit import Final

from timm.layers import Mlp, DropPath, use_fused_attn

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # Extend dimensions for broadcasting
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MaskedAutoencoder(nn.Module):
    
    """ Masked Autoencoder with Transformer backbone
    """
    
    def __init__(self, rec_len=25, embed_dim=64, depth=4, num_heads=4,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_field_loss=False, encode_func='linear'):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        if encode_func == 'active':
            self.mask_embed = ActiveEmbed(rec_len, embed_dim)
        else:
            self.mask_embed = MaskEmbed(rec_len, embed_dim)
        
        self.rec_len = rec_len
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pad_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, rec_len + 1, embed_dim), requires_grad=False)  
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, rec_len + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 1, bias=True)  # decoder to patch
        
        # --------------------------------------------------------------------------

        self.norm_field_loss = norm_field_loss
        self.initialize_weights()


    def initialize_weights(self):
        
        # initialization

        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.mask_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def random_masking(self, x, m, mask_ratio, exclude_columns=[]):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        if self.training:
            # Calculate the effective length excluding missing values
            effective_lengths = (m > eps).sum(dim=1).float()
            len_keep = torch.ceil(effective_lengths * (1 - mask_ratio)).long()
        else:
            len_keep = torch.ceil(torch.sum(m, dim=1)).long()

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        noise[m < eps] = 1

        # Exclude specific columns from masking
        if exclude_columns is not None and len(exclude_columns) > 0:
            noise[:, exclude_columns] = 0

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset as per len_keep
        ids_keep_list = [ids_shuffle[i, :len_keep[i]] for i in range(N)]
        max_len_keep = len_keep.max().item()
        ids_keep_padded = [torch.cat([ids_keep_list[i], torch.full((max_len_keep - len_keep[i],), L, device=x.device)], dim=0) for i in range(N)]
        ids_keep = torch.stack(ids_keep_padded, dim=0)
        
        x_padded = torch.cat([x, torch.zeros((N, 1, D), device=x.device)], dim=1)
        x_masked = torch.gather(x_padded, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        for i in range(N):
            mask[i, :len_keep[i]] = 0
            
        mask = torch.gather(mask, dim=1, index=ids_restore)
        nask = torch.ones([N, L], device=x.device) - mask

        if self.training:
            mask[m < eps] = 0
        
        # Create attention mask
        attn_mask = torch.zeros(N, max_len_keep, device=x.device)
        for i in range(N):
            attn_mask[i, :len_keep[i]] = 1
            
        return x_masked, mask, nask, ids_restore, attn_mask


    def forward_encoder(self, x, m, mask_ratio=0.5, exclude_columns=[]):
        
        # embed patches
        x = self.mask_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]    
        
        #print(x.shape)
        
        # masking: length -> length * mask_ratio
        #x, mask, nask, ids_restore = self.random_masking(x, m, mask_ratio, exclude_columns)
        x, mask, nask, ids_restore, attn_mask = self.random_masking(x, m, mask_ratio, exclude_columns)
        
        #print(x.shape)
        #print(attn_mask.shape)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        attn_mask = torch.cat((torch.ones(attn_mask.shape[0], 1, device=x.device), attn_mask), dim=1)
        
        #print(attn_mask.shape)
        #print(attn_mask)
        #print(x.shape)
        #print(x)
        
        # apply Transformer blocks
        for blk in self.blocks:
            #x = blk(x)
            x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)

        return x, mask, nask, ids_restore, attn_mask


    def forward_decoder(self, x, ids_restore):
        
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        # x = self.decoder_pred(x)
        x = torch.tanh(self.decoder_pred(x))/2 + 0.5

        # remove cls token
        x = x[:, 1:, :]
    
        return x


    def forward_loss(self, data, pred, mask, nask):
        """
        data: [N, 1, L]
        pred: [N, L]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # target = self.patchify(data)
        target = data.squeeze(dim=1)
        N, L = target.shape
        
        if self.norm_field_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + eps)**.5
        
        loss = (pred.squeeze(dim=2) - target) ** 2 
        loss = (loss * mask).sum() / mask.sum()  + (loss * nask).sum() / nask.sum()
        
        # mean loss on removed patches
        return loss

    
    def forward(self, data, miss_idx, mask_ratio=0.5, exclude_columns=[]):
        
        latent, mask, nask, ids_restore, _ = self.forward_encoder(data, miss_idx, mask_ratio, exclude_columns)
        pred = self.forward_decoder(latent, ids_restore) 
        loss = self.forward_loss(data, pred, mask, nask)
        return loss, pred, mask, nask

    def extract_embeddings(self, data, miss_idx, mask_ratio=0.0, exclude_columns=[]):
        """
        Extract embeddings from the encoder.
        """
        # Forward pass through the encoder
        latent, mask, nask, ids_restore, attn_mask = self.forward_encoder(data, miss_idx, mask_ratio, exclude_columns)
        
        latent = latent[:, 1:, :]  # Remove the CLS token
        
        # Return the latent representation (embeddings)
        return latent
    
    def extract_embeddings_with_nan(self, data, miss_idx, mask_ratio=0.0, exclude_columns=[]):
        """
        Extract embeddings from the encoder, with missing positions replaced by NaN.
        """
        # Forward pass through the encoder to get latent embeddings and the restore indices
        latent, mask, nask, ids_restore, attn_mask = self.forward_encoder(data, miss_idx, mask_ratio, exclude_columns)
        
        # append mask tokens to sequence
        mask_embed = torch.zeros(1, 1, latent.shape[-1], device=latent.device)
        mask_tokens = mask_embed.repeat(latent.shape[0], ids_restore.shape[1] + 1 - latent.shape[1], 1)

        x_ = torch.cat([latent[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, latent.shape[2]))  # unshuffle

        # Use miss_idx to replace the missing values in the latent embeddings with NaN
        x = x.squeeze(dim=1)
        # convert miss_idx to boolean mask or to tensor if is not
        if isinstance(miss_idx, torch.Tensor):
            miss_idx = miss_idx.bool()
        else:
            miss_idx = torch.tensor(miss_idx, dtype=torch.bool)
        
        # Replace the missing values with NaN (missing values are False in miss_idx)
        x[~miss_idx] = np.nan
        
        # Return the latent representation (embeddings) with missing values replaced by NaN
        return x
        
def mae_base(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=16, depth=4, num_heads=4,
        decoder_embed_dim=16, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=2., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model


def mae_medium(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=32, depth=4, num_heads=4,
        decoder_embed_dim=32, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model


def mae_large(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=64, depth=8, num_heads=8,
        decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model
    

if __name__ == '__main__':

    model = MaskedAutoencoder(
        rec_len=4, embed_dim=8, depth=1, num_heads=1,
        decoder_embed_dim=8, decoder_depth=1, decoder_num_heads=1,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=eps)
    )
    
    X = pd.DataFrame([[np.nan, 0.5, np.nan, 0.8]])

    X = torch.tensor(X.values, dtype=torch.float32)
    M = 1 - (1 * (np.isnan(X)))
    X = torch.nan_to_num(X)
    
    X = X.unsqueeze(dim=1)
    print(model.forward(X, M, 0.75))
