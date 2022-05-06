""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020, Ross Wightman 
Modified by Bhishma Dedhia
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_, DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }




class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, prune = False, retain_rate = 0.8, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., tau_imp = 0.5):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.prune = prune
        self.retain_rate = retain_rate
        self.tau_imp = tau_imp
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, token_mask, memory_token, eps = 1e-6):
        B, N, C = x.shape

        qkv = self.qkv(torch.cat((x,memory_token),dim=1)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q = q[:,:,:-1,:] #omitting memory token query
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.training and self.prune:
            true_token = torch.ones(B,1).to(token_mask.device)
            token_mask_all = torch.cat((true_token,token_mask),1)
            # attn = attn + (1-token_mask_all)[:,None,None,:]*(-10e5)
            importance = self.get_importance(attn[:,:,:-1], token_mask_all, self.tau_imp)#tokenrank between tokens
            # eye = torch.eye(N)[None,None,:,:].to(x.device)
            attn = attn.exp()*token_mask_all[:,None,None,:]
            #stabilize training
            attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
            attn = self.attn_drop(attn)

            # rel_imp = importance/torch.sum(importance,dim= -1,keepdim=True).detach()
            #next_mask = torch.sigmoid((rel_imp - self.pruning_threshold)/self.tau_prune)
            all_index = torch.cat((torch.arange(N-1)[None,:],)*B,0).to(token_mask.device)
            selected = all_index[token_mask.bool()].view(B,-1)
            _, inds = torch.sort(importance[token_mask.bool()].view(B,-1),dim=-1,descending=True)
            inds = inds[:,:int(inds.shape[1]*self.retain_rate)]
            imp_inds = selected.gather(1,inds)
            # print(selected[0])
            # print(imp_inds[0])
            next_mask = torch.zeros((B,N-1)).to(token_mask.device)
            next_mask = next_mask.scatter_(1,imp_inds,1)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, next_mask, importance


        elif self.training and not(self.prune):
            true_token = torch.ones(B,1).to(token_mask.device)
            token_mask_all = torch.cat((true_token,token_mask),1)
            eye = torch.eye(N)[None,None,:,:].to(x.device)
            attn = attn.exp()*token_mask_all[:,None,None,:]
            #stabilize training
            attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            next_mask = torch.ones(B,N-1).to(token_mask.device)
            return x, next_mask

        elif not(self.training) and not(self.prune):
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            next_mask = torch.ones(B,N-1).to(token_mask.device).bool()
            return x, next_mask

        else:
            importance = self.get_importance(attn[:,:,:,:-1], None, self.tau_imp)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            selected = torch.cat((torch.arange(N-1)[None,:],)*B,0).to(token_mask.device)
            _, inds = torch.sort(importance,dim=-1,descending=True)
            inds = inds[:,:int(inds.shape[1]*self.retain_rate)]
            imp_inds = selected.gather(1,inds)
            next_mask = torch.zeros((B,N-1)).to(token_mask.device)
            next_mask = next_mask.scatter_(1,imp_inds,1)
            return x, next_mask, importance

    def get_importance(self, attention_score, token_mask, tau_imp=0.1, eps = 1e-6, d=0.):
        B,_,N,_ = attention_score.shape
        attn_mean = torch.mean(attention_score/tau_imp,dim=1)
        if token_mask is not None:
            attn = attn_mean.exp()*token_mask[:,None,:]
        else:
            attn = attn_mean.exp()
        M = attn/ attn.sum(dim=-1, keepdim=True) 
        dist  = (torch.ones(B,1,N)/N).to(M.device)
        dist = dist@M@M@M@M@M
        # v,e = torch.linalg.eig(M)
        # importance = torch.real(e[:,1:,0])
        #ssd_normalize = torch.abs(normalize(ssd,dim=1,p=1))
        importance  = dist.view(B,N) 
        return importance[:,1:]



class Block(nn.Module):

    def __init__(self, dim, num_heads, prune = False, retain_rate = 0.8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, tau_imp = 0.5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.prune = prune
        self.attn = Attention(dim, num_heads=num_heads, prune = prune, retain_rate = retain_rate, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, tau_imp = tau_imp)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.retain_rate = retain_rate
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, memory_token, token_mask=None):
        B,N,C = x.shape
        pruned_tokens = None
        if self.prune:
            y, next_mask, importance= self.attn(self.norm1(x), token_mask)
        else: 
            importance = None
            y, next_mask = self.attn(self.norm1(x),token_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.prune:
            pruned_tokens = x[:,1:][(1-next_mask).bool()].view(B,-1,C)

        if not(self.training):
            modified =  x[:,1:][next_mask.bool()].view(B,-1,C)
            x = torch.cat((x[:,0].view(B,1,-1),modified),1)
        return x, next_mask, importance, pruned_tokens


class TokenRankVisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, prune_list, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distill_model = True, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, retain_rate = 0.8, tau_imp = 0.5, weight_init=''):
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
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.prune_list = prune_list
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.memory_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks =  nn.ModuleList([
            Block(
                prune = True, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, 
                tau_imp = tau_imp, retain_rate = retain_rate) if i in prune_list else 

            Block(
                prune = False, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, 
                tau_imp = tau_imp,retain_rate = retain_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        self.memory_kv = nn.Linear(embed_dim,2*embed_dim)
        self.memory_q = nn.Linear(embed_dim,embed_dim)


        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def update_memory(self, memory_token, pruned_tokens):
        q = self.memory_q(memory_token)
        kv = self.memory_kv(torch.cat((memory_token,pruned_tokens),1))
        k,v = kv.unbind(0)
        att = q@k.transpose(-2,-1)*((q.shape[-1])**(-0.5))
        att = att.softmax(-1)
        memory_token = att@v
        return memory_token

    def forward(self, x):
        x = self.patch_embed(x)
        B,N,_ = x.shape
        token_mask = torch.ones(B,N).to(x.device)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        memory_token = self.memory_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        all_importance  = ()
        all_token_mask = ()
        all_features = ()
        for i, block in enumerate(self.blocks):
            if self.training:
                x[:,1:] = x[:,1:]*token_mask[:,:,None]
            x, next_mask, importance, pruned_tokens = block(x, memory_token, token_mask)
            if self.prune:
                memory_token = self.update_memory(memory_token,pruned_tokens)
            if self.training:
                token_mask = token_mask*next_mask
                all_features = all_features +  (x[:,1:],)
                all_token_mask = all_token_mask+(token_mask.detach(),)
                
            else:
                token_mask = next_mask


        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)
        if self.training:
            return x, all_features, all_token_mask
        else:
            return x



def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)




