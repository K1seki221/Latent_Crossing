import copy
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

#-----------------------------
from .tensor_decomp import validate_tt_rank
#from opt_einsum import contract
import yaml
import sympy
from pathlib import Path
contract = torch.einsum
current_file = Path(__file__)
work_path = current_file.parent.parent.parent
config_dir = f'{work_path}/config'

einsum_patterns_8cores = [
    'abcde, fbg -> acdefg',  
    'acdefg, gci -> adefi',   
    'adefi, idj -> aefj',     
    'aefj, jeh -> afh',    
    'afh, hij -> afij',      
    'afij, jkl -> afikl',     
    'afikl, lmn -> afikmn',   
    'afikmn, npq -> afikmpq'  
]

einsum_patterns_6cores = [
    'abcd, bg -> acdg',
    'acdg, gci -> adi',
    'adi, idj -> aj',
    'aj, jkl -> akl',
    'akl, lmn -> akmn',
    'akmn, np -> akmp'
]

einsum_patterns_4cores = [
    'abc, bg -> acg',
    'acg, gcj -> aj',
    'aj, jkl -> akl',
    'akl, ln -> akn',
]
def most_balanced_factor_pair(n: int):
    div_list = sympy.divisors(n)
    a_best, b_best = 1, n
    min_diff = abs(n-1)
    
    for d in div_list:
        comp = n // d
        diff = abs(d - comp)
        if diff < min_diff:
            min_diff = diff
            a_best, b_best = d, comp
    
    return a_best, b_best

def readConfig(config:str):
    if isinstance(config,str):
        if config.endswith("yaml") or config.endswith("yml"):
            with open(config,"r") as f:
                layerconfig=yaml.safe_load(f)
        else:
            raise ValueError("Loading Config File ERROR for TensorizedLinear")
    else:
        raise ValueError("No Config File for TensorizedLinear")
    return layerconfig

__all__ = ['LaxVisionTransformer']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)

class TensorizedLinear(nn.Module):
    def __init__(self, indim=768, outdim=768, middim=[12,8,8,8,8,12], 
                 rank=256,InputShape=[12,8,8], 
                 activation='GELU', InterRes=False, IntraRes=False, 
                 InterResGate=False, IntraResGate=False,TenserizedGate=False, 
                 LatentAct=True, bn=False, fact=False, bias=True):
        super(TensorizedLinear,self).__init__()
        self.bn = None
        if bn is not False:
            self.bn = nn.BatchNorm1d(outdim)
        self.fact = fact
        self.indim = indim
        self.outdim = outdim
        self.middim = middim
        self.InputShape = InputShape
        self.activation = activation
        self.InterRes = InterRes
        self.IntraRes = IntraRes
        self.ranks = validate_tt_rank(self.middim)
        
        if rank is not False:
            self.ranks = validate_tt_rank(self.middim,rank)
        self.LatentAct = LatentAct
        if self.activation == 'ReLU':
            self.act = nn.ReLU()
        elif self.activation == 'GELU':
            self.act = nn.GELU()
        elif self.activation == 'SiLU':
            self.act = nn.SiLU()
        else:
            raise ValueError("No Activation Config for TensorizedLinear")

        #-----Cores-----#
        self.cores = []
        # addflag:
        cores_num = len(self.middim)
        if cores_num == 6:
            for i_core in range(cores_num):
                if i_core == 0: # doing this since there is a gradient-graph issue for the dim that equals to 1
                    core = nn.Parameter(torch.randn(self.middim[i_core], self.ranks[i_core+1]))
                elif i_core==5: # doing this since there is a gradient-graph issue for the dim that equals to 1
                    core = nn.Parameter(torch.randn(self.ranks[i_core], self.middim[i_core]))
                else:
                    core = nn.Parameter(torch.randn(self.ranks[i_core], self.middim[i_core], self.ranks[i_core+1]))
                self.cores.append(core)
            self.cores = nn.ParameterList(self.cores)
        elif cores_num == 4:
            for i_core in range(cores_num):
                if i_core == 0: # doing this since there is a gradient-graph issue for the dim that equals to 1
                    core = nn.Parameter(torch.randn(self.middim[i_core], self.ranks[i_core+1]))
                elif i_core==3: # doing this since there is a gradient-graph issue for the dim that equals to 1
                    core = nn.Parameter(torch.randn(self.ranks[i_core], self.middim[i_core]))
                else:
                    core = nn.Parameter(torch.randn(self.ranks[i_core], self.middim[i_core], self.ranks[i_core+1]))
                self.cores.append(core)
            self.cores = nn.ParameterList(self.cores)
        elif cores_num == 2:
            for i_core in range(cores_num):
                if i_core == 0: # doing this since there is a gradient-graph issue for the dim that equals to 1
                    core = nn.Parameter(torch.randn(self.middim[i_core], rank))
                elif i_core==1: # doing this since there is a gradient-graph issue for the dim that equals to 1
                    core = nn.Parameter(torch.randn(rank, self.middim[i_core]))
                self.cores.append(core)
            self.cores = nn.ParameterList(self.cores)

        
        #-----InterResGate-----#
        self.Inter_resgate = None
        if InterResGate is True:
            if InterRes is False:
                raise ValueError("Initializing ResGate, but InterRes is not enable")
            self.Inter_resgate = nn.ParameterList()
            self.Inter_resgateShape = None
            if TenserizedGate:
                self.TensorizeGate()
            else:
                self.Inter_resgate = nn.Linear(self.ranks[len(self.ranks)//2],self.ranks[len(self.ranks)//2])
                
        #-----IntraResGate-----#        
        self.Intra_resgate = None
        if IntraResGate is True:
            if IntraRes is False:
                raise ValueError("Initializing ResGate, but IntraRes is not enable")
            self.Intra_resgate = nn.ParameterList()
            self.IntraTensorizeGate()
                  
        self.initparam()
        if bias is True:
            self.bias = nn.Parameter(torch.randn(self.outdim))
        else:
            self.bias = None
        if InterRes:
            self.res_layernorm = nn.LayerNorm(normalized_shape=self.ranks[len(self.ranks)//2],eps=1e-6)

    def IntraTensorizeGate(self):
        for i in range(len(self.middim)//2):
            self.Intra_resgate.append(
                nn.Parameter(
                    nn.init.kaiming_uniform_(
                        torch.Tensor(self.middim[i],self.middim[-1-i])
                    )
                )
            )

            
    def TensorizeGate(self):
        self.Inter_resgateShape = most_balanced_factor_pair(self.ranks[len(self.ranks)//2])
        resgate_encoder = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(max(self.Inter_resgateShape),max(self.Inter_resgateShape))))
        resgate_decoder = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(min(self.Inter_resgateShape),min(self.Inter_resgateShape))))
        self.Inter_resgate.append(resgate_encoder)
        self.Inter_resgate.append(resgate_decoder)

    def initparam(self):
        for core in self.cores:
            nn.init.kaiming_normal_(core,nonlinearity='relu')

    def LeftHalfContractionIntra(self,x):
        x = x.reshape(-1,*self.InputShape)
        Intra_Res = []
        if len(self.cores) == 6:
            for lCore in range(3):
                x = torch.einsum(einsum_patterns_6cores[lCore], x, self.cores[lCore])
                if lCore != 2:
                    Intra_Res.append(x)
        if len(self.cores) == 4:
            for lCore in range(2):
                x = torch.einsum(einsum_patterns_4cores[lCore], x, self.cores[lCore])
                if lCore != 1:
                    Intra_Res.append(x)
        return x, Intra_Res
    
    def RightHalfContractionIntra(self,x,Intra_Res):
        Intra_Res.reverse()
        if len(self.cores) == 6:
            for i,rCore in enumerate(range(3,6)):
                x = torch.einsum(einsum_patterns_6cores[rCore], x, self.cores[rCore])
                if rCore != 5:
                    if self.Intra_resgate is not None:
                        x = x + self.IntraResGate_Forward(Intra_Res[i],i).reshape(x.shape)
                    else:    
                        x = x + Intra_Res[i]
        if len(self.cores) == 4:
            for i,rCore in enumerate(range(2,4)):
                x = torch.einsum(einsum_patterns_4cores[rCore], x, self.cores[rCore])
                if rCore != 3:
                    if self.Intra_resgate is not None:
                        x = x + self.IntraResGate_Forward(Intra_Res[i]).reshape(x.shape)
                    else:
                        x = x + Intra_Res[i] 
        return x

    def IntraRes_RightHalfContractionForward(self, x, Intra_Res, res_former):
        if res_former is not None and self.InterRes is not False:
            #print('flag:using interRes Connection')
            if self.Inter_resgate is not None:
                #print('flag:using ResGate')
                x = x + self.ResGateForward(res_former).reshape(x.shape)
            else:
                #print('flag:No ResGate')
                x = x +res_former
            x = self.res_layernorm(x)
        if self.LatentAct:
            #print('flag: LatentAct')
            x = self.act(x)
        inter_res = x
        x = self.RightHalfContractionIntra(x,Intra_Res)
        return x, inter_res

    def IntraRes_LinearContractionForward(self,x,res_former):
        x, Intra_Res = self.LeftHalfContractionIntra(x)
        x, inter_res = self.IntraRes_RightHalfContractionForward(x,Intra_Res,res_former)
        x = x.reshape(-1,self.outdim)
        return x, inter_res
    
    def ResGateForward(self,res):
        if self.Inter_resgateShape is not None:
            x = contract('abcd,de,cf-> abfe', 
                         res.view(-1,1,self.Inter_resgateShape[0],self.Inter_resgateShape[1]), 
                         self.Inter_resgate[0],
                         self.Inter_resgate[1])
        else:
            x = self.Inter_resgate(res)
        return x
    
    def IntraResGate_Forward(self,res):
        res = contract('abc,cd,be->aed',res,self.Intra_resgate[0],self.Intra_resgate[1])
        return res
    def LeftHalfContractionForward(self,x):
        x = x.reshape(-1,*self.InputShape)
        if len(self.cores) == 6:
            x = contract('abcd,bg,gci,idj->aj', x,
                    self.cores[0],self.cores[1],self.cores[2])
        elif len(self.cores) == 4:
            x = contract('abc,bg,gcj->aj', x, self.cores[0],self.cores[1])
        elif len(self.cores) == 2:
            x = contract('ab,bc->ac', x, self.cores[0])
        return x

    def RightHalfContractionForward(self,x):
        if len(self.cores) == 6:
            x = contract('aj,jkl,lmn,np -> akmp',x,
                        self.cores[3],self.cores[4],self.cores[5])
        elif len(self.cores) == 4:
            x = contract('aj,jkl,ln->akn', x, self.cores[2],self.cores[3])
        elif len(self.cores) == 2:
            x = contract('ab,bc->ac', x, self.cores[1])
        return x

    def InterRes_RightHalfContractionForward(self,x,res_former):
        if res_former is not None and self.InterRes is not False:
            #print('flag:using interRes Connection')
            if self.Inter_resgate is not None:
                #print('flag:using ResGate')
                x = x + self.ResGateForward(res_former).reshape(x.shape)
            else:
                #print('flag:No ResGate')
                x = x + res_former
            x = self.res_layernorm(x)
        if self.LatentAct:
            #print('flag: LatentAct')
            x = self.act(x)
        inter_res = x
        x = self.RightHalfContractionForward(x)
        return x, inter_res
    
    def InterRes_LinearContractionForward(self,x,res_former):
        
        x = self.LeftHalfContractionForward(x)
        x,inter_res = self.InterRes_RightHalfContractionForward(x,res_former)
        x = x.reshape(-1,self.outdim)
        return x, inter_res
    def get_forward(self):
        if self.IntraRes is True and self.InterRes is True:
            f = self.IntraRes_LinearContractionForward
        if self.IntraRes is True and self.InterRes is False:
            f = self.IntraRes_LinearContractionForward
        if self.IntraRes is False and self.InterRes is True:
            f = self.InterRes_LinearContractionForward
        if self.IntraRes is False and self.InterRes is False:
            f = self.InterRes_LinearContractionForward
        return f
    def forward(self,x,res_former=None):
        method_forward = self.get_forward()
        x, ResTensorCores = method_forward(x,res_former)
        if self.bn is not None:
            x = self.bn(x)
        if self.bias is not None:
            x = x + self.bias
        if self.fact is True:
            x = self.act(x)
        return x, ResTensorCores
    

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
            middim: list = [12,8,8,8,8,12],
            rank = None,
            InputShape: list = [12,8,8], 
            activation: str = 'GELU', 
            InterRes_Pos=False,
            IntraRes_Pos=False, 
            InterResGate_Pos=False,
            IntraResGate_Pos=False, 
            TenserizedGate_Pos=False, 
            LatentAct_Pos=None, 
            bn_Pos=None, 
            fact_Pos=None,

    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #---> tflag: Replace qkv to tensorizedlinear layer
        self.to_q = TensorizedLinear(dim, dim, middim, rank, InputShape, activation,
                                     InterRes_Pos['q'], IntraRes_Pos['q'], 
                                     InterResGate_Pos['q'], IntraResGate_Pos['q'],
                                     TenserizedGate_Pos['q'], LatentAct_Pos['q'], 
                                     bn_Pos['q'], fact_Pos['q'], qkv_bias)
        self.to_k = TensorizedLinear(dim, dim, middim, rank, InputShape, activation,
                                     InterRes_Pos['k'], IntraRes_Pos['k'], 
                                     InterResGate_Pos['k'], IntraResGate_Pos['k'],
                                     TenserizedGate_Pos['k'], LatentAct_Pos['k'], 
                                     bn_Pos['k'], fact_Pos['k'], qkv_bias)
        self.to_v = TensorizedLinear(dim, dim, middim, rank, InputShape, activation,
                                     InterRes_Pos['v'], IntraRes_Pos['v'], 
                                     InterResGate_Pos['v'], IntraResGate_Pos['v'],
                                     TenserizedGate_Pos['v'], LatentAct_Pos['v'], 
                                     bn_Pos['v'], fact_Pos['v'], qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        #self.proj = nn.Linear(dim, dim)
        #---> tflag: Replace proj to tensorizedlinear layer
        self.proj = TensorizedLinear(dim, dim, middim, rank, InputShape, activation,
                                     InterRes_Pos['out'], IntraRes_Pos['out'], 
                                     InterResGate_Pos['out'], IntraResGate_Pos['out'],
                                     TenserizedGate_Pos['out'], LatentAct_Pos['out'], 
                                     bn_Pos['out'], fact_Pos['out'], qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, res_former=None) -> torch.Tensor:
        B, N, C = x.shape
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        #q, k, v = qkv.unbind(0)
        ##---> tflag: change qkv to q,k,v
        if res_former is None:
            res_former = [None,None,None,None]
        q, q_res = self.to_q(x, res_former[0])
        k, k_res = self.to_k(x, res_former[1])
        v, v_res = self.to_v(x, res_former[2])
        q = q.view(B, N, self.num_heads, -1).transpose(1, 2)
        k = k.view(B, N, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, N, self.num_heads, -1).transpose(1, 2)
        #-----------
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x, out_res = self.proj(x, res_former[3])
        x = self.proj_drop(x).reshape(B, N, C)
        return x, [q_res,k_res,v_res,out_res]


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    def __init__(
            self,
            #dim: int,
            #num_heads: int,
            #mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            #act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            #mlp_layer: nn.Module = Mlp,
            blk_config = None,
            first = False,
    ) -> None:
        super().__init__()
        #----Attention Config----
        num_heads, dim = blk_config.get('heads'), blk_config.get('indim')
        outdim, middim = blk_config.get('outdim'), blk_config.get('middim')
        rank, InputShape, activation = blk_config.get('rank'), blk_config.get('InputShape'), blk_config.get('activation')
        InterRes_Pos, InterResGate_Pos = blk_config.get('InterRes_Pos'), blk_config.get('InterResGate_Pos')
        IntraRes_Pos, IntraResGate_Pos = blk_config.get('IntraRes_Pos'), blk_config.get('IntraResGate_Pos')
        LatentAct_Pos, _, _ = blk_config.get('LatentAct_Pos'), blk_config.get('proj_drop'), blk_config.get('attn_drop')
        TenserizedGate_Pos = blk_config.get('TenserizedGate_Pos')
        qk_norm, qkv_bias = blk_config.get('qk_norm'), blk_config.get('qkv_bias')
        bn_Pos = blk_config.get('bn_Pos')
        fact_Pos = blk_config.get('fact_Pos')
        #latent_attn = blk_config.get('latent_attn')
        #onlyattn = blk_config.get('onlyattn')
        # ---- MLP Config ----
        mlp1_outdim, mlp1_middim = blk_config.get('mlp1_outdim'), blk_config.get('mlp1_middim')
        mlp1_rank = blk_config.get('mlp1_rank')
        mlp1_bias = blk_config.get('mlp1_bias')
        mlp1_dropout = blk_config.get('mlp1_dropout')

        
        mlp2_middim = blk_config.get('mlp2_middim')
        mlp2_rank, mlp2_InputShape = blk_config.get('mlp2_rank'), blk_config.get('mlp2_InputShape')
        mlp2_bias = blk_config.get('mlp2_bias')
        mlp2_dropout = blk_config.get('mlp2_dropout')
        if first:
            InterRes_Pos = {key: False for key in InterRes_Pos}
            InterResGate_Pos = {key: False for key in InterResGate_Pos}
            TenserizedGate_Pos = {key: False for key in TenserizedGate_Pos}
        # ---------------------------------------------------------
        self.norm1 = norm_layer(dim)
        #----> tflg: change attn to restensor
        """ self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        ) """
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            middim=middim,
            rank=rank,
            InputShape=InputShape,
            activation=activation,
            InterRes_Pos=InterRes_Pos,
            IntraRes_Pos=IntraRes_Pos,
            InterResGate_Pos=InterResGate_Pos,
            IntraResGate_Pos=IntraResGate_Pos,
            TenserizedGate_Pos=TenserizedGate_Pos,
            LatentAct_Pos=LatentAct_Pos,
            bn_Pos=bn_Pos,
            fact_Pos=fact_Pos,
        )
        #-----------------------
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        """ self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        ) """

        self.fc1 = TensorizedLinear(dim, mlp1_outdim, mlp1_middim, mlp1_rank, InputShape, activation,
                                     InterRes_Pos['mlp1'], IntraRes_Pos['mlp1'], 
                                     InterResGate_Pos['mlp1'], IntraResGate_Pos['mlp1'],
                                     TenserizedGate_Pos['mlp1'], LatentAct_Pos['mlp1'], 
                                     bn_Pos['mlp1'], fact_Pos['mlp1'], qkv_bias)
        if activation == 'ReLU':
            self.act = nn.ReLU()
        elif activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'SiLU':
            self.act = nn.SiLU()
        else:
            raise ValueError("No Activation Config for Block")
        
        self.drop1 = nn.Dropout(mlp1_dropout)
        self.mlpnorm = norm_layer(mlp1_outdim) if norm_layer is not None else nn.Identity()

        self.fc2 = TensorizedLinear(mlp1_outdim, dim, mlp2_middim, mlp2_rank, mlp2_InputShape, activation,
                                     InterRes_Pos['mlp2'], IntraRes_Pos['mlp2'], 
                                     InterResGate_Pos['mlp2'], IntraResGate_Pos['mlp2'],
                                     TenserizedGate_Pos['mlp2'], LatentAct_Pos['mlp2'], 
                                     bn_Pos['mlp2'], fact_Pos['mlp2'], qkv_bias)
        self.drop2 = nn.Dropout(mlp2_dropout)

        
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, res_former=None) -> torch.Tensor:
        #x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        #x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        if res_former is None:
            res_former=[None,None,None,None,None,None]
        residual_connection_attn = x

        x = self.norm1(x)
        attn,res = self.attn(x,res_former)
        x = residual_connection_attn + self.drop_path1(self.ls1(attn))
        
        #x = self.norm1(x) # original one 
        residual_connection_mlp = x
        x = self.norm2(x)
        # ffn
        x, mlp1_res = self.fc1(x,res_former[-2])
        x = self.act(x)
        x = self.drop1(x)
        x = self.mlpnorm(x) #
        x, mlp2_res = self.fc2(x,res_former[-1])
        x = self.drop2(x)
        x = residual_connection_mlp + self.drop_path2(self.ls2(x)).view(residual_connection_mlp.shape)
        #x = self.norm2(x) # original one 
        res.append(mlp1_res)
        res.append(mlp2_res)

        return x,res


class ResPostBlock(nn.Module):
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
        self.init_values = init_values

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelScalingBlock(nn.Module):
    """ Parallel ViT block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """
    fused_attn: Final[bool]

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
            mlp_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer('qkv_bias', None)
            self.register_parameter('mlp_bias', None)
        else:
            self.register_buffer('qkv_bias', torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)

        self.ls = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            # Concat constant zero-bias for qkv w/ trainable mlp_bias.
            # Appears faster than adding to x_mlp separately
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.fused_attn:
            x_attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out_proj(x_attn)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        # Add residual w/ drop path & layer scale applied
        y = self.drop_path(self.ls(x_attn + x_mlp))
        x = x + y
        return x


class ParallelThingsBlock(nn.Module):
    """ Parallel ViT block (N parallel attention followed by N parallel MLP)
    Based on:
      `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_parallel: int = 2,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                )),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', mlp_layer(
                    dim,
                    hidden_features=int(dim * mlp_ratio),
                    act_layer=act_layer,
                    drop=proj_drop,
                )),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward_jit(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)


def global_pool_nlc(
        x: torch.Tensor,
        pool_type: str = 'token',
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == 'token':
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == 'avg':
            x = x.mean(dim=1)
        elif pool_type == 'avgmax':
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == 'max':
            x = x.amax(dim=1)
        else:
            assert not pool_type, f'Unknown pool type {pool_type}'

    return x


class LaxVisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            pos_embed: str = 'learn',
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            final_norm: bool = True,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            blk_config = None,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block0 = block_fn(
                #dim=embed_dim,
                #num_heads=num_heads,
                #mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[0],
                norm_layer=norm_layer,
                #act_layer=act_layer,
                #mlp_layer=mlp_layer,
                blk_config = blk_config,
                first=True,
            )
        self.blocks = nn.Sequential(*[
            block_fn(
                #dim=embed_dim,
                #num_heads=num_heads,
                #mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i+1],
                norm_layer=norm_layer,
                #act_layer=act_layer,
                #mlp_layer=mlp_layer,
                blk_config = blk_config,
            )
            for i in range(depth-1)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        if hasattr(self.patch_embed, 'set_grad_checkpointing'):
            self.patch_embed.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int]] = None,
            patch_size: Optional[Tuple[int, int]] = None,
    ):
        """Method updates the input image resolution, patch size

        Args:
            img_size: New input resolution, if None current resolution is used
            patch_size: New patch size, if None existing patch size is used
        """
        prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
        if self.pos_embed is not None:
            num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
            num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
            if num_new_tokens != self.pos_embed.shape[1]:
                self.pos_embed = nn.Parameter(resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=self.patch_embed.grid_size,
                    old_size=prev_grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    verbose=True,
                ))

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        if reshape:
            # reshape to BCHW output format
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if not torch.jit.is_scripting() and return_prefix_tokens:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, List[int], Tuple[int]] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> List[torch.Tensor]:
        """ Intermediate layer accessor inspired by DINO / DINOv2 interface.
        NOTE: This API is for backwards compat, favour using forward_intermediates() directly.
        """
        return self.forward_intermediates(
            x, n,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            output_fmt='NCHW' if reshape else 'NLC',
            intermediates_only=True,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            assert('todo grad_checkpointing')
            x = checkpoint_seq(self.blocks, x)
        else:
            x,res = self.block0(x)
            for block in self.blocks:
                x,res = block(x,res)
        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.0) -> None:
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode: str = 'jax', head_bias: float = 0.0) -> Callable:
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
) -> torch.Tensor:
    """ Rescale the grid of position embeddings when loading from state_dict.
    *DEPRECATED* This function is being deprecated in favour of using resample_abs_pos_embed
    """
    ntok_new = posemb_new.shape[1] - num_prefix_tokens
    ntok_old = posemb.shape[1] - num_prefix_tokens
    gs_old = [int(math.sqrt(ntok_old))] * 2
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    return resample_abs_pos_embed(
        posemb, gs_new, gs_old,
        num_prefix_tokens=num_prefix_tokens,
        interpolation=interpolation,
        antialias=antialias,
        verbose=True,
    )


@torch.no_grad()
def _load_weights(model: LaxVisionTransformer, checkpoint_path: str, prefix: str = '') -> None:
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True, idx=None):
        if idx is not None:
            w = w[idx]
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
        elif 'params/img/embedding/kernel' in w:
            prefix = 'params/img/'
            big_vision = True

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        old_shape = pos_embed_w.shape
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if (isinstance(model.head, nn.Linear) and
            f'{prefix}head/bias' in w and
            model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]):
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    if model.attn_pool is not None:
        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        model.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        model.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        model.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        model.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        model.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        model.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        model.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        model.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(model.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(model.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        if f'{prefix}Transformer/encoderblock/LayerNorm_0/scale' in w:
            block_prefix = f'{prefix}Transformer/encoderblock/'
            idx = i
        else:
            block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
            idx = None
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale'], idx=idx))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias'], idx=idx))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False, idx=idx).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False, idx=idx).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel'], idx=idx).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias'], idx=idx))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale'], idx=idx))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias'], idx=idx))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel'], idx=idx))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias'], idx=idx))


def _convert_openai_clip(
        state_dict: Dict[str, torch.Tensor],
        model: LaxVisionTransformer,
        prefix: str = 'visual.',
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'),
        ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'),
        ('ln_pre', 'norm_pre'),
        ('ln_post', 'norm'),
        ('ln_', 'norm'),
        ('in_proj_', 'qkv.'),
        ('out_proj', 'proj'),
        ('mlp.c_fc', 'mlp.fc1'),
        ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
        out_dict[k] = v
    return out_dict


def _convert_dinov2(
        state_dict: Dict[str, torch.Tensor],
        model: LaxVisionTransformer,
) -> Dict[str, torch.Tensor]:
    import re
    out_dict = {}
    state_dict.pop("mask_token", None)
    if 'register_tokens' in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: LaxVisionTransformer,
        adapt_layer_scale: bool = False,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    if 'visual.class_embedding' in state_dict:
        state_dict = _convert_openai_clip(state_dict, model)
    elif 'module.visual.class_embedding' in state_dict:
        state_dict = _convert_openai_clip(state_dict, model, prefix='module.visual.')
    elif "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)
    elif "encoder" in state_dict:
        # IJEPA, vit in an 'encoder' submodule
        state_dict = state_dict['encoder']
        prefix = 'module.'
    elif 'visual.trunk.pos_embed' in state_dict or 'visual.trunk.blocks.0.norm1.weight' in state_dict:
        # OpenCLIP model with timm vision encoder
        prefix = 'visual.trunk.'
        if 'visual.head.proj.weight' in state_dict and isinstance(model.head, nn.Linear):
            # remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
            out_dict['head.weight'] = state_dict['visual.head.proj.weight']
            out_dict['head.bias'] = torch.zeros(state_dict['visual.head.proj.weight'].shape[0])

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }

default_cfgs = {

    # re-finetuned augreg 21k FT on in1k weights
    'restensor_vit_base_patch16_224': _cfg(
        hf_hub_id='timm/'),
}

_quick_gelu_cfgs = [n for n, c in default_cfgs.items() if c.get('notes', ()) and 'quickgelu' in c['notes'][0]]
for n in _quick_gelu_cfgs:
    # generate quickgelu default cfgs based on contents of notes field
    c = copy.deepcopy(default_cfgs[n])
    if c['hf_hub_id'] == 'timm/':
        c['hf_hub_id'] = 'timm/' + n  # need to use non-quickgelu model name for hub id
    default_cfgs[n.replace('_clip_', '_clip_quickgelu_')] = c
default_cfgs = generate_default_cfgs(default_cfgs)


def _create_restensor_vision_transformer(variant: str, pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    out_indices = kwargs.pop('out_indices', 3)
    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = kwargs.pop('pretrained_strict', True)
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        LaxVisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
def _read_blk_config(cfg):
    with open(cfg,'r') as f:
        layerconfig=yaml.safe_load(f)
    return layerconfig

#-----------SVD---------------
@register_model
def Plain_CoLA_base_16_224_Ablation_SVD(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Plain_CoLA_base_16_224_Ablation_SVD.yaml'
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Plain_CoLA_base_16_224_Ablation_SVD', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def Plain_CoLA_base_16_224_Ablation_LaxSVD(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Plain_CoLA_base_16_224_Ablation_LaxSVD.yaml'
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Plain_CoLA_base_16_224_Ablation_LaxSVD', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def Plain_CoLA_large_16_224_Ablation_SVD(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Plain_CoLA_large_16_224_Ablation_SVD.yaml'
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Plain_CoLA_large_16_224_Ablation_SVD', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def Plain_CoLA_large_16_224_Ablation_LaxSVD(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Plain_CoLA_large_16_224_Ablation_LaxSVD.yaml'
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Plain_CoLA_large_16_224_Ablation_LaxSVD', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


#-----------TT---------------
@register_model
def Plain_TT_4cores_vit_base(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Plain_TT_4cores_vit_base.yaml' #
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Plain_TT_4cores_vit_base', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def Lax_TT_4cores_vit_base(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Lax_TT_4cores_vit_base.yaml'
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Lax_TT_4cores_vit_base', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def Plain_TT_4cores_vit_large(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Plain_TT_4cores_vit_large.yaml'
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Plain_TT_4cores_vit_large', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def Lax_TT_4cores_vit_large(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Lax_TT_4cores_vit_large.yaml' #
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Lax_TT_4cores_vit_large', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

#----CoLA------------

@register_model
def Plain_CoLA_vit_base_16_224(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Plain_CoLA_vit_base_16_224.yaml'
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Plain_CoLA_vit_base_16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def Lax_CoLA_vit_base_16_224(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Lax_CoLA_vit_base_16_224.yaml'
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Lax_CoLA_vit_base_16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def Plain_CoLA_vit_large_16_224(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Plain_CoLA_vit_large_16_224.yaml'
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Plain_CoLA_vit_large_16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def Lax_CoLA_vit_large_16_224(pretrained: bool = False, **kwargs) -> LaxVisionTransformer:
    """
    Important Note: model config is create only by blk_config, but other args should be consistant to blk_config.
    """
    blk_config = f'{config_dir}/Lax_CoLA_vit_large_16_224.yaml'
    layerconfig = _read_blk_config(blk_config)
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, blk_config=layerconfig)
    model = _create_restensor_vision_transformer('Lax_CoLA_vit_large_16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

