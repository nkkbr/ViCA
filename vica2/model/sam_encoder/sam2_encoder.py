# Adapted from Meta's code base: https://github.com/facebookresearch/sam2

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# print(torch.cuda.memory_summary())

import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP
from transformers import PretrainedConfig, PreTrainedModel
import json
from vica2.model.sam_encoder.sam2_image_processor import SAM2ImageProcessor
from llava.utils import rank0_print


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


def enhanced_scaled_dot_product_attention(query, key, value):
    """
    Computes scaled dot-product attention with a safeguard for large batch sizes.

    In practice, if the batch size or the resulting tensor size exceeds 2**16, 
    it can cause CUDA launch or memory errors due to hardware limitations. 
    To address this, we check whether the intermediate tensor size exceeds this threshold.
    If it does, we split the attention computation into smaller chunks, 
    perform the attention calculation on each chunk separately, 
    and finally merge the results to obtain the final attention output.
    """

    batch_size = query.shape[0]
    if batch_size<=2**15:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            )
    else:
        results = []
        chunk_size = 2**15
        for i in range(0,batch_size,chunk_size):
            q_chunk = query[i:i+chunk_size]
            k_chunk = key[i:i+chunk_size]
            v_chunk = value[i:i+chunk_size]
            out_chunk = F.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk)
            results.append(out_chunk)
        x_chunked = torch.cat(results, dim=0)
        return x_chunked


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        # x = F.scaled_dot_product_attention(
        #     q.transpose(1, 2),
        #     k.transpose(1, 2),
        #     v.transpose(1, 2),
        # )

        x = enhanced_scaled_dot_product_attention(
            query=q.transpose(1, 2),
            key=k.transpose(1, 2),
            value=v.transpose(1, 2),
        )

        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            # logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))
            res = self.load_state_dict(chkpt, strict=False)
            logging.info(f"loading Hiera: {res}")

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)


class HieraConfig(PretrainedConfig):
    model_type = "hiera"

    def __init__(
        self,
        embed_dim=96,
        num_heads=1,
        drop_path_rate=0.0,
        q_pool=3,
        q_stride=(2, 2),
        stages=(2, 3, 16, 3),
        dim_mul=2.0,
        head_mul=2.0,
        window_pos_embed_bkg_spatial_size=(14, 14),
        window_spec=(8, 4, 14, 7),
        global_att_blocks=(12, 16, 20),
        weights_path=None,
        return_interm_layers=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
        self.q_pool = q_pool
        self.q_stride = q_stride
        self.stages = stages
        self.dim_mul = dim_mul
        self.head_mul = head_mul
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.window_spec = window_spec
        self.global_att_blocks = global_att_blocks
        self.weights_path = weights_path
        self.return_interm_layers = return_interm_layers

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class HieraVisionModel(PreTrainedModel):
    config_class = HieraConfig
    _no_split_modules = ["Hiera"]
    
    def __init__(self, config, weights_path=None):
        super().__init__(config)
        self.hiera = Hiera(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.drop_path_rate,
            q_pool=config.q_pool,
            q_stride=config.q_stride,
            stages=config.stages,
            dim_mul=config.dim_mul,
            head_mul=config.head_mul,
            window_pos_embed_bkg_spatial_size=config.window_pos_embed_bkg_spatial_size,
            window_spec=config.window_spec,
            global_att_blocks=config.global_att_blocks,
            return_interm_layers=config.return_interm_layers,
            weights_path=weights_path,
        )

    def forward(self, x):
        return self.hiera(x)

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    #     config = HieraConfig.from_json_file(f"{pretrained_model_name_or_path}/config.json")
    #     weights_path = f"{pretrained_model_name_or_path}/pytorch_model.pt"
    #     model = cls(config, weights_path=weights_path)
        
    #     for module in model.modules():
    #         module._is_hf_initialized = True
    #     return model
    

class SAM2VisionTower(nn.Module):
    def __init__(self, vision_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        assert self.vision_tower_name == "nkkbr/hiera-base-plus-in-sam2.1", "We currently only support nkkbr/hiera-base-plus-in-sam2.1"
        self.image_processor = SAM2ImageProcessor()

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return
        
        self.vision_tower = HieraVisionModel.from_pretrained("nkkbr/hiera-base-plus-in-sam2.1")
        self.vision_tower = self.vision_tower.hiera
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def process(self, images):
        image_forward_out = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
        image_features = image_forward_out[3] # 选取了最后一个特征，将来可以做修改
        B, C, H, W = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
        assert image_features.shape[-2] == 1024
        return image_features

    # def process(self, images):
    #     """
    #     Processes input images in batches to avoid out-of-memory (OOM) issues.

    #     Note:
    #         The Hiera model has relatively high memory consumption, especially
    #         when dealing with videos where the number of input frames (images)
    #         can be very large. To address this, this function modifies the
    #         original implementation by processing the images in smaller batches,
    #         reducing peak memory usage and helping to prevent OOM errors.
    #     """
    #     ret = []
    #     chunk_size = 1 # 设置为64则相当于没有这个设置（视频默认是选取64帧进行训练）如果OOM，需要减小这个数值
    #     for i in range(0,images.shape[0], chunk_size):
    #         images_curr = images[i:i+chunk_size]
    #         image_forward_out = self.vision_tower(images_curr.to(device=self.device, dtype=self.dtype))
    #         image_features = image_forward_out[3] # 选取了最后一个特征，将来可以做修改
    #         B, C, H, W = image_features.shape
    #         image_features = image_features.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
    #         assert image_features.shape[-2] == 1024
    #         ret.append(image_features)
    #     return torch.cat(ret,dim=0)

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image = image.unsqueeze(0)
                image_feature = self.process(image)
                image_feature = image.squeeze(0)
                image_features.append(image_feature)
        else:
            image_features = self.process(images)
        return image_features

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device