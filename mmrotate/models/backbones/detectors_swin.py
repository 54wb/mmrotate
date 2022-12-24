from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint

from ..builder import ROTATED_BACKBONES
from mmdet.models.utils.transformer import PatchMerging
from mmdet.models.backbones.swin import  SwinTransformer
from mmdet.models.backbones.swin import SwinBlock as _SwinBlock

class SwinBlock(_SwinBlock):
    def __init__(self,
                 rfp_inplanes=None,
                 init_cfg = None,
                 **kwargs):
        super(SwinBlock,self).__init__(**kwargs)

        self.rfp_inplanes = rfp_inplanes
        if self.rfp_inplanes:
            self.rfp_conv = build_conv_layer(
                None,
                self.rfp_inplanes,
                kwargs['embed_dims'],
                1,
                stride=1,
                bias=True)
            if init_cfg is None:
                self.init_cfg = dict(
                    type='Constant', val=0, override=dict(name='rfp_conv'))

    def rfp_forward(self, x, rfp_feat, hw_shape):
        "The forward function that also takes the RFP features as input"

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        
        if self.rfp_inplanes:
            rfp_feat = self.rfp_conv(rfp_feat)
            rfp_feat = rfp_feat.flatten(2).transpose(1,2)
            x = x + rfp_feat
        
        return x


class SwinBlockSequence(BaseModule):
    "one stage in swin transformer for RFP in detectors"
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 rfp_inplanes=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates)  == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        
        self.blocks = ModuleList()
        for i in range(depth):
            if i == 0:
                block = SwinBlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    window_size=window_size,
                    shift=False if i % 2 == 0 else True,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rates[i],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    rfp_inplanes=rfp_inplanes)
            else:
                block = SwinBlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    window_size=window_size,
                    shift=False if i % 2 == 0 else True,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rates[i],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    init_cfg=None)
            self.blocks.append(block)
        
        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)
        
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape
    
    def rfp_forward(self, x, rfp_feat, hw_shape):
        for block in self.blocks:
            x = block.rfp_forward(x, rfp_feat, hw_shape)
        
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape
    
    






@ROTATED_BACKBONES.register_module()
class DetectoRS_SwinTransformer(SwinTransformer):
    "SwinTransformer backbone for DetectoRS"

    def __init__(self,
                 rfp_inplanes=None,
                 strides=(4,2,2,2),
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 output_img=False,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if init_cfg is not None:
            assert isinstance(init_cfg, dict), \
                f'init_cfg must be a dict, but got {type(init_cfg)}'
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        self.rfp_inplanes = rfp_inplanes
        self.output_img = output_img
        super(DetectoRS_SwinTransformer,self).__init__(**kwargs,init_cfg=self.init_cfg)

        # set stochastic depth decay rule
        depths = kwargs['depths']
        num_layers = len(depths)
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], total_depth)
        ]

        self.stages = ModuleList()
        in_channels = kwargs['embed_dims']
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2*in_channels,
                    stride=strides[i+1],
                    norm_cfg=norm_cfg,
                    init_cfg=None)
            else:
                downsample = None
            
            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=kwargs['num_heads'][i],
                feedforward_channels=kwargs['mlp_ratio']*in_channels,
                depth=depths[i],
                window_size=kwargs['window_size'],
                qkv_bias=kwargs['qkv_bias'],
                qk_scale=kwargs['qk_scale'],
                drop_rate=kwargs['drop_rate'],
                attn_drop_rate=kwargs['attn_drop_rate'],
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i+1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=kwargs['with_cp'],
                rfp_inplanes=rfp_inplanes if i > 0 else None,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels


    def forward(self, x):
        "forward function for DetectoRS"
        outs = list(super(DetectoRS_SwinTransformer,self).forward(x))
        if self.output_img:
            outs.insert(0,x)
        return tuple(outs)

    def rfp_forward(self, x, rfp_feats):
        "forward function for RFP"
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            h, w = self.absolute_pos_embed.shape[1:3]
            if hw_shape[0] != h or hw_shape[1] != w:
                absolute_pos_embed = F.interpolate(
                    self.absolute_pos_embed,
                    size=hw_shape,
                    mode='bicubic',
                    align_corners=False).flatten(2).transpose(1, 2)
            else:
                absolute_pos_embed = self.absolute_pos_embed.flatten(
                    2).transpose(1, 2)
            x = x + absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            rfp_feat = rfp_feats[i] if i > 0 else None
            x, hw_shape, out, out_hw_shape = stage.rfp_forward(x, rfp_feat, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs


            






