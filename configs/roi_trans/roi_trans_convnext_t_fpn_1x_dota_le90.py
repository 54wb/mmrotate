_base_ = [
    './roi_trans_r50_fpn_1x_dota_le90.py',
]

checkpoint_file = '/disk0/lwb/pretrain/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='ConvNeXt',
        arch = 'tiny',
        out_indices=[0,1,2,3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',checkpoint=checkpoint_file)),
    neck=dict(in_channels=[96, 192, 384, 768])
)


optimizer = dict(
    _delete_ = True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
    })
)
lr_config = dict(warmup_iters=1000,step=[8,11])
runner = dict(max_epochs=12)