_base_ = [
    './roi_trans_r50_fpn_1x_dota_le90.py'
]

checkpoint_file = '/disk0/lwb/pretrain/convnext_tiny_22k_1k_384.pth'

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

evaluation = dict(interval=6, metric='mAP')
optimizer = dict(
    _delete_ = True,
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate':0.95,
        'decay_type':'layer_wise',
        'num_layers':6}
)
data = dict(
    samples_per_gpu=4
)
lr_config = dict(warmup_iters=1000,step=[27,33])
runner = dict(max_epochs=36)
checkpoint_config = dict(interval=4)
fp16 = dict(loss_scale='dynamic')