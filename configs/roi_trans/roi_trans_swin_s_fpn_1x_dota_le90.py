_base_ = ['./roi_trans_swin_tiny_fpn_1x_dota_le90.py']

pretrained = '/home/lwb/work/code/mmrotate/work_dirs/pretrain/swin_small_patch4_window7_224.pth'  # noqa

model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2], 
        init_cfg=dict(type='Pretrained', checkpoint=pretrained))) 

