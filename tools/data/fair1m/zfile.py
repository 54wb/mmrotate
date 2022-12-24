import os
import os.path as osp
train_dir = '/disk0/lwb/datasets/Fair1m1_0/dota/train'
test_img_dir = '/disk0/lwb/datasets/Fair1m1_0/dota/test/images'

for root, dirs, files in os.walk(test_img_dir):

    for f in files:
        src = osp.join(root, f)
        tar = "P"+f[:-4][1:].zfill(4)+".png"
        tar = osp.join(root, tar)
        os.rename(src,tar)

