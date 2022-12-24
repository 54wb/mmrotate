import copy

import numpy as np
import torch
from ..bbox import bbox_mapping_back

def merge_aug_rbboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented rotated detection bboxes and scores

    Args:
        aug_bboxes: shape (n, 5*#class)
        aug_scores: shape (n, #class)
        img_metas: shape (3,)
        rcnn_test_cfg: rcnn test configs
    
    Returns:
        tuple: (bboxes, scores)
    """

    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip, flip_direction)
        recovered_bboxes.append(bboxes)
    
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores