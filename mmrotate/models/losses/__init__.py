# Copyright (c) OpenMMLab. All rights reserved.
from .convex_giou_loss import BCConvexGIoULoss, ConvexGIoULoss
from .gaussian_dist_loss import GDLoss
from .gaussian_dist_loss_v1 import GDLoss_v1
from .kf_iou_loss import KFLoss
from .kld_reppoints_loss import KLDRepPointsLoss
from .rotated_iou_loss import RotatedIoULoss
from .smooth_focal_loss import SmoothFocalLoss
from .spatial_border_loss import SpatialBorderLoss
from .label_smooth_loss import LabelSmoothLoss
from .cross_entrypy_loss import CrossEntropyloss
from .arcface_loss import ArcFaceLoss
from .cls_smooth_loss import ClsSmoothLoss
from .utils import (convert_to_one_hot, weight_reduce_loss)
__all__ = [
    'GDLoss', 'GDLoss_v1', 'KFLoss', 'ConvexGIoULoss', 'BCConvexGIoULoss',
    'KLDRepPointsLoss', 'SmoothFocalLoss', 'RotatedIoULoss',
    'SpatialBorderLoss', 'LabelSmoothLoss', 'convert_to_one_hot', 'CrossEntropyloss',
    'weight_reduce_loss', 'ArcFaceLoss', 'ClsSmoothLoss'
]
