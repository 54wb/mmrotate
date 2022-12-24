import numpy as np
from .obb_nms import (single_obb_nms, single_poly_nms)

def obb_nms(bboxes, scores, iou_thr, score_thr, max_num=-1, small_thr=1e-6, mode="obb"):
    assert bboxes.ndim == 2 and scores.ndim == 1
    bboxes = bboxes.astype(np.float64)
    scores = scores.astype(np.float64)

    if len(bboxes) == 0:
        return np.empty((0, ), dtype=np.int64)
    else:
        if mode == "obb":
            areas = bboxes[:, 2] * bboxes[:, 3]
        elif mode == "poly":
            areas = (np.max(bboxes[:, 0::2]) - np.min(bboxes[:, 0::2])) \
                * (np.max(bboxes[:, 1::2]) - np.min(bboxes[:, 1::2])) / 2
        else:
            raise NotImplementedError
        filter_mask = np.bitwise_and((scores > score_thr), (areas > small_thr))
        filter_inds = np.nonzero(filter_mask)[0]
        _scores = scores[filter_inds]
        _bboxes = bboxes[filter_inds]

        num_to_keep = 0
        orders = np.argsort(_scores)[::-1]
        num_dets = len(_bboxes)
        suppress = np.zeros(num_dets, dtype=np.uint8)
        keeps = np.zeros(num_dets, dtype=np.int64)

        for i in range(num_dets):
            index_i = orders[i]
            if suppress[index_i] == 1:
                continue
            keeps[num_to_keep] = index_i
            num_to_keep += 1
            for j in range(i + 1, num_dets):
                index_j = orders[j]
                if suppress[index_j] == 1:
                    continue
                if mode == "obb":
                    ovr = single_obb_nms(_bboxes[index_i], _bboxes[index_j])
                elif mode == "poly":
                    ovr = single_poly_nms(_bboxes[index_i], _bboxes[index_j])
                if ovr >= iou_thr:
                    suppress[index_j] = 1
        
        keeps = filter_inds[keeps[:num_to_keep]]
        if max_num > 0:
            keeps = keeps[:max_num]

        return keeps
    