# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import cv2
import mmcv
import random
import torch
import numpy as np
from mmdet.datasets.pipelines.transforms import (Mosaic, RandomCrop,
                                                 RandomFlip, Resize)

from mmrotate.core import (norm_angle, obb2poly_np, poly2obb_np, 
                           rbbox_overlaps, bbox_overlaps, obb2poly, 
                           imshow_det_rbboxes, poly2obb)
from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class RResize(Resize):
    """Resize images & rotated bbox Inherit Resize pipeline class to handle
    rotated bboxes.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio).
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None):
        super(RResize, self).__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=True)

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            orig_shape = bboxes.shape
            bboxes = bboxes.reshape((-1, 5))
            w_scale, h_scale, _, _ = results['scale_factor']
            bboxes[:, 0] *= w_scale
            bboxes[:, 1] *= h_scale
            bboxes[:, 2:4] *= np.sqrt(w_scale * h_scale)
            results[key] = bboxes.reshape(orig_shape)


@ROTATED_PIPELINES.register_module()
class RRandomFlip(RandomFlip):
    """

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'.
        version (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self, flip_ratio=None, direction='horizontal', version='oc'):
        self.version = version
        super(RRandomFlip, self).__init__(flip_ratio, direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally or vertically.

        Args:
            bboxes(ndarray): shape (..., 5*k)
            img_shape(tuple): (height, width)

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        assert bboxes.shape[-1] % 5 == 0
        orig_shape = bboxes.shape
        bboxes = bboxes.reshape((-1, 5))
        flipped = bboxes.copy()
        if direction == 'horizontal':
            flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
        elif direction == 'vertical':
            flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
        elif direction == 'diagonal':
            flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
            flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
            return flipped.reshape(orig_shape)
        else:
            raise ValueError(f'Invalid flipping direction "{direction}"')
        if self.version == 'oc':
            rotated_flag = (bboxes[:, 4] != np.pi / 2)
            flipped[rotated_flag, 4] = np.pi / 2 - bboxes[rotated_flag, 4]
            flipped[rotated_flag, 2] = bboxes[rotated_flag, 3]
            flipped[rotated_flag, 3] = bboxes[rotated_flag, 2]
        else:
            flipped[:, 4] = norm_angle(np.pi - bboxes[:, 4], self.version)
        return flipped.reshape(orig_shape)


@ROTATED_PIPELINES.register_module()
class PolyRandomRotate(object):
    """Rotate img & bbox.
    Reference: https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA

    Args:
        rotate_ratio (float, optional): The rotating probability.
            Default: 0.5.
        mode (str, optional) : Indicates whether the angle is chosen in a
            random range (mode='range') or in a preset list of angles
            (mode='value'). Defaults to 'range'.
        angles_range(int|list[int], optional): The range of angles.
            If mode='range', angle_ranges is an int and the angle is chosen
            in (-angles_range, +angles_ranges).
            If mode='value', angles_range is a non-empty list of int and the
            angle is chosen in angles_range.
            Defaults to 180 as default mode is 'range'.
        auto_bound(bool, optional): whether to find the new width and height
            bounds.
        rect_classes (None|list, optional): Specifies classes that needs to
            be rotated by a multiple of 90 degrees.
        version  (str, optional): Angle representations. Defaults to 'le90'.
    """

    def __init__(self,
                 rotate_ratio=0.5,
                 mode='range',
                 angles_range=180,
                 auto_bound=False,
                 rect_classes=None,
                 version='le90'):
        self.rotate_ratio = rotate_ratio
        self.auto_bound = auto_bound
        assert mode in ['range', 'value'], \
            f"mode is supposed to be 'range' or 'value', but got {mode}."
        if mode == 'range':
            assert isinstance(angles_range, int), \
                "mode 'range' expects angle_range to be an int."
        else:
            assert mmcv.is_seq_of(angles_range, int) and len(angles_range), \
                "mode 'value' expects angle_range as a non-empty list of int."
        self.mode = mode
        self.angles_range = angles_range
        self.discrete_range = [90, 180, -90, -180]
        self.rect_classes = rect_classes
        self.version = version

    @property
    def is_rotate(self):
        """Randomly decide whether to rotate."""
        return np.random.rand() < self.rotate_ratio

    def apply_image(self, img, bound_h, bound_w, interp=cv2.INTER_LINEAR):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0:
            return img
        return cv2.warpAffine(
            img, self.rm_image, (bound_w, bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y)
        points
        """
        if len(coords) == 0:
            return coords
        coords = np.asarray(coords, dtype=float)
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def create_rotation_matrix(self,
                               center,
                               angle,
                               bound_h,
                               bound_w,
                               offset=0):
        """Create rotation matrix."""
        center += offset
        rm = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        if self.auto_bound:
            rot_im_center = cv2.transform(center[None, None, :] + offset,
                                          rm)[0, 0, :]
            new_center = np.array([bound_w / 2, bound_h / 2
                                   ]) + offset - rot_im_center
            rm[:, 2] += new_center
        return rm

    def filter_border(self, bboxes, h, w):
        """Filter the box whose center point is outside or whose side length is
        less than 5."""
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        w_bbox, h_bbox = bboxes[:, 2], bboxes[:, 3]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) & \
                    (w_bbox > 5) & (h_bbox > 5)
        return keep_inds

    def __call__(self, results):
        """Call function of PolyRandomRotate."""
        if not self.is_rotate:
            results['rotate'] = False
            angle = 0
        else:
            results['rotate'] = True
            if self.mode == 'range':
                angle = self.angles_range * (2 * np.random.rand() - 1)
            else:
                i = np.random.randint(len(self.angles_range))
                angle = self.angles_range[i]

            class_labels = results['gt_labels']
            for classid in class_labels:
                if self.rect_classes:
                    if classid in self.rect_classes:
                        np.random.shuffle(self.discrete_range)
                        angle = self.discrete_range[0]
                        break

        h, w, c = results['img_shape']
        img = results['img']
        results['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = \
            abs(np.cos(angle / 180 * np.pi)), abs(np.sin(angle / 180 * np.pi))
        if self.auto_bound:
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos,
                 h * abs_cos + w * abs_sin]).astype(int)
        else:
            bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(image_center, angle,
                                                     bound_h, bound_w)
        self.rm_image = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w, offset=-0.5)

        img = self.apply_image(img, bound_h, bound_w)
        results['img'] = img
        results['img_shape'] = (bound_h, bound_w, c)
        gt_bboxes = results.get('gt_bboxes', [])
        labels = results.get('gt_labels', [])
        gt_bboxes = np.concatenate(
            [gt_bboxes, np.zeros((gt_bboxes.shape[0], 1))], axis=-1)
        polys = obb2poly_np(gt_bboxes, self.version)[:, :-1].reshape(-1, 2)
        polys = self.apply_coords(polys).reshape(-1, 8)
        gt_bboxes = []
        for pt in polys:
            pt = np.array(pt, dtype=np.float32)
            obb = poly2obb_np(pt, self.version) \
                if poly2obb_np(pt, self.version) is not None\
                else [0, 0, 0, 0, 0]
            gt_bboxes.append(obb)
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        keep_inds = self.filter_border(gt_bboxes, bound_h, bound_w)
        gt_bboxes = gt_bboxes[keep_inds, :]
        labels = labels[keep_inds]
        if len(gt_bboxes) == 0:
            return None
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = labels

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_ratio={self.rotate_ratio}, ' \
                    f'base_angles={self.base_angles}, ' \
                    f'angles_range={self.angles_range}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class RRandomCrop(RandomCrop):
    """Random crop the image & bboxes.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels must be aligned. That is, `gt_bboxes`
          corresponds to `gt_labels`, and `gt_bboxes_ignore` corresponds to
          `gt_labels_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 version='oc'):
        self.version = version
        super(RRandomCrop, self).__init__(crop_size, crop_type,
                                          allow_negative_crop)

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('bbox_fields', []):
            assert results[key].shape[-1] % 5 == 0

        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        height, width, _ = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, 0, 0, 0],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset

            valid_inds = (bboxes[:, 0] >=
                          0) & (bboxes[:, 0] < width) & (bboxes[:, 1] >= 0) & (
                              bboxes[:, 1] < height)
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

        return results


@ROTATED_PIPELINES.register_module()
class RMosaic(Mosaic):
    """Rotate Mosaic augmentation. Inherit from
    `mmdet.datasets.pipelines.transforms.Mosaic`.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text
                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:
         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Defaults to 0.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` is invalid. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        version  (str, optional): Angle representations. Defaults to `oc`.
    """

    def __init__(self,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=10,
                 bbox_clip_border=True,
                 skip_filter=True,
                 pad_val=114,
                 prob=1.0,
                 version='oc'):
        super(RMosaic, self).__init__(
            img_scale=img_scale,
            center_ratio_range=center_ratio_range,
            min_bbox_size=min_bbox_size,
            bbox_clip_border=bbox_clip_border,
            skip_filter=skip_filter,
            pad_val=pad_val,
            prob=1.0)

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            np.random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            np.random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0] = \
                    scale_ratio_i * gt_bboxes_i[:, 0] + padw
                gt_bboxes_i[:, 1] = \
                    scale_ratio_i * gt_bboxes_i[:, 1] + padh
                gt_bboxes_i[:, 2:4] = \
                    scale_ratio_i * gt_bboxes_i[:, 2:4]

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            mosaic_bboxes, mosaic_labels = \
                self._filter_box_candidates(
                    mosaic_bboxes, mosaic_labels,
                    2 * self.img_scale[1], 2 * self.img_scale[0]
                )
        # If results after rmosaic does not contain any valid gt-bbox,
        # return None. And transform flows in MultiImageMixDataset will
        # repeat until existing valid gt-bbox.
        if len(mosaic_bboxes) == 0:
            return None

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results

    def _filter_box_candidates(self, bboxes, labels, w, h):
        """Filter out small bboxes and outside bboxes after Mosaic."""
        bbox_x, bbox_y, bbox_w, bbox_h = \
            bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        valid_inds = (bbox_x > 0) & (bbox_x < w) & \
                     (bbox_y > 0) & (bbox_y < h) & \
                     (bbox_w > self.min_bbox_size) & \
                     (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds]

def find_inside_objs(bboxes, target_h, target_w, iof_thre=0.7):
    # image_box = np.array([[target_w / 2, target_h / 2, target_w, target_h, 0]], dtype=bboxes.dtype) # TODO: TEST
    image_box = np.array([[0, 0, target_w, target_h]], dtype=bboxes.dtype) # TODO: TEST
    iof = bbox_overlaps(bboxes, image_box, mode="iof")   #NOTE: the image_box is (x,y,x,y)
    keep_masks = iof > iof_thre # (n, 1)
    keep_masks = np.squeeze(keep_masks, axis=-1) # (n, )
    keep_inds = np.nonzero(keep_masks)[0]
    return keep_inds

@ROTATED_PIPELINES.register_module()
class RRandomPutImage:
    def __init__(self, 
                 img_scale=(1024, 1024),
                 ratio_range=(0.5, 1.5),
                 flip_ratio=0.5,
                 pad_val=114,
                 max_iters=15,
                 iof_thre=0.5,
                 with_put_polys=False):
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.iof_thre = iof_thre
        self.with_put_polys = with_put_polys

    def __call__(self, results):
        return self.random_put(results)

    def random_put(self, results):
        assert "img" in results
        assert 'mix_results' in results
        assert len(results['mix_results']) == 1, 'Only support put 2 images'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            return results, None

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']
        results['class'] = retrieve_results['class']

        areas =  retrieve_results['gt_bboxes'][:,2]*retrieve_results['gt_bboxes'][:,3]
        areas_mean = np.mean(areas)
        if areas_mean < 1000:
            jit_factor = random.uniform(1,1.5)
        else:
            jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_flip:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w,
                                          target_w), 3)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(1, padded_img.shape[1] - target_w)
        put_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_polys = obb2poly(torch.tensor(retrieve_results['gt_bboxes']), 'le90')
        retrieve_gt_polys = np.array(retrieve_gt_polys)

        # retrieve_gt_polys[:, 0::2] = retrieve_gt_polys[:, 0::2] * scale_ratio
        # retrieve_gt_polys[:, 1::2] = retrieve_gt_polys[:, 1::2] * scale_ratio
        retrieve_gt_polys = retrieve_gt_polys * scale_ratio

        if is_flip:
            retrieve_gt_polys[:, 0::2] = origin_w - retrieve_gt_polys[:, 0::2]

        # 7. filter
        cp_retrieve_gt_polys = retrieve_gt_polys.copy()
        cp_retrieve_gt_polys[:, 0::2] = \
            cp_retrieve_gt_polys[:, 0::2] - x_offset
        cp_retrieve_gt_polys[:, 1::2] = \
            cp_retrieve_gt_polys[:, 1::2] - y_offset

        inside_keep_inds = find_inside_objs(cp_retrieve_gt_polys, target_h, target_w, iof_thre=self.iof_thre)
        cp_retrieve_gt_polys = cp_retrieve_gt_polys[inside_keep_inds]
        
        gt_bboxes = poly2obb(torch.tensor(cp_retrieve_gt_polys),'le90')
        retrieve_results["gt_bboxes"] = np.array(gt_bboxes,dtype=np.float32)
        retrieve_results["gt_labels"] = retrieve_results["gt_labels"][inside_keep_inds]
        for k in retrieve_results.get("aligned_fields", []):
            retrieve_results[k] = retrieve_results[k][inside_keep_inds]
        
        if self.with_put_polys:
            retrieve_results["gt_polys"] = cp_retrieve_gt_polys
        else:
            retrieve_results.pop("gt_polys", None)
            
        retrieve_results["img"] = put_img.astype(np.uint8)
        retrieve_results["img_shape"] = put_img.shape
        results["img"] = ori_img.astype(np.uint8)
        results["img_shape"] = ori_img.shape
        return results, retrieve_results

    def get_random_results(self, dataset):
        for _ in range(self.max_iters):
            index = random.randint(0, len(dataset) - 1) # TODO: test -1
            gt_bboxes_i = dataset.get_ann_info(index)['bboxes']
            if len(gt_bboxes_i) != 0:
                break
        results = copy.deepcopy(dataset(index))
        return results

def poly2rbb(poly):
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a


@ROTATED_PIPELINES.register_module()
class RCopyPaste(RRandomPutImage):
    def __init__(self,
                 img_scale=(1024,1024),
                 ratio_range=(0.5,1.5),
                 flip_ratio=0.5,
                 pad_val=114,
                 copy_choice_prob=1.,
                 copy_iof_thre=0.5,
                 is_resample=False,
                 max_iters=15,
                 iof_thre=0.5,
                 ignore_classes=[],
                 sample_frequency={},
                 with_polys=False,
                 version = 'le90'):
        self.version = version
        super().__init__(
            img_scale=img_scale,
            ratio_range=ratio_range,
            flip_ratio=flip_ratio,
            pad_val=pad_val,
            max_iters=max_iters,
            with_put_polys=True,
            iof_thre=iof_thre
        )

        assert isinstance(img_scale,tuple)
        self.is_resample = is_resample
        self.copy_choice_prob = copy_choice_prob
        self.copy_iof_thre = copy_iof_thre
        self.with_polys = with_polys
        self.ignore_classes = ignore_classes
        self.sample_frequency = sample_frequency


    def __call__(self, results): 
        results = self._copy_transform(results)
        return results

    def _copy_transform(self,results):
        ori_results, put_results = self.random_put(results)
 
        if put_results is None or len(put_results['gt_bboxes']) == 0:
            return ori_results

        put_img = put_results["img"]
        put_gt_labels = put_results["gt_labels"]
        put_gt_bboxes = put_results["gt_bboxes"]
        put_gt_polys = put_results["gt_polys"]


        ori_img = ori_results["img"]
        ori_gt_labels = ori_results["gt_labels"]
        if "gt_polys" in ori_results:
            ori_gt_polys = ori_results["gt_polys"]
        else:
            ori_gt_polys = ori_results['ann_info']["polygons"]
        ori_gt_bboxes = ori_results["gt_bboxes"]
        
        blank_img = np.zeros(put_img.shape, dtype=np.uint8)
        keep_inds = []
        for i, put_gt_poly in enumerate(put_gt_polys):
            put_gt_rbox = np.array(poly2rbb(put_gt_poly), dtype=ori_gt_bboxes.dtype)
            ious = rbbox_overlaps(torch.tensor(ori_gt_bboxes), torch.tensor(put_gt_rbox[None]), mode="iof") # (1, num_ori_objs)
            if (ious < self.copy_iof_thre).all():
                keep_inds.append(i)
                cv2.drawContours(
                    blank_img, [put_gt_poly.reshape(-1, 1, 2).astype(np.int32)], 
                    -1, (255, 255, 255), cv2.FILLED
                )

        if len(keep_inds):
            keep_inds = np.asarray(keep_inds, dtype=np.int64) # (pos, )
            blank_mask = blank_img > 0
            ori_img[blank_mask] = put_img[blank_mask]
            ori_results["gt_labels"] = np.concatenate((ori_gt_labels, put_gt_labels[keep_inds]), axis=0)
            ori_results["gt_bboxes"] = np.concatenate((ori_gt_bboxes, put_gt_bboxes[keep_inds]), axis=0)
            for k in ori_results.get("aligned_fields", []):
                ori_results[k] = np.concatenate((ori_results[k], put_results[k][keep_inds]), axis=0)
            ori_gt_polys = np.vstack((ori_gt_polys, put_gt_polys[keep_inds]))
        if self.with_polys:
            ori_results["gt_polys"] = ori_gt_polys
        else:
            ori_results.pop("gt_polys", None)

        ori_results["img"] = ori_img.astype(np.uint8)
        ori_results["img_shape"] = ori_img.shape
        #self.vis(ori_results)
        return ori_results


    def get_random_results(self, dataset):
        self.ignore_classes = list(self.ignore_classes) \
            if isinstance(self.ignore_classes, tuple) else self.ignore_classes
        per_cls_indexes = dataset.per_cls_indexes
        if self.is_resample:
            max_num_per_cls = max([len(indexes) for indexes in per_cls_indexes])
            per_cls_frequency = [(1 / len(indexes) * max_num_per_cls) if len(indexes) and i not in self.ignore_classes \
                else 0 for i, indexes in enumerate(per_cls_indexes)]
            if not hasattr(self, "per_cls_frequency"):
                for i, id in enumerate(self.ignore_classes):
                    if isinstance(id, str):
                        if id in dataset.CLASSES:
                            self.ignore_classes[i] = (dataset.CLASSES).index(id)
                        else:
                            raise KeyError
                    elif isinstance(id, int):
                        self.ignore_classes[i] = id
                    else:
                        raise NotImplementedError

            self.per_cls_frequency = per_cls_frequency

            k = random.choices(range(len(self.per_cls_frequency)), self.per_cls_frequency, k=1)[0]
            class_ids = per_cls_indexes[k]
            index = class_ids[random.randint(0, len(class_ids) - 1)]
            results = copy.deepcopy(dataset.__getitem__(index,get_anthor=True))
            keep_inds = np.nonzero((results["gt_labels"] == k))[0]
            results["gt_bboxes"] = results["gt_bboxes"][keep_inds]
            results["gt_labels"] = results["gt_labels"][keep_inds]
            results['class'] = dataset.CLASSES
            for k in results.get("aligned_fields", []):
                results[k] = results[k][keep_inds]
        return results

    def vis(self,
            results,
            score_thr=0.3,
            bbox_color=(72,101,241),
            text_color=(72,101,241),
            mask_color=None,
            thickness=2,
            font_size=13,
            win_name='',
            show=False,
            wait_time=0,
            out_file=None):
        img = results['img']
        bboxes = results['gt_bboxes']
        class_names = results['class']
        img = mmcv.imread(img)
        img = img.copy()
        scores = np.ones((bboxes.shape[0],1))
        bboxes = np.concatenate((bboxes,scores),1)        
        labels = results['gt_labels']
        segms = None
        out_file = osp.join('work_dirs/debug/vis',results['ori_filename'])
        if out_file is not None:
            show = False
        #draw bounding boxes
        img = imshow_det_rbboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file
        )



    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'dynamic_scale={self.dynamic_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'is_resample={self.is_resample})'
        repr_str += f'copy_choice_prob={self.copy_choice_prob})'
        repr_str += f'copy_iou_thre={self.copy_iof_thre})'
        repr_str += f"max_iters={self.max_iters}"
        repr_str += f"ignore_classes={self.ignore_classes}"
        return repr_str