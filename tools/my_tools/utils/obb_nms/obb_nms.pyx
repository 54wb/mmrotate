import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

cdef extern from "m_obb_nms.cc":
    ctypedef float TYPE
    T single_box_iou_rotated[T](const T * const bo1_raw, const T * const box2_raw, const int & nms_type)


def single_obb_nms(np.ndarray[DTYPE_t, ndim=1] box1, np.ndarray[DTYPE_t, ndim=1] box2):
    if not box1.flags['C_CONTIGUOUS']:
        box1 = np.ascontiguousarray(box1)
    if not box2.flags['C_CONTIGUOUS']:
        box2 = np.ascontiguousarray(box2)
    return single_box_iou_rotated[double](&box1[0], &box2[0], 0)

def single_poly_nms(np.ndarray[DTYPE_t, ndim=1] box1, np.ndarray[DTYPE_t, ndim=1] box2):
    if not box1.flags['C_CONTIGUOUS']:
        box1 = np.ascontiguousarray(box1)
    if not box2.flags['C_CONTIGUOUS']:
        box2 = np.ascontiguousarray(box2)
    return single_box_iou_rotated[double](&box1[0], &box2[0], 1)

