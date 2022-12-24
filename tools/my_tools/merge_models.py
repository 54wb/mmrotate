import cv2
import os
import argparse
import numpy as np
from loguru import logger
from glob import glob
from tqdm import tqdm
from utils import *

def getOpts():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", "--src_results_dirs", nargs="+", type=str, default=[])
    parser.add_argument("-dst", "--dst_results_dir", type=str, default=None) 
    parser.add_argument("--image_suffix", type=str, default="tif")
    parser.add_argument("--class_ignore", nargs="+", type=str, default=[])
    parser.add_argument("--options", nargs="+", action=DictAction)
    return parser.parse_args()



def main():
    args = getOpts()
    src_results_dirs = args.src_results_dirs
    dst_results_dir = args.dst_results_dir 
    os.makedirs(dst_results_dir, exist_ok=True)
    kwargs = {} if args.options is None else args.options
    nms_iou_thr = kwargs.pop("nms_iou_thr", 0.1)
    score_thr = kwargs.pop("score_thr", 0.05)
    ignore_class = kwargs.pop("ignore_class", [])

    ignore_class = list(map(str, ignore_class))

    results_collect = {}
    logger.info("collect results...")
    for src_results_dir in tqdm(src_results_dirs):
        src_results_collect = collect_results(src_results_dir, ignore_class)
        for image_id, results in src_results_collect.items():
            if image_id in results_collect:
                for class_name, result in results.items():
                    if class_name in results_collect[image_id]:
                        results_collect[image_id][class_name] \
                            = np.concatenate((results_collect[image_id][class_name], result), axis=0)
                    else:
                        results_collect[image_id][class_name] = result
            else:
                results_collect[image_id] = results

    logger.info("manage the image info...")
    results_per_class = {}
    for image_id, results in tqdm(results_collect.items()):
        for class_name, result in results.items():
            bboxes, scores = result[:, :8], result[:, 8]
            bboxes = poly2obb_np(bboxes)
            keep = obb_nms(bboxes, scores, iou_thr=nms_iou_thr, score_thr=score_thr, mode="obb")
            result = result[keep]
            results_collect[image_id][class_name] = result
            if class_name in results_per_class:
                if image_id in results_per_class[class_name]:
                    results_per_class[class_name][image_id] = np.concatenate((results_per_class[class_name][image_id], result), axis=0)
                else:
                    results_per_class[class_name][image_id] = result
            else:
                results_per_class[class_name] = {image_id: result}

    class_names = results_per_class.keys()
    logger.info(f"save results[{class_names}]...")
    for class_name, results in tqdm(results_per_class.items()):
        dst_result_file = os.path.join(dst_results_dir, class_name + ".txt")
        with open(dst_result_file, "w") as fw:
            for image_id, image_info in results.items():
                for info in image_info:
                    msg = []
                    bbox, score = info[:8].tolist(), info[8]
                    msg.append(image_id)
                    msg.append(str(score))
                    bbox_str_list = list(map(str, bbox))
                    msg.extend(bbox_str_list)
                    fw.write(" ".join(msg) + "\n")
    

if __name__ == "__main__":
    main()


