import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description='analyze each cls between score and map ')
    parser.add_argument('res_dir', help='results dir for dota format')
    parser.add_argument('score_thre', help='choose under score_thre to analyze')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    res_dir = Path(args.res_dir)
    score_thre = float(args.score_thre)
    cls_res = res_dir.glob( "*.txt")
    cls_score = dict()
    for cls_file in tqdm(cls_res):
        cls = cls_file.stem.split("_")[1]
        with open(cls_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                score = float(line[1])
                if score <= score_thre:
                    continue
                if cls not in cls_score.keys():
                    cls_score[cls] = [score]
                else:
                    cls_score[cls].append(score)
    print("mean score for each cls:\n")
    for i in cls_score.keys():
        print("{}:{}".format(i,np.mean(cls_score[i])))
if __name__ == '__main__':
    main()