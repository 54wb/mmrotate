import argparse
import os
import os.path as osp
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from mmrotate.core.bbox.transforms import poly2obb_le90



def parse_args():
    parser = argparse.ArgumentParser(description='analyze areas for each class')
    parser.add_argument('ann_dir', help='annotations for dota format')
    parser.add_argument('output_dir', help='output a picture')
    args = parser.parse_args()
    return args

def collect_areas(ann_dir):
    polys = []
    for file in tqdm(os.listdir(ann_dir),desc="collecting objects:"):
        label_file = osp.join(ann_dir, file)
        with open(label_file, 'r') as f:
            for line in f.readlines():
                l = line.split(" ")
                if len(l)<8:
                    continue
                else:
                    poly = list(map(float,l[1:9]))
                    polys.append(poly)
    print("all dataset has {} objects".format(len(polys)))
    polys = torch.tensor(polys)             
    obbs = poly2obb_le90(polys)
    areas = obbs[:,2] * obbs[:,3]
    return areas.sort()
                    

def main():
    args = parse_args()
    ann_dir = args.ann_dir
    output_dir = args.output_dir
    areas,index = collect_areas(ann_dir)
    plt.hist(areas,bins=20)
    plt.title('areas analyze')
    plt.xlabel('areas')
    plt.ylabel('rate')
    areas
if __name__ == '__main__':
    main()