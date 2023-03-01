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
                    if ann_dir.split("/")[-3] == 'plane':
                        poly = list(map(float,l[1:9]))
                    else:
                        poly = list(map(float,l[:8]))
                    polys.append(poly)
    print("this dataset has {} objects".format(len(polys)))
    polys = torch.tensor(polys)             
    obbs = poly2obb_le90(polys)
    areas = obbs[:,2] * obbs[:,3]
    return areas.sort()
                    

def main():
    args = parse_args()
    ann_dir = args.ann_dir
    output_dir = args.output_dir
    areas,index = collect_areas(ann_dir)
    num = len(areas)
    if output_dir.split('/')[-1] == 'dior-r-ship' or output_dir.split('/')[-2] == 'ssdd':
        areas = areas[areas<6000] 
    elif output_dir.split('/')[-2] == 'ship':
        areas = areas[areas<35000]
    else:
        areas = areas[areas<10000]
    plt.hist(areas,bins=10)
    plt.title('areas analyze_{}'.format(len(areas)))
    plt.xlabel('areas')
    plt.ylabel('num')
    plt.savefig(osp.join(output_dir,('areas_{}.jpg').format(num)))
if __name__ == '__main__':
    main()