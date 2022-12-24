import argparse
import os
import os.path as osp
from xml.dom.minidom import parse
from tqdm import tqdm

import mmcv
import torch
import cv2

from mmcv import Config, DictAction
from mmrotate.utils import setup_multi_processes



   
def parse_args():
    "parser arguments"
    parser = argparse.ArgumentParser(description='transform fair1m to dota format')
    parser.add_argument(
        "config",
        type=str,
        default=None,
        help='config for transform process')
    parser.add_argument(
        '--nproc', type=int, default=10, help='the procession number')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    
    return args


def solve_xml(src, tar):
    domTree = parse(src)
    rootNode = domTree.documentElement
    objects = rootNode.getElementsByTagName("objects")[0].getElementsByTagName("object")
    box_list=[]
    for obj in objects:
        name=obj.getElementsByTagName("possibleresult")[0].getElementsByTagName("name")[0].childNodes[0].data
        points=obj.getElementsByTagName("points")[0].getElementsByTagName("point")
        bbox=[]
        for point in points[:4]:
            x=point.childNodes[0].data.split(",")[0]
            y=point.childNodes[0].data.split(",")[1]
            bbox.append(float(x))
            bbox.append(float(y))
        box_list.append({"name":name, "bbox":bbox})
    
    file=open(tar,'w')
    print("imagesource:GoogleEarth",file=file)
    print("gsd:0.0",file=file)
    for box in box_list:
        ss=""
        for f in box["bbox"]:
            ss+=str(f)+" "
        name=  box["name"]
        name = name.replace(" ", "_")
        ss+=name+" 0"
        print(ss,file=file)
    file.close()


def fair_to_dota(in_path, out_path):
    if osp.exists(in_path):
        os.makedirs(out_path,exist_ok=True)
        os.makedirs(osp.join(out_path, "images"), exist_ok=True)
    
    tasks = []
    for root, dirs, files in os.walk(osp.join(in_path, "images")):
        for f in files:
            src = osp.join(root, f)
            tar = "P"+f[:-4].zfill(4)+".png"
            tar = osp.join(out_path,"images",tar)
            tasks.append((src, tar))
    print("processing images")
    for task in tqdm(tasks):
        file = cv2.imread(task[0], 1)
        cv2.imwrite(task[1], file)
    
    if (osp.exists(osp.join(in_path, "labelXml"))):
        os.makedirs(osp.join(out_path, "labelTxt"), exist_ok=True)
        tasks =[]
        for root, dirs, files in os.walk(osp.join(in_path, "labelXml")):
            for f in files:
                src = osp.join(root,f)
                tar = "P"+f[:-4].zfill(4)+".txt"
                tar = osp.join(out_path, 'labelTxt', tar)
                tasks.append((src, tar))
        print("processing labels")
        for task in tqdm(tasks):
            solve_xml(task[0], task[1])
    

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if (cfg.type == "FAIR" or cfg.type == 'Fair1M'):
        for task in cfg.convert_tasks:
            print('===================')
            print("convert to dota:", task)
            fair_to_dota(osp.join(cfg.source_fair_path, task), osp.join(cfg.target_dota_path, task))


if __name__ == "__main__":
    main()
