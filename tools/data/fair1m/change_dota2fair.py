import argparse
from pathlib import Path
import os
import shutil
from tqdm import tqdm
# import xml.etree.ElementTree as ET
from xml.dom.minidom import Document

img_list = []

def write_xml_noobject(img_name):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    source = doc.createElement('source')
    annotation.appendChild(source)

    filename = doc.createElement('filename')
    source.appendChild(filename)

    img_name = img_name + '.tif'
    filename_txt = doc.createTextNode(img_name)
    filename.appendChild(filename_txt)

    origin = doc.createElement('origin')
    source.appendChild(origin)

    origin_txt = doc.createTextNode('GF2/GF3')
    origin.appendChild(origin_txt)

    research = doc.createElement('research')
    annotation.appendChild(research)

    version = doc.createElement('version')
    research.appendChild(version)
    version_txt = doc.createTextNode('1.0')
    version.appendChild(version_txt)

    provider = doc.createElement('provider')
    research.appendChild(provider)
    provider_txt = doc.createTextNode('NWPU')
    provider.appendChild(provider_txt)

    author = doc.createElement('author')
    research.appendChild(author)
    author_txt = doc.createTextNode('lee')
    author.appendChild(author_txt)

    pluginname = doc.createElement('pluginname')
    research.appendChild(pluginname)
    pluginname_txt = doc.createTextNode('FAIR1M')
    pluginname.appendChild(pluginname_txt)

    pluginclass = doc.createElement('pluginclass')
    research.appendChild(pluginclass)
    pluginclass_txt = doc.createTextNode('object detection')
    pluginclass.appendChild(pluginclass_txt)

    time = doc.createElement('time')
    research.appendChild(time)
    time_txt = doc.createTextNode('2021-03')
    time.appendChild(time_txt)

    objects = doc.createElement('objects')
    annotation.appendChild(objects)

    object = doc.createElement('object')
    objects.appendChild(object)

    coordinate = doc.createElement('coordinate')
    object.appendChild(coordinate)
    coordinate_txt = doc.createTextNode('pixel')
    coordinate.appendChild(coordinate_txt)

    type = doc.createElement('type')
    object.appendChild(type)
    type_txt = doc.createTextNode('rectangle')
    type.appendChild(type_txt)

    description = doc.createElement('description')
    object.appendChild(description)
    description_txt = doc.createTextNode('None')
    description.appendChild(description_txt)

    possibleresult = doc.createElement('possibleresult')
    object.appendChild(possibleresult)

    name = doc.createElement('name')
    possibleresult.appendChild(name)
    name_txt = doc.createTextNode('Small Car')
    name.appendChild(name_txt)

    probability = doc.createElement('probability')
    possibleresult.appendChild(probability)
    probability_txt = doc.createTextNode('0.00')
    probability.appendChild(probability_txt)

    points = doc.createElement('points')    
    object.appendChild(points)

    for i in range(0,10,2):
        point = doc.createElement('point')
        points.appendChild(point)
        point_txt = doc.createTextNode('5'+','+'5')
        point.appendChild(point_txt)
    

    return doc



def write_xml(img_name, result_list):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    source = doc.createElement('source')
    annotation.appendChild(source)

    filename = doc.createElement('filename')
    source.appendChild(filename)

    img_name = img_name + '.tif'
    filename_txt = doc.createTextNode(img_name)
    filename.appendChild(filename_txt)

    origin = doc.createElement('origin')
    source.appendChild(origin)

    origin_txt = doc.createTextNode('GF2/GF3')
    origin.appendChild(origin_txt)

    research = doc.createElement('research')
    annotation.appendChild(research)

    version = doc.createElement('version')
    research.appendChild(version)
    version_txt = doc.createTextNode('1.0')
    version.appendChild(version_txt)

    provider = doc.createElement('provider')
    research.appendChild(provider)
    provider_txt = doc.createTextNode('NWPU')
    provider.appendChild(provider_txt)

    author = doc.createElement('author')
    research.appendChild(author)
    author_txt = doc.createTextNode('lee')
    author.appendChild(author_txt)

    pluginname = doc.createElement('pluginname')
    research.appendChild(pluginname)
    pluginname_txt = doc.createTextNode('FAIR1M')
    pluginname.appendChild(pluginname_txt)

    pluginclass = doc.createElement('pluginclass')
    research.appendChild(pluginclass)
    pluginclass_txt = doc.createTextNode('object detection')
    pluginclass.appendChild(pluginclass_txt)

    time = doc.createElement('time')
    research.appendChild(time)
    time_txt = doc.createTextNode('2021-03')
    time.appendChild(time_txt)

    objects = doc.createElement('objects')
    annotation.appendChild(objects)

    for result in result_list:
        r = result.split()
        cls_name = r[0].replace('_', ' ')
        score = str(round(float(r[1]),2))
        

        object = doc.createElement('object')
        objects.appendChild(object)

        coordinate = doc.createElement('coordinate')
        object.appendChild(coordinate)
        coordinate_txt = doc.createTextNode('pixel')
        coordinate.appendChild(coordinate_txt)

        type = doc.createElement('type')
        object.appendChild(type)
        type_txt = doc.createTextNode('rectangle')
        type.appendChild(type_txt)

        description = doc.createElement('description')
        object.appendChild(description)
        description_txt = doc.createTextNode('None')
        description.appendChild(description_txt)

        possibleresult = doc.createElement('possibleresult')
        object.appendChild(possibleresult)



        name = doc.createElement('name')
        possibleresult.appendChild(name)
        name_txt = doc.createTextNode(cls_name)
        name.appendChild(name_txt)

        probability = doc.createElement('probability')
        possibleresult.appendChild(probability)
        probability_txt = doc.createTextNode(score)
        probability.appendChild(probability_txt)

        points = doc.createElement('points')
        object.appendChild(points)

        for i in range(2,10,2):
            point = doc.createElement('point')
            points.appendChild(point)
            point_txt = doc.createTextNode(r[i]+','+r[i+1])
            point.appendChild(point_txt)

        point = doc.createElement('point')
        points.appendChild(point)
        point_txt = doc.createTextNode(r[2]+','+r[3])
        point.appendChild(point_txt)
    return doc

def parse_args():
    parser = argparse.ArgumentParser(description='convert dota format to fair format')
    parser.add_argument(
        "dota_dir",
        type=str,
        default=None,
        help='the dota results'
    )
    parser.add_argument(
        "fair_dir",
        type=str,
        default=None,
        help='wishing to write fair dir'
    )
    args = parser.parse_args()
    return args

args = parse_args()

ori_img_path = Path('/disk0/lwb/datasets/Fair1m1_0/test/images/')#存放fair1m的test图像的路径

dota_path =  Path(args.dota_dir) #dota格式的结果文件路径
write_path = Path(args.fair_dir) #预期存放fair1m格式文件的路径
write_path = write_path/"test/"

if os.path.exists(write_path):
    print("文件夹已存在,是否覆盖:[y/n]")
    if input().lower() == 'y':
        shutil.rmtree(write_path)
    else:
        exit(0)

os.makedirs(write_path)
dota_list = dota_path.glob("*.txt") 

dict_content = {}


for cls_txt in tqdm(dota_list,desc="Reading dota results:"):
    cls = cls_txt.stem.split("1_")[1]
    with open(cls_txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        l = line.strip().split()
        res = l[0][1:].lstrip('0')
        if res == '':
            res = '0'
        img_list.append(res)
        if res not in dict_content.keys():
            dict_content[res] = [cls + ' ' + " ".join(l[1:])]

        else:
            dict_content[res].append(cls + ' ' + " ".join(l[1:]))

print("the number of images can be detected is {}".format(len(set(img_list))))
ori_img_list = ori_img_path.glob("*.tif")
for im in ori_img_list:
    im = im.stem
    if im not in set(img_list):
        doc = write_xml_noobject(im)
        tempfile = write_path / (im + ".xml")
        f = open(tempfile, "w")
        doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()
        # print(im)

for img_name in tqdm(dict_content.keys(),desc='Writing for fair1m results:'):


    doc = write_xml(img_name,dict_content[img_name])

    tempfile = write_path / (img_name + ".xml")
    f = open(tempfile, "w")
    doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
    f.close()





