import os
from tqdm import tqdm


src_file = "test/20221007/0.6891/after_nms"
dst_file = "test/20221007/0.6891/convert"



if os.path.exists(src_file):
    os.makedirs(dst_file,exist_ok=True)
    for file in tqdm(os.listdir(src_file)):
        src = os.path.join(src_file,file)
        dst = os.path.join(dst_file,file)
        with open(src,'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                if len(line) < 10:
                    continue
                l = "P"+line[0].zfill(4)
                for i in range(1,10):
                    l += " "+line[i]
                l = l+'\n'
                with open(dst,'a') as ff:
                    ff.write(l)


