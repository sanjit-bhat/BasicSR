import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import multiprocessing as mp

factor = 5
scale = 1/factor
rootdir_REDS = '/home/sanjit/BasicSR/datasets/REDS/train_sharp'
# rootdir_REDS4 = '/home/sanjit/BasicSR/datasets/REDS4/GT'

def process(data):
    subdir, dirs, files = data
    for file in files:
        path = os.path.join(subdir, file)
        folder = os.path.basename(subdir)
        image = Image.open(path).convert("RGB")
        h, w = image.size
        new_h = int(h * scale)
        new_w = int(w * scale)
        resized_image = image.resize((new_h, new_w),Image.BICUBIC)
        new_dir = os.path.join(rootdir_REDS, "..", f"train_sharp_bicubic/X{factor}", folder)
        # new_dir = os.path.join(rootdir_REDS, "..", f"sharp_bicubic/X{factor}", folder)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        dst_path = new_dir + "/" + file
        resized_image.save(dst_path,"PNG")
    print('done with:', subdir)

if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count() - 2)
    data = list(os.walk(rootdir))
    resources = pool.map(process, data)
