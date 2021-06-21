from skimage.measure import compare_ssim
# from scipy.misc import imread
import numpy as np
import os
import heapq
import random

bs_dir = '/data/vdd/fangneng.zfn/datasets/'
npy_dir = bs_dir + 'COCO-Stuff/val_vgg_feature/'
sv_path = bs_dir + 'COCO-Stuff/coco_ref_test.txt'
f = open(sv_path, 'w')

nms = os.listdir(npy_dir)
n = len(nms)
for i in range(n):
    npy_path = npy_dir + nms[i]
    feature = np.load(npy_path)
    dis = -1000000
    ref_nm = nms[i].replace('npy', 'jpg')

    # retrieve_idx = random.sample(range(0, n), )
    for j in range(n):
        if j != i:
            npy_path2 = npy_dir + nms[j]
            feature2 = np.load(npy_path2)
            ssim = compare_ssim(feature, feature2, multichannel=True)
            if ssim > dis:
                dis = ssim
                ref_nm = nms[j].replace('npy', 'jpg')
    line = nms[i].replace('npy', 'jpg')
    # for k in range(10):
    line += ',' + ref_nm
    line += '\n'
    f.write(line)

    print (i)
