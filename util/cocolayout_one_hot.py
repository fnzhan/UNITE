"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from collections import defaultdict
import json
from PIL import Image
import numpy as np
import torch
import cv2
import util
import os
# import matplotlib
import matplotlib.cm


class one_hot():
    def process(self):
        root = '/data/vdd/fangneng.zfn/datasets/COCO-Stuff'
        phase = 'train'
        subfolder = 'val2017' if phase == 'test' else 'train2017'
        # cache = False if opt.phase == 'test' else True
        # all_images = sorted(make_dataset(root + '/' + subfolder, recursive=True, read_cache=cache, write_cache=False))
        npy_dir = root + '/train_vgg_feature/'
        nms = os.listdir(npy_dir)
        nms = [nm.replace('.npy', '.jpg') for nm in nms]

        im_dir = root + '/image/' + subfolder + '/'
        # nms = os.listdir(im_dir)
        im_paths = [im_dir + nm for nm in nms][:40000]
        label_paths = [im_path.replace('.jpg', '.png').replace('image', 'label') for im_path in im_paths]

        inst_path = root + '/annotations/' + 'instances_{}.json'.format(subfolder)
        stuff_path = root + '/annotations/' + 'stuff_{}.json'.format(subfolder)

        self.cmap = getCMap()
        self.process_json(inst_path, stuff_path)
        idx = 0
        for path in label_paths:
            label_tensor = self.get_label_tensor(path)
            one_hot_tensor = label_tensor.detach().cpu().numpy()
            sv_path = root + '/one_hot/' + subfolder + '/' + os.path.basename(path).replace('.png', '.npy')
            np.save(sv_path, one_hot_tensor)
            idx += 1
            print (idx)


        # return label_paths, im_paths

    def process_json(self, inst_path, stuff_path):
        with open(inst_path, 'r') as f:
            inst_data = json.load(f)
        with open(stuff_path, 'r') as f:
            stuff_data = json.load(f)

        self.im_ids, self.id2nm,self.nm2id, self.id2sz = [], {}, {}, {}
        for im_data in inst_data['images']:
            im_id, nm = im_data['id'], im_data['file_name']
            width, height = im_data['width'], im_data['height']
            self.im_ids.append(im_id)
            self.id2nm[im_id] = nm
            self.nm2id[nm] = im_id
            self.id2sz[im_id] = (width, height)

        self.vocab = {'objnm2idx': {}, 'prednm2idx': {}}
        oid2nm, inst_nms, stuff_nms = {}, [], []
        for category_data in inst_data['categories']:
            category_id, category_name = category_data['id'], category_data['name']
            inst_nms.append(category_name)
            oid2nm[category_id] = category_name
            self.vocab['objnm2idx'][category_name] = category_id
        for category_data in stuff_data['categories']:
            category_id, category_name = category_data['id'], category_data['name']
            stuff_nms.append(category_name)
            oid2nm[category_id] = category_name
            self.vocab['objnm2idx'][category_name] = category_id
        category_whitelist = set(inst_nms) | set(stuff_nms)

        # Add object data from instances
        self.id2obj = defaultdict(list)
        for object_data in inst_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.id2sz[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > 0.02
            object_name = oid2nm[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other'
            if box_ok and category_ok and other_ok:
                self.id2obj[image_id].append(object_data)

        # Add object data from stuff
        image_ids_with_stuff = set()
        for object_data in stuff_data['annotations']:
            image_id = object_data['image_id']
            image_ids_with_stuff.add(image_id)
            _, _, w, h = object_data['bbox']
            W, H = self.id2sz[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > 0.02
            object_name = oid2nm[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other'
            if box_ok and category_ok and other_ok:
                self.id2obj[image_id].append(object_data)


        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['objnm2idx']['__image__'] = 0

        # Build object_idx_to_name
        name_to_idx = self.vocab['objnm2idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx2nm = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['objnm2idx'].items():
            idx2nm[idx] = name
        self.vocab['oid2nm'] = idx2nm
        self.num_objects = len(self.vocab['oid2nm'])

        # Prune images that have too few or too many objects
        new_image_ids, total_objs = [], 0
        for im_id in self.im_ids:
            num_objs = len(self.id2obj[im_id])
            total_objs += num_objs
            if 3 <= num_objs <= 12:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

        self.vocab['predidx2nm'] = ['__in_image__', 'left of',
            'right of', 'above', 'below', 'inside', 'surrounding']
        self.vocab['prednm2idx'] = {}
        for idx, name in enumerate(self.vocab['predidx2nm']):
            self.vocab['prednm2idx'][name] = idx



    def get_label_tensor(self, path):
        nm = os.path.basename(path)
        im_id = self.nm2id[nm.replace('.png', '.jpg')]

        label = Image.open(path)
        # params = get_params(self.opt, label.size)

        WW, HH = label.size
        H, W = 256, 256

        box_map = np.ones((H, W, 3)) * 255.0
        box_map = cv2.rectangle(box_map, (0, 0), (255, 255), (0, 0, 0), 1)

        # mask = np.zeros((H, W))
        label_onehot = np.zeros((183, H, W))
        label_onehot_tensor = torch.from_numpy(label_onehot).float()

        objs, boxes, masks = [], [], []
        for object_data in self.id2obj[im_id]:
            objs.append(object_data['category_id'])
            x, y, w, h = object_data['bbox']
            x0 = x / WW
            y0 = y / HH
            x1 = (x + w) / WW
            y1 = (y + h) / HH
            boxes.append(torch.FloatTensor([x0, y0, x1, y1]))
            # This will give a numpy array of shape (HH, WW)
            mask = torch.zeros(H, W)

            ymin, ymax = round(y0 * H), max(round(y0 * H) + 1, round(y1 * H))
            xmin, xmax = round(x0 * W), max(round(x0 * W) + 1, round(x1 * W))

            mask[ymin:ymax, xmin:xmax] = object_data['category_id']
            mask = mask.reshape((1, H, W)).long()
            # mask_tensor = torch.from_numpy(mask).long()
            label_onehot_tensor.scatter_(0, mask, 1.0)

            if object_data['category_id'] > 0:
                color = self.cmap[int(object_data['category_id'])].tolist()
                box_map = cv2.rectangle(box_map, (xmin, ymin), (xmax, ymax), color, 2)
        # print (objs)
        # print (label_onehot_tensor.shape)

        # for i in range(183):
        #     map = label_onehot_tensor[i].detach().cpu().numpy()
        #     map = Image.fromarray((map*255.0).astype('uint8'))
        #     map.save('tmp/{}.jpg'.format(i))
        # label.save('label.jpg')
        # box_map = Image.fromarray(box_map.astype('uint8'))
        # box_map.save('box.jpg')
        # print (1/0)

        box_map = torch.from_numpy(box_map).permute(2, 0, 1)
        label_tensor = torch.cat((box_map.float(), label_onehot_tensor.float()), dim=0)
        return label_tensor

def getCMap(stuffStartId=1, stuffEndId=182, cmapName='jet', addThings=True, addUnlabeled=True, addOther=True):
    '''
    Create a color map for the classes in the COCO Stuff Segmentation Challenge.
    :param stuffStartId: (optional) index where stuff classes start
    :param stuffEndId: (optional) index where stuff classes end
    :param cmapName: (optional) Matlab's name of the color map
    :param addThings: (optional) whether to add a color for the 91 thing classes
    :param addUnlabeled: (optional) whether to add a color for the 'unlabeled' class
    :param addOther: (optional) whether to add a color for the 'other' class
    :return: cmap - [c, 3] a color map for c colors where the columns indicate the RGB values
    '''

    # Get jet color map from Matlab
    labelCount = stuffEndId - stuffStartId + 1
    cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    # Add black (or any other) color for each thing class
    if addThings:
        thingsPadding = np.zeros((stuffStartId - 1, 3))
        cmap = np.vstack((thingsPadding, cmap))

    # Add black color for 'unlabeled' class
    if addUnlabeled:
        cmap = np.vstack(((0.0, 0.0, 0.0), cmap))

    # Add yellow/orange color for 'other' class
    # if addOther:
    #     cmap = np.vstack((cmap, (1.0, 0.843, 0.0)))

    return cmap * 255.0

x = one_hot()
x.process()