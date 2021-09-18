"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.base_dataset import get_params, get_transform
from data.image_folder import make_dataset
from collections import defaultdict
import json
from PIL import Image
import numpy as np
import torch
import cv2
from util import util
import os

class CocolayoutDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--coco_no_portraits', action='store_true')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=182)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)
        return parser


    def get_paths(self, opt):
        root = opt.dataroot
        subfolder = 'val2017' if opt.phase == 'test' else 'train2017'

        inst_path = root + '/annotations/' + 'instances_{}.json'.format(subfolder)
        stuff_path = root + '/annotations/' + 'stuff_{}.json'.format(subfolder)

        self.cmap = util.getCMap()
        refine_set = self.process_json(inst_path, stuff_path)

        npy_dir = root + '/train_vgg_feature/'
        nms = os.listdir(npy_dir)

        refine_nms = []
        for nm in nms:
            nm = nm.replace('.npy', '.jpg')
            if nm in refine_set:
                refine_nms.append(nm)
        # print (len(refine_nms))

        im_dir = root + '/image/' + subfolder + '/'
        im_paths = [im_dir + nm for nm in refine_nms]
        label_paths = [im_path.replace('.jpg', '.png').replace('image', 'label') for im_path in im_paths]

        return label_paths, im_paths


    def get_ref(self, opt):
        extra = '_test' if opt.phase == 'test' else ''
        with open('./data/coco_ref{}.txt'.format(extra)) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            if opt.phase == 'test':
                val = items[1:]
            else:
                val = [items[1], items[2]]
            ref_dict[key] = val
        train_test_folder = ('', '')
        return ref_dict, train_test_folder

    def process_json(self, inst_path, stuff_path):
        with open(inst_path, 'r') as f:
            inst_data = json.load(f)
        with open(stuff_path, 'r') as f:
            stuff_data = json.load(f)

        self.im_ids, self.im_nms, self.id2nm,self.nm2id, self.id2sz = [], [], {}, {}, {}
        for im_data in inst_data['images']:
            im_id, nm = im_data['id'], im_data['file_name']
            width, height = im_data['width'], im_data['height']
            self.im_ids.append(im_id)
            self.im_nms.append(nm)
            self.id2nm[im_id] = nm
            self.nm2id[nm] = im_id
            self.id2sz[im_id] = (width, height)

        # self.vocab = {'objnm2idx': {}, 'prednm2idx': {}}
        oid2nm, inst_nms, stuff_nms = {}, [], []
        for category_data in inst_data['categories']:
            category_id, category_name = category_data['id'], category_data['name']
            inst_nms.append(category_name)
            oid2nm[category_id] = category_name
            # self.vocab['objnm2idx'][category_name] = category_id
        for category_data in stuff_data['categories']:
            category_id, category_name = category_data['id'], category_data['name']
            stuff_nms.append(category_name)
            oid2nm[category_id] = category_name
            # self.vocab['objnm2idx'][category_name] = category_id
        category_whitelist = set(inst_nms) | set(stuff_nms)

        # Add object data from instances
        self.nm2obj = defaultdict(list)
        for object_data in inst_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.id2sz[image_id]
            nm = self.id2nm[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > 0.02
            object_name = oid2nm[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other'
            if box_ok and category_ok and other_ok:
                self.nm2obj[nm].append(object_data)

        # Add object data from stuff
        image_ids_with_stuff = set()
        for object_data in stuff_data['annotations']:
            image_id = object_data['image_id']
            image_ids_with_stuff.add(image_id)
            _, _, w, h = object_data['bbox']
            W, H = self.id2sz[image_id]
            nm = self.id2nm[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > 0.02
            object_name = oid2nm[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other'
            if box_ok and category_ok and other_ok:
                self.nm2obj[nm].append(object_data)

        # Prune images that have too few or too many objects
        refine_set = []
        for im_nm in self.im_nms:
            if 3 <= len(self.nm2obj[im_nm]) <= 8:
                refine_set.append(im_nm)
        return refine_set



    def get_label_tensor(self, path):

        nm = os.path.basename(path).replace('.png', '.jpg')
        # im_id = self.nm2id[nm.replace('.png', '.jpg')]
        WW, HH = Image.open(path).size
        params = get_params(self.opt, (WW, HH))
        H, W = 256, 256

        box_map = np.ones((H, W, 3)) * 255.0
        box_map = cv2.rectangle(box_map, (0, 0), (255, 255), (0, 0, 0), 1)
        label_onehot_tensor = torch.from_numpy(np.zeros((183, H, W))).float()

        objs, boxes, masks = [], [], []
        for object_data in self.nm2obj[nm]:
            objs.append(object_data['category_id'])
            x, y, w, h = object_data['bbox']
            x0, y0, x1, y1 = x / WW, y / HH, (x + w) / WW, (y + h) / HH
            boxes.append(torch.FloatTensor([x0, y0, x1, y1]))
            mask = torch.zeros(H, W)
            ymin, ymax = round(y0 * H), max(round(y0 * H) + 1, round(y1 * H))
            xmin, xmax = round(x0 * W), max(round(x0 * W) + 1, round(x1 * W))

            mask[ymin:ymax, xmin:xmax] = object_data['category_id']
            mask = mask.reshape((1, H, W)).long()
            label_onehot_tensor.scatter_(0, mask, 1.0)

            if object_data['category_id'] > 0:
                color = self.cmap[int(object_data['category_id'])].tolist()
                box_map = cv2.rectangle(box_map, (xmin, ymin), (xmax, ymax), color, 2)

        # for i in range(183):
        #     map = label_onehot_tensor[i].detach().cpu().numpy()
        #     map = Image.fromarray((map*255.0).astype('uint8'))
        #     map.save('tmp/{}.jpg'.format(i))
        # label.save('label.jpg')
        # box_map = Image.fromarray(box_map.astype('uint8'))
        # box_map.save('box.jpg')
        # print (1/0)

        # seg_map = Image.open(path)
        # transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        # seg_tensor = transform_label(seg_map)
        # label_onehot_tensor[0,:,:] = seg_tensor


        box_map = torch.from_numpy(box_map).permute(2, 0, 1)
        label_tensor = torch.cat((box_map.float(), label_onehot_tensor.float()), dim=0)

        return label_tensor, params
