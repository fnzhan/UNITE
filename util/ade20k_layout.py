# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from data.base_dataset import get_params, get_transform
import numpy as np
from PIL import Image
import torch
import scipy.io as sio
import cv2
from util import util


class ADE20KLAYOUTDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=150)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'
        subfolder = 'validation' if opt.phase == 'test' else 'training'
        cache = False if opt.phase == 'test' else True
        all_images = sorted(make_dataset(root + '/' + subfolder, recursive=True, read_cache=cache, write_cache=False))

        image_paths = []
        label_paths = []
        for p in all_images:
            if '_%s_' % phase not in p:
                continue
            if p.endswith('.jpg'):
                image_paths.append(p)
            elif p.endswith('.png'):
                label_paths.append(p)

        return label_paths, image_paths

    def get_ref(self, opt):
        extra = '_test' if opt.phase == 'test' else ''
        with open('./data/ade20k_ref{}.txt'.format(extra)) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            if opt.phase == 'test':
                val = items[1:]
            else:
                val = [items[1], items[-1]]
            ref_dict[key] = val
        train_test_folder = ('training', 'validation')
        return ref_dict, train_test_folder

    def get_label_tensor(self, path):

        label = Image.open(path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        fine_label_tensor = transform_label(label) * 255.0
        fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1), fine_label_tensor.size(2))

        label_np, box_mask = self.get_onehot_box_tensor(fine_label_tensor)
        label_tensor = label_np.float()
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        label_tensor = torch.cat((box_mask, label_tensor), dim=0)
        return label_tensor, params

    def get_onehot_box_tensor(self, fine_label_tensor):
        label_np = fine_label_tensor.data.cpu().numpy()
        # label_np = label_np-1
        # objects_stuff_mat = sio.loadmat('data/objectSplit35-115.mat')
        # objects_stuff_list = objects_stuff_mat['stuffOrobject']
        # object_list = []
        # for object_id in range(len(objects_stuff_list)):
        #     if objects_stuff_list[object_id] == 2:
        #         object_list.append(object_id)

        save_label = np.zeros(label_np.shape)
        label_onehot = np.zeros((1,151,) + label_np.shape)
        label_onehot_tensor = torch.from_numpy(label_onehot).float()
        label_seq = np.unique(label_np)

        box_map = np.ones((label_np.shape[0], label_np.shape[1], 3)) * 255.0
        box_map = cv2.rectangle(box_map, (0, 0), (255, 255), (0,0,0), 1)

        obj_count = 0
        for label_id in label_seq:
            label_id = int(label_id)
            if label_id > 0:
                temp_label = np.zeros(label_np.shape)
                temp_label[label_np==label_id] = label_id
                temp_label = temp_label.astype('uint8')
                contours, hierarchy = cv2.findContours(temp_label,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                for conid in range(len(contours)):
                    mask = np.zeros(label_np.shape, dtype="uint8")
                    cv2.drawContours(mask, contours, conid, int(label_id), -1)
                    i, j = np.where(mask==label_id)
                    indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                                          np.arange(min(j), max(j) + 1), indexing='ij')
                    y1, x1, y2, x2 = min(i), min(j), max(i), max(j)
                    if ((x2-x1) * (y2-y1)) < 30:
                        continue
                    save_label[indices] = label_id
                    # save_label[label_np==label_id] = obj
                    save_label_batch = save_label.reshape((1,1,)+save_label.shape)
                    save_label_tensor = torch.from_numpy(save_label_batch).long()
                    label_onehot_tensor.scatter_(1, save_label_tensor, 1.0)

                    color = util.id2rgb(label_id).tolist()
                    box_map = cv2.rectangle(box_map, (x1, y1), (x2, y2), color, 2)
                    obj_count += 1
        box_mask = torch.from_numpy(box_map).permute(2, 0, 1).float()
        label_onehot_tensor = label_onehot_tensor.squeeze(0).float()

        return label_onehot_tensor, box_mask

