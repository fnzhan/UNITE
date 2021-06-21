# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import data
import numpy as np
from util.util import masktorgb
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel

opt = TestOptions().parse()
   
torch.manual_seed(0)
dataloader = data.create_dataloader(opt)
dataloader.dataset[0]

model = Pix2PixModel(opt)
model.eval()

save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output')

# test
for i, data_i in enumerate(dataloader):
    print('{} / {}'.format(i, len(dataloader)))
    if i * opt.batchSize >= 100:
        break
    imgs_num = data_i['label'].shape[0]
    #data_i['stage1'] = torch.ones_like(data_i['stage1'])
    
    out = model(data_i, mode='inference')

    data_i['label'] = data_i['label'][:, :3, :, :]
    label = data_i['label'].expand(-1, 3, -1, -1).float() / data_i['label'].max()
    imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), out['warp_tmp'].cpu()), 0)
    vutils.save_image(imgs, './output/euc_match/' + 'tmp_{}.png'.format(i), nrow=4, padding=0, normalize=True)
    # print(1 / 0)

    # if not os.path.exists(save_root + '/test/' + opt.name + '/pre'):
    #     os.makedirs(save_root + '/test/' + opt.name + '/pre')
    #     os.makedirs(save_root + '/test/' + opt.name + '/gt')
    #
    #
    # if opt.dataset_mode == 'deepfashion':
    #     label = data_i['label'][:,:3,:,:]
    # elif opt.dataset_mode == 'celebahqedge':
    #     label = data_i['label'].expand(-1, 3, -1, -1).float()
    # else:
    #     label = masktorgb(data_i['label'].cpu().numpy())
    #     label = torch.from_numpy(label).float() / 128 - 1
    #
    # # imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), out['fake_image'].data.cpu(), data_i['image'].cpu()), 0)
    # pre = out['fake_image'].data.cpu()
    # gt = data_i['image'].cpu()
    # try:
    #     pre = (pre + 1) / 2
    #     vutils.save_image(pre, save_root + '/test/' + opt.name + '/pre/' + str(i) + '.png',
    #             nrow=imgs_num, padding=0, normalize=False)
    #
    #     gt = (gt + 1) / 2
    #     vutils.save_image(gt, save_root + '/test/' + opt.name + '/gt/' + str(i) + '.png',
    #                       nrow=imgs_num, padding=0, normalize=False)
    # except OSError as err:
    #     print(err)
