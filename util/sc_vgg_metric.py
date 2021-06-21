import os
import sys
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

def vgg_preprocess(tensor, vgg_normal_correct=False):
    if vgg_normal_correct:
        tensor = (tensor + 1) / 2
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst

class VGG19_feature_color_torchversion(nn.Module):
    ''' 
    NOTE: there is no need to pre-process the input 
    input tensor should range in [0,1]
    '''

    def __init__(self, pool='max', vgg_normal_correct=False, ic=3):
        super(VGG19_feature_color_torchversion, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        ''' 
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

class Dataset_(torch.utils.data.Dataset):
    def __init__(self, folder, dataset_mode):
        self.folder = folder
        self.dataset_mode = dataset_mode
        if dataset_mode == 'ade20k':
            self.label_folder = '/mnt/blob/Dataset/ADEChallengeData2016/images/validation'
            self.img_folder = '/mnt/blob/Dataset/ADEChallengeData2016/images/validation'
            imgs_name = os.listdir(self.img_folder)
            imgs_name = [it for it in imgs_name if it[-4:] == '.jpg']
            if 'SIMS' in self.folder:
                fd = open('/mnt/blob/Dataset/ADEChallengeData2016/outdoor.txt')
                outdoor = fd.readlines()
                fd.close()
                outdoor = [it.strip() for it in outdoor]
        elif 'celebahq' in dataset_mode:
            self.label_folder = '/mnt/blob/Dataset/CelebAMask-HQ/CelebAMask-HQ-mask-anno/all_parts_except_glasses'
            self.img_folder = '/mnt/blob/Dataset/CelebAMask-HQ/CelebA-HQ-img'
            fd = open('/mnt/blob/Dataset/CelebAMask-HQ/val.txt')
            imgs_name = fd.readlines()
            fd.close()
            imgs_name = [it.strip() + '.jpg' for it in imgs_name]
        elif dataset_mode == 'flickr':
            self.label_folder = '/mnt/blob/Dataset/Flickr/test/mask'
            self.img_folder = '/mnt/blob/Dataset/Flickr/test/images'
        elif dataset_mode == 'deepfashion':
            self.label_folder = '/mnt/blob/Dataset/DeepFashion/parsing'
            self.img_folder = '/mnt/blob/Dataset/DeepFashion/fid_256'
            imgs_name = os.listdir(self.img_folder)
        label_paths = []
        img_paths = []
        gen_img_paths = []
        j = 0
        for i in range(len(imgs_name)):
            name =imgs_name[i]
            if 'SIMS' in self.folder:
                if j < len(outdoor) and name.replace('.jpg', '.png') == outdoor[j]:
                    j += 1
                    name = str(j).zfill(4) + '.png'
            if not os.path.exists(os.path.join(self.folder, name)):
                name = name.replace('.jpg', '.png')
            if not os.path.exists(os.path.join(self.folder, name)):
                aa, bb = name.split('.')
                name = aa.zfill(5) + '.' + bb
            if not os.path.exists(os.path.join(self.folder, name)):
                aa, bb = name.split('.')
                name = aa + '_{}.' + bb
            if not os.path.exists(os.path.join(self.folder, name)):
                print(imgs_name[i] + ' not find!')
            else:
                label_paths.append(os.path.join(self.label_folder, self.tolabel_path(imgs_name[i])))
                img_paths.append(os.path.join(self.img_folder, imgs_name[i]))
                gen_img_paths.append(os.path.join(self.folder, name))
        self.label_paths = label_paths
        self.img_paths = img_paths
        self.gen_img_paths = gen_img_paths
        
        self.transform_img = transforms.Compose([transforms.Resize((256, 256)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5),
                                                                    (0.5, 0.5, 0.5))])
        self.transform_label = transforms.ToTensor()

    def tolabel_path(self, path):
        if 'celebahq' in self.dataset_mode:
            name, ext = path.split('.')
            path = name.zfill(5) + '.' + ext
        return path.replace('.jpg', '.png')

    def __getitem__(self, index):
        label = Image.open(self.label_paths[index]).convert('L').resize((256, 256), Image.NEAREST)
        label = self.transform_label(label)*255
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = self.transform_img(img)
        gen_img = Image.open(self.gen_img_paths[index]).convert('RGB')
        gen_img = self.transform_img(gen_img)
        return label, img, gen_img

    def __len__(self):
        return len(self.label_paths)

def cal_feat_dist(feat_1, feat_2, label, use_cos):
    label_u = torch.unique(label)
    theta_value = 0
    for k in range(len(label_u)):
        num_1 = (label == label_u[k]).sum().cpu().item()
        mask = torch.zeros_like(label)
        mask[label == label_u[k]] = 1
        mean_1 = (feat_1 * mask.expand(feat_1.shape[0], -1, -1)).sum(-1).sum(-1) / num_1
        mean_2 = (feat_2 * mask.expand(feat_2.shape[0], -1, -1)).sum(-1).sum(-1) / num_1
        cos_value = (normalize(mean_1.unsqueeze(0).cpu().numpy()) * normalize(mean_2.unsqueeze(0).cpu().numpy())).sum()
        if use_cos == 'cos':
            theta_value += cos_value * num_1 / (mask.shape[-1] * mask.shape[-2])
        else:
            theta = np.arccos(cos_value) / np.pi * 180
            theta_value += theta * num_1 / (mask.shape[-1] * mask.shape[-2])
    #theta_value = np.arccos(theta_value) / np.pi * 180
    return theta_value

# def cal_feat_dist(feat_1, feat_2, label):
#     cos_value = (feat_1.permute(1, 0).cpu().numpy() * feat_2.permute(1, 0).cpu().numpy()).sum(-1).mean()
#     theta = np.arccos(cos_value) / np.pi * 180
#     return theta

vgg = VGG19_feature_color_torchversion(vgg_normal_correct=True)
bs_dir = '/home/fangneng.zfn/projects/SFERT8/'
vgg.load_state_dict(torch.load(bs_dir + 'models/vgg19_conv.pth'))
vgg.cuda()
for param in vgg.parameters():
    param.requires_grad = False
dataset = Dataset_(sys.argv[1], sys.argv[2])

value = {'r32':[], 'r42':[], 'r52':[]}
layers = ['r32', 'r42', 'r52']
for i in range(len(dataset)):
    if i % 100 == 0:
        print('{} / {}'.format(i, len(dataset)))
    label, img, gen_img = dataset[i]
    label = label.cuda()
    img = img.cuda()
    gen_img = gen_img.cuda()
    img_features = vgg(img.unsqueeze(0), layers, preprocess=True)
    gen_img_features = vgg(gen_img.unsqueeze(0), layers, preprocess=True)
    for j in range(len(layers)):
        img_feat = F.interpolate(img_features[j], size=[256, 256], mode='nearest').squeeze()
        gen_img_feat = F.interpolate(gen_img_features[j], size=[256, 256], mode='nearest').squeeze()
        # img_feat = img_feat.view(img_feat.shape[0], -1)
        # img_feat = torch.div(img_feat, torch.norm(img_feat, p=None, dim=0, keepdim=True).expand_as(img_feat))
        # gen_img_feat = gen_img_feat.view(gen_img_feat.shape[0], -1)
        # gen_img_feat = torch.div(gen_img_feat, torch.norm(gen_img_feat, p=None, dim=0, keepdim=True).expand_as(gen_img_feat))
        theta = cal_feat_dist(img_feat, gen_img_feat, label, sys.argv[3])
        value[layers[j]].append(theta)

for key in value.keys():
    mean = np.mean(value[key])
    print(key + ':' + str(mean))

#python sc_vgg_metric.py /mnt/blob/Output/image_translation_methods/SPADE/output/test_per_img/ade20k_vae_v100 ade20k