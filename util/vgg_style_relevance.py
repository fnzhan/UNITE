import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
# from util.util import vgg_preprocess
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
# sys.path.append('../')
from skimage.measure import compare_ssim

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

def get_transform(method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    # if 'resize' in opt.preprocess_mode:
    osize = [256, 256]
    transform_list.append(transforms.Resize(osize, interpolation=method))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)




torch.cuda.set_device(1)

bs_dir = '/home/fangneng.zfn/projects/SFERT8/'

vggnet_fix = VGG19_feature_color_torchversion(vgg_normal_correct=False)
vggnet_fix.load_state_dict(torch.load(bs_dir + 'models/vgg19_conv.pth'))
vggnet_fix.eval()
for param in vggnet_fix.parameters():
    param.requires_grad = False
vggnet_fix.to(1)

root_dir = '/data/vdd/fangneng.zfn/CoCosNet/ade20k0/'
# coco_dir = bs_dir + 'COCO-Stuff/val2017/'
# sv_dir = bs_dir + 'COCO-Stuff/val_vgg_feature/'
pre_dir = root_dir + 'pre/'
ref_dir = root_dir + 'gt/'

nms = os.listdir(pre_dir)
nm_im = len(nms)
idx = 0

m_all = 0
v_all = 0

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

for i in range(nm_im):
    pre_path = pre_dir + nms[i]
    pre = Image.open(pre_path).convert('RGB')
    # transform_image = get_transform()
    # gt_tensor = transform_image(gt).view(1, 3, 256, 256).cuda()

    ref_path = ref_dir + nms[i+1]
    ref = Image.open(ref_path).convert('RGB')
    # pre_tensor = transform_image(pre).view(1, 3, 256, 256).cuda()

    pre = np.array(pre).astype(np.float32)
    pre_tensor = torch.from_numpy(pre).permute(2, 0, 1).view(1, 3, 256, 256).cuda() / 255.0

    ref = np.array(ref).astype(np.float32)
    ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).view(1, 3, 256, 256).cuda() / 255.0

    pre_feat = vggnet_fix(pre_tensor, ['r12'], preprocess=True)[0]
    ref_feat = vggnet_fix(ref_tensor, ['r12'], preprocess=True)[0]


    pre_feat = pre_feat.view(1, 64, -1)
    ref_feat = ref_feat.view(1, 64, -1)

    pre_var = torch.var(pre_feat, dim=2)
    ref_var = torch.var(ref_feat, dim=2)

    pre_m = torch.mean(pre_feat, dim=2)
    ref_m = torch.mean(ref_feat, dim=2)

    # print (pre_var - ref_var)
    # print (pre_var.shape)

    v = cos(pre_var, ref_var)
    m = cos(pre_m, ref_m)
    print (m, v)

    v_all += v
    m_all += m

    # print (m)
    # print (1/0)
    # ssim += tmp
print ('******')
print (m_all / nm_im)
print (v_all / nm_im)

