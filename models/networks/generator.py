# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#08.09 change pad

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer, equal_lr
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SEACEResnetBlock as SEACEResnetBlock
from models.networks.architecture import Ada_SPADEResnetBlock as Ada_SPADEResnetBlock
from models.networks.architecture import Attention
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm1d

class SEACEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = 64
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        ic = opt.semantic_nc
        self.fc = nn.Conv2d(16 * nf, 16 * nf, 3, stride=1, padding=1)
        self.G_head_0 = SEACEResnetBlock(16 * nf, 16 * nf, opt, feat_nc=512, atten=False)

        self.G_middle_0 = SEACEResnetBlock(16 * nf, 16 * nf, opt, feat_nc=512, atten=True)
        self.G_middle_1 = SEACEResnetBlock(16 * nf, 16 * nf, opt, feat_nc=512, atten=False)

        self.G_up_0 = SEACEResnetBlock(16 * nf, 8 * nf, opt, feat_nc=256, atten=True)
        self.G_up_1 = SEACEResnetBlock(8 * nf, 4 * nf, opt, feat_nc=256, atten=False)
        self.attn = Attention(4 * nf, 'spectral' in opt.norm_G)

        self.G_out_0 = SEACEResnetBlock(4 * nf, 2 * nf, opt, feat_nc=128, atten=True)
        self.G_out_1 = SEACEResnetBlock(2 * nf, 1 * nf, opt, feat_nc=3, atten=False)

        self.conv_img1 = nn.Conv2d(1 * nf, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

        self.attn2 = Self_Attn(128, 'relu')
        self.attn3 = Self_Attn(256, 'relu')
        self.attn4 = Self_Attn(512, 'relu')

    def compute_latent_vector_size(self, opt):
        num_up_layers = 5
        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def forward(self, warp_out=None):

        seg_feat1, seg_feat2, seg_feat3, seg_feat4, seg_feat5, \
        ref_feat1, ref_feat2, ref_feat3, ref_feat4, ref_feat5, conf_map = warp_out
        #  3, 128, 256, 512, 512

        atten2 = self.attn2(seg_feat2, size=64)
        atten3 = self.attn3(seg_feat3, size=32)
        atten4 = self.attn4(seg_feat4, size=16)

        x = torch.cat((seg_feat5, ref_feat5), 1)
        x = F.interpolate(x, size=(self.sh, self.sw))
        x = self.fc(x)

        x = self.G_head_0(x, seg_feat5, ref_feat5, None, conf_map)
        x = self.up(x)
        x = self.G_middle_0(x, seg_feat4, ref_feat4, atten4, conf_map) # 16
        x = self.G_middle_1(x, seg_feat4, ref_feat4, None, conf_map)
        x = self.up(x)

        x = self.G_up_0(x, seg_feat3, ref_feat3, atten3, conf_map) # 32
        x = self.up(x)
        x = self.G_up_1(x, seg_feat3, ref_feat3, None, conf_map)
        x = self.up(x)

        x = self.attn(x) # 128,
        x = self.G_out_0(x, seg_feat2, ref_feat2, atten2, conf_map)
        x = self.up(x)
        x = self.G_out_1(x, seg_feat1, ref_feat1, None, conf_map)

        x = self.conv_img1(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1, padding=0, bias=False)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1, padding=0, bias=False)
        # nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, size):
        x = F.interpolate(x, size=(size, size), mode='nearest')
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        # proj_query = proj_query - proj_query.mean(dim=1, keepdim=True)

        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(m_batchsize, C, width, height)
        # out = self.gamma * out + x
        return attention




class AdaptiveFeatureGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks")
        return parser

    def __init__(self, opt):
        # TODO: kernel=4, concat noise, or change architecture to vgg feature pyramid
        super().__init__()
        self.opt = opt
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        nf = 64
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(opt.spade_ic, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt
        self.head_0 = Ada_SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)

        # self.head_1 = Ada_SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        if opt.adaptor_nonlocal:
            self.attn = Attention(8 * nf, False)
        self.G_middle_0 = Ada_SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        self.G_middle_1 = Ada_SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)

        # if opt.adaptor_res_deeper:
        # self.deeper0 = Ada_SPADEResnetBlock(8 * nf, 4 * nf, opt)
            # if opt.dilation_conv:
        # self.deeper1 = Ada_SPADEResnetBlock(8 * nf, 4 * nf, opt, dilation=2)
        self.deeper2 = Ada_SPADEResnetBlock(8 * nf, 4 * nf, opt, dilation=4)
        self.degridding0 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=2, dilation=2))
        # self.degridding1 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=1))
            # else:
            #     self.deeper1 = Ada_SPADEResnetBlock(4 * nf, 4 * nf, opt)
            #     self.deeper2 = Ada_SPADEResnetBlock(4 * nf, 4 * nf, opt)

    def forward(self, input, seg, multi=False):
        x = self.layer1(input)
        x = self.layer2(self.actvn(x))
        x2 = x

        x = self.layer3(self.actvn(x))
        x3 = x

        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.head_0(x, seg)
        x4 = x

        # x = self.head_1(x, seg)
        if self.opt.adaptor_nonlocal:
            x = self.attn(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x5 = x

        # x = self.deeper0(x, seg)
        # x = self.deeper1(x, seg)
        x = self.deeper2(x, seg)
        x = self.degridding0(x)
        # x = self.degridding1(x)

        if multi == True:
            return x2, x3, x4, x5, x
        else:
            return x


class DomainClassifier(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        nf = opt.ngf
        kw = 4 if opt.domain_rela else 3
        pw = int((kw - 1.0) / 2)
        self.feature = nn.Sequential(nn.Conv2d(4 * nf, 2 * nf, kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(2 * nf, affine=True),
                                nn.LeakyReLU(0.2, False),
                                nn.Conv2d(2 * nf, nf, kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(nf, affine=True),
                                nn.LeakyReLU(0.2, False),
                                nn.Conv2d(nf, int(nf // 2), kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(int(nf // 2), affine=True),
                                nn.LeakyReLU(0.2, False))  #32*8*8
        model = [nn.Linear(int(nf // 2) * 8 * 8, 100),
                SynchronizedBatchNorm1d(100, affine=True),
                nn.ReLU()]
        if opt.domain_rela:
            model += [nn.Linear(100, 1)]
        else:
            model += [nn.Linear(100, 2),
                      nn.LogSoftmax(dim=1)]
        self.classifier = nn.Sequential(*model)

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = self.mu
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]
