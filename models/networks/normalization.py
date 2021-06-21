# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
try:
    import apex
    from apex import amp
except:
    print('apex not found')
    pass

def PositionalNorm2d(x, epsilon=1e-5):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output

class SEACE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, PONO=False, use_apex=False, feat_nc=None, atten=False):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.norm_nc, self.atten = norm_nc, atten

        if PONO:
            self.param_free_norm = PositionalNorm2d
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            if use_apex:
                self.param_free_norm = apex.parallel.SyncBatchNorm(norm_nc, affine=False)
            else:
                self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('not a recognized norm')
        nhidden, pw = 128, ks // 2
        self.nhidden = nhidden
        self.feat_nc = feat_nc
        if feat_nc == 3:
            self.seg_nc = label_nc
        else:
            self.seg_nc = feat_nc
        self.seg_shared = nn.Sequential(
                nn.Conv2d(self.seg_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        self.ref_shared = nn.Sequential(
            nn.Conv2d(feat_nc, feat_nc, kernel_size=ks, padding=pw), nn.ReLU())
        self.ref_shared2 = nn.Sequential(
            nn.Conv2d(feat_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        # self.ref_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        # self.ref_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.coef = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, seg_map, ref_map, atten_map, conf_map):
        normalized = self.param_free_norm(x)
        b, _, h, w = x.size()

        seg_map = F.interpolate(seg_map, size=(h, w), mode='nearest')
        ref_map = F.interpolate(ref_map, size=(h, w), mode='nearest')
        conf_map = F.interpolate(conf_map, size=(h, w), mode='nearest')

        conf_map = conf_map.repeat(1, self.nhidden, 1, 1)


        seg_feat = self.seg_shared(seg_map)
        if self.atten:
            if h <= 64:
                ref_feat = self.ref_shared(ref_map)
                # print (atten_map.shape, ref_feat.shape)
                ref_aggr = torch.bmm(ref_feat.view(b, self.feat_nc, h*w), atten_map.permute(0, 2, 1))
                ref_aggr = ref_aggr.view(b, self.feat_nc, h, w)
            else:
                ref_map_aggr = F.interpolate(ref_map, size=(64, 64), mode='nearest')
                ref_feat = self.ref_shared(ref_map_aggr)

                ref_aggr = torch.bmm(ref_feat.view(b, self.feat_nc, 64 * 64), atten_map.permute(0, 2, 1))
                ref_aggr = ref_aggr.view(b, self.feat_nc, 64, 64)
                ref_aggr = F.interpolate(ref_aggr, size=(h, w), mode='nearest')
            ref_map = self.coef * ref_aggr + ref_map
        ref_feat = self.ref_shared2(ref_map)
        # ref_feat = F.interpolate(ref_feat, size=(h, w), mode='nearest')
        # print (conf_map.shape, seg_feat.shape, ref_feat.shape)
        feat = seg_feat * (1 - conf_map) + ref_feat * conf_map
        feat = F.interpolate(feat, size=(h, w), mode='nearest')
        gamma = self.gamma(feat)
        beta = self.beta(feat)

        out = normalized * (1 + gamma) + beta

        return out





class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, PONO=False, use_apex=False):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.pad_type = 'nozero'

        if PONO:
            self.param_free_norm = PositionalNorm2d
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            if use_apex:
                self.param_free_norm = apex.parallel.SyncBatchNorm(norm_nc, affine=False)
            else:
                self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        if self.pad_type != 'zero':
            self.mlp_shared = nn.Sequential(
                nn.ReflectionPad2d(pw),
                nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=0),
                nn.ReLU()
            )
            self.pad = nn.ReflectionPad2d(pw)
            self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
            self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
        else:
            self.mlp_shared = nn.Sequential(
                    nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
            self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, similarity_map=None):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        if self.pad_type != 'zero':
            gamma = self.mlp_gamma(self.pad(actv))
            beta = self.mlp_beta(self.pad(actv))
        else:
            gamma = self.mlp_gamma(actv)
            beta = self.mlp_beta(actv)

        if similarity_map is not None:
            similarity_map = F.interpolate(similarity_map, size=gamma.size()[2:], mode='nearest')
            gamma = gamma * similarity_map
            beta = beta * similarity_map
        # apply scale and bias
        # print (normalized.shape)
        # print (gamma.shape)
        out = normalized * (1 + gamma) + beta

        return out




def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            if opt.eqlr_sn:
                layer = equal_lr(layer)
            else:
                layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            if opt.apex:
                norm_layer = apex.parallel.SyncBatchNorm(get_out_channel(layer), affine=True)
            else:
                norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer




class SPADE_TwoPath(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc_example, label_nc_imagine, PONO=False, use_apex=False):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.pad_type = 'nozero'

        if PONO:
            self.param_free_norm = PositionalNorm2d
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            if use_apex:
                self.param_free_norm = apex.parallel.SyncBatchNorm(norm_nc, affine=False)
            else:
                self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        if self.pad_type != 'zero':
            self.mlp_shared_example = nn.Sequential(
                nn.ReflectionPad2d(pw),
                nn.Conv2d(label_nc_example, nhidden, kernel_size=ks, padding=0),
                nn.ReLU()
            )
            self.pad = nn.ReflectionPad2d(pw)
            self.mlp_gamma_example = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
            self.mlp_beta_example = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)

            self.mlp_shared_imagine = nn.Sequential(
                nn.ReflectionPad2d(pw),
                nn.Conv2d(label_nc_imagine, nhidden, kernel_size=ks, padding=0),
                nn.ReLU()
            )
            self.mlp_gamma_imagine = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
            self.mlp_beta_imagine = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
        else:
            self.mlp_shared_example = nn.Sequential(
                    nn.Conv2d(label_nc_example, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
            self.mlp_gamma_example = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_beta_example = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

            self.mlp_shared_imagine = nn.Sequential(
                    nn.Conv2d(label_nc_imagine, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
            self.mlp_gamma_imagine = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_beta_imagine = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, warpmap, segmap, similarity_map):
        similarity_map = similarity_map.detach()
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        warpmap = F.interpolate(warpmap, size=x.size()[2:], mode='nearest')
        actv_example = self.mlp_shared_example(warpmap)
        actv_imagine = self.mlp_shared_imagine(segmap)
        if self.pad_type != 'zero':
            gamma_example = self.mlp_gamma_example(self.pad(actv_example))
            beta_example = self.mlp_beta_example(self.pad(actv_example))
            gamma_imagine = self.mlp_gamma_imagine(self.pad(actv_imagine))
            beta_imagine = self.mlp_beta_imagine(self.pad(actv_imagine))
        else:
            gamma_example = self.mlp_gamma_example(actv_example)
            beta_example = self.mlp_beta_example(actv_example)
            gamma_imagine = self.mlp_gamma_imagine(actv_imagine)
            beta_imagine = self.mlp_beta_imagine(actv_imagine)

        similarity_map = F.interpolate(similarity_map, size=x.size()[2:], mode='nearest')
        gamma = gamma_example * similarity_map + gamma_imagine * (1 - similarity_map)
        beta = beta_example * similarity_map + beta_imagine * (1 - similarity_map)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class EqualLR:
    def __init__(self, name):
        self.name = name
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * np.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn
    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module