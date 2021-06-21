import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from geomloss import SamplesLoss
from PIL import Image

class WTA_scale(torch.autograd.Function):
    """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """

    @staticmethod
    def forward(ctx, input, scale=1e-4):
        """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
        activation_max, index_max = torch.max(input, -1, keepdim=True)
        input_scale = input * scale  # default: 1e-4
        # input_scale = input * scale  # default: 1e-4
        output_max_scale = torch.where(input == activation_max, input, input_scale)

        mask = (input == activation_max).type(torch.float)
        ctx.save_for_backward(input, mask)
        return output_max_scale

    @staticmethod
    def backward(ctx, grad_output):
        """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
        # import pdb
        # pdb.set_trace()
        input, mask = ctx.saved_tensors
        mask_ones = torch.ones_like(mask)
        mask_small_ones = torch.ones_like(mask) * 1e-4
        # mask_small_ones = torch.ones_like(mask) * 1e-4

        grad_scale = torch.where(mask == 1, mask_ones, mask_small_ones)
        grad_input = grad_output.clone() * grad_scale
        return grad_input, None

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

class NoVGGCorrespondence():
    # input is Al, Bl, channel = 1, range~[0,255]
    def __init__(self):
        super().__init__()

        self.p = 2
        self.blur = 0.075
        self.sampleloss = SamplesLoss("sinkhorn", p=self.p, blur=self.blur, debias=False, potentials=True)

    def ot(self, temperature=0.01,
                detach_flag=False,
                WTA_scale_weight=1):
        # coor_out = {}
        # batch_size = ref_img.shape[0]
        # image_height = ref_img.shape[2]
        # image_width = ref_img.shape[3]
        # feature_height = int(image_height / self.opt.down)
        # feature_width = int(image_width / self.opt.down)

        x = torch.tensor([[0.5, 0.5, 0.5, 0.5],
                         [0.4, 0.2, 0.3, 0.1]]).permute(1, 0).view(1, 4, 2).cuda()
        y = torch.tensor([[0.1, 0.1, 0.1, 0.1],
                         [0.1, 0.2, 0.3, 0.4]]).permute(1, 0).view(1, 4, 2).cuda()
        # x_w = torch.tensor([[0.25, 0.25, 0.25, 0.25],
        #                     [0.25, 0.25, 0.25, 0.25]]).view(1, 4, 2).cuda()
        # y_w = torch.tensor([[0.25, 0.25, 0.25, 0.25],
        #                     [0.25, 0.25, 0.25, 0.25]]).view(1, 4, 2).cuda()

        F_, G_, loss_ot = self.sampleloss(x, y)

        print (x)
        print (y)


        _, N, D = x.shape
        p = self.p
        blur = self.blur
        x_i = x.view(-1, N, 1, D)
        y_j = x.view(-1, 1, N, D)
        F_i, G_j = F_.view(-1, N, 1), G_.view(-1, 1, N)
        C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)  # (N,M) cost matrix
        eps = blur ** p
        f = ((F_i + G_j - C_ij) / eps).exp()
        f = f.permute(0, 2, 1)

        print (loss_ot)
        print (f)
        print (1/0)





        _, N, D = theta_permute.shape
        phi_permute = phi.permute(0, 2, 1).contiguous()
        theta_permute = theta_permute.contiguous()

        if self.opt.correspondence == 'ot':
            # weight
            if self.opt.ot_weight:
                mat = torch.matmul(theta_permute, phi)
                mat1 = self.relu(mat) + 1e-9
                theta_w = mat1.sum(-1)
                # theta_w = theta_w / theta_w.sum(-1).view(-1, 1)

                # theta_w_ = theta_w.view(-1, 1, 32, 32).repeat(1, 3, 1, 1)
                # coor_out['weight'] = F.interpolate(theta_w_, size=(256, 256))
                mat2 = mat.permute(0, 2, 1)
                mat2 = self.relu(mat2) + 1e-9
                phi_w = mat2.sum(-1)
                # phi_w = phi_w / phi_w.sum(-1).view(-1, 1)

                # print(theta_w[0, :5])
                # print(phi_w[0, :5])

                weight_input = torch.cat((theta.view(batch_size, -1, 32, 32), phi.view(batch_size, -1, 32, 32)), dim=1)
                weight = self.weight_layer(weight_input) + 1e-8

                # print (weight.min(), weight.max())
                theta_w, phi_w = weight[:, :1, :, :].view(-1, 1024), weight[:, 1:, :, :].view(-1, 1024)

                theta_w = theta_w / theta_w.sum(-1).view(-1, 1)
                phi_w = phi_w / phi_w.sum(-1).view(-1, 1)

                # print (theta_w.min(), theta_w.max())
                # print (phi_w.min(), phi_w.max())
                # print (theta_w[0, :5])
                # print (phi_w[0, :5])
                # print (1/0)


                theta_w_ = theta_w.view(-1, 1, 32, 32).repeat(1, 3, 1, 1)
                phi_w_ = phi_w.view(-1, 1, 32, 32).repeat(1, 3, 1, 1)
                coor_out['weight1'] = F.interpolate(theta_w_, size=(128, 128))
                coor_out['weight2'] = F.interpolate(phi_w_, size=(128, 128))


                F_, G_, loss_ot = self.sampleloss(theta_w, theta_permute, phi_w, phi_permute)
            else:
                F_, G_, loss_ot = self.sampleloss(theta_permute, phi_permute)

            p = self.p
            blur = self.blur
            x_i = theta_permute.view(-1, N, 1, D)
            y_j = phi_permute.view(-1, 1, N, D)
            F_i, G_j = F_.view(-1, N, 1), G_.view(-1, 1, N)
            C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)  # (N,M) cost matrix
            eps = blur ** p
            f = ((F_i + G_j - C_ij) / eps).exp()

        else:
            f = torch.matmul(theta_permute, phi)
            mat = f

        if WTA_scale_weight == 1:
            f_WTA = f
        else:
            f_WTA = WTA_scale.apply(f, WTA_scale_weight)
        f_WTA = f_WTA / temperature

        if self.opt.correspondence == 'ot':
            f_div_C = f_WTA / f_WTA.sum(-1).view(-1, 1024, 1)
            # mat = WTA_scale.apply(mat, WTA_scale_weight)
            # mat = mat / temperature
            # mat = F.softmax(mat.squeeze(), dim=-1)
        else:
            f_div_C = F.softmax(f_WTA.squeeze(), dim=-1)


model = NoVGGCorrespondence()
model.ot()
