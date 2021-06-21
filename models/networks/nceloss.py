import torch
from torch import nn


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class BidirectionalNCE1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool
        self.l2_norm = Normalize(2)

    def forward(self, feat_q, feat_k):

        bs, dim, h, w = feat_q.shape
        feat_q = feat_q.view(bs, dim, -1).view(-1, h*w)
        feat_k = feat_k.view(bs, dim, -1).view(-1, h*w)


        feat_q = self.l2_norm(feat_q)  # 3, 410, 64, 64
        feat_k = self.l2_norm(feat_k)
        feat_k = feat_k.detach()

        # print('***', feat_q.shape)

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]

        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # batch_dim_for_bmm = self.args.batch_size
        batch_dim_for_bmm = bs

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        # l_neg_curbatch.masked_fill_cuda(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)


        out = torch.cat((l_pos, l_neg), dim=1) / 0.05

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        # print('***', l_pos.shape, l_neg.shape, loss)
        # 1/0
        return loss



class BidirectionalNCE2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool
        self.l2_norm = Normalize(2)

    def forward(self, feat_q, feat_k):

        feat_q = self.l2_norm(feat_q)
        feat_k = self.l2_norm(feat_k)
        feat_k = feat_k.detach()

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]

        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)


        batch_dim_for_bmm = self.args.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        # l_neg_curbatch.masked_fill_cuda(diagonal, -10.0)
        # masked_fill__cuda()
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.args.nce_T


        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class SRNCE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool  #torch.bool
        self.l2_norm = Normalize(2)

    def forward(self, feat_q, feat_k, feat_c):
        feat_q = self.l2_norm(feat_q)
        feat_k = self.l2_norm(feat_k)
        feat_c = self.l2_norm(feat_c)
        feat_k = feat_k.detach()
        feat_c = feat_c.detach()

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]

        l_pos1 = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos1 = l_pos1.view(batchSize, 1)
        sim_q_c = torch.bmm(feat_c.view(batchSize, 1, -1), feat_q.view(batchSize, -1, 1))
        sim_q_c = sim_q_c.view(batchSize, 1)
        pos_weight = torch.bmm(feat_c.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        pos_weight = pos_weight.view(batchSize, 1)

        l_pos1 = l_pos1 - (0.7 * (1 - pos_weight))
        l_pos2 = (1 - sim_q_c) - (0.7 * (1 - pos_weight))

        batch_dim_for_bmm = self.args.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        trans_feat_q = feat_q.transpose(2, 1)
        trans_feat_k = feat_k.transpose(2, 1)
        feat_c = feat_c.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_k, trans_feat_q)

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        # l_neg_curbatch.masked_fill_cuda(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out1 = torch.cat((l_pos1, l_pos2, l_neg), dim=1) / self.args.nce_T

        loss1 = self.cross_entropy_loss(out1, torch.zeros(out1.size(0), dtype=torch.long,
                                                          device=feat_q.device))
        out2 = l_pos2 / self.args.nce_T

        loss2 = self.cross_entropy_loss(out2, torch.zeros(out2.size(0), dtype=torch.long,
                                                          device=feat_q.device))

        loss = loss1 + loss2
        return loss
