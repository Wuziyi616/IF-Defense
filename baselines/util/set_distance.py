"""Adopted from https://github.com/XuyangBai/FoldingNet/blob/master/loss.py"""
import torch
import torch.nn as nn


class _Distance(nn.Module):

    def __init__(self):
        super(_Distance, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        pass

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))  # [B, K, K]
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(
            1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


class ChamferDistance(_Distance):

    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, preds, gts):
        """
        preds: [B, N1, 3]
        gts: [B, N2, 3]
        """
        P = self.batch_pairwise_dist(gts, preds)  # [B, N2, N1]
        mins, _ = torch.min(P, 1)  # [B, N1], find preds' nearest points in gts
        loss1 = torch.mean(mins, dim=1)  # [B]
        mins, _ = torch.min(P, 2)  # [B, N2], find gts' nearest points in preds
        loss2 = torch.mean(mins, dim=1)  # [B]
        return loss1, loss2


class HausdorffDistance(_Distance):

    def __init__(self):
        super(HausdorffDistance, self).__init__()

    def forward(self, preds, gts):
        """
        preds: [B, N1, 3]
        gts: [B, N2, 3]
        """
        P = self.batch_pairwise_dist(gts, preds)  # [B, N2, N1]
        # max_{y \in pred} min_{x \in gt}
        mins, _ = torch.min(P, 1)  # [B, N1]
        loss1 = torch.max(mins, dim=1)[0]  # [B]
        # max_{y \in gt} min_{x \in pred}
        mins, _ = torch.min(P, 2)  # [B, N2]
        loss2 = torch.max(mins, dim=1)[0]  # [B]
        return loss1, loss2


chamfer = ChamferDistance()
hausdorff = HausdorffDistance()
