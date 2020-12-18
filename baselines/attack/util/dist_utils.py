import numpy as np

import torch
import torch.nn as nn

from util.set_distance import chamfer, hausdorff


class L2Dist(nn.Module):

    def __init__(self):
        """Compute global L2 distance between two point clouds.
        """
        super(L2Dist, self).__init__()

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Compute L2 distance between two point clouds.
        Apply different weights for batch input for CW attack.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3] or [B, 3, K]
            ori_pc (torch.FloatTensor): [B, K, 3] or [B, 3, k]
            weights (torch.FloatTensor, optional): [B], if None, just use avg
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        weights = weights.float().cuda()
        dist = torch.sqrt(torch.sum(
            (adv_pc - ori_pc) ** 2, dim=[1, 2]))  # [B]
        dist = dist * weights
        if batch_avg:
            return dist.mean()
        return dist


class ChamferDist(nn.Module):

    def __init__(self, method='adv2ori'):
        """Compute chamfer distance between two point clouds.

        Args:
            method (str, optional): type of chamfer. Defaults to 'adv2ori'.
        """
        super(ChamferDist, self).__init__()

        self.method = method

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Compute chamfer distance between two point clouds.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            ori_pc (torch.FloatTensor): [B, K, 3]
            weights (torch.FloatTensor, optional): [B], if None, just use avg
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        loss1, loss2 = chamfer(adv_pc, ori_pc)  # [B], adv2ori, ori2adv
        if self.method == 'adv2ori':
            loss = loss1
        elif self.method == 'ori2adv':
            loss = loss2
        else:
            loss = (loss1 + loss2) / 2.
        weights = weights.float().cuda()
        loss = loss * weights
        if batch_avg:
            return loss.mean()
        return loss


class HausdorffDist(nn.Module):

    def __init__(self, method='adv2ori'):
        """Compute hausdorff distance between two point clouds.

        Args:
            method (str, optional): type of hausdorff. Defaults to 'adv2ori'.
        """
        super(HausdorffDist, self).__init__()

        self.method = method

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Compute hausdorff distance between two point clouds.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            ori_pc (torch.FloatTensor): [B, K, 3]
            weights (torch.FloatTensor, optional): [B], if None, just use avg
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        loss1, loss2 = hausdorff(adv_pc, ori_pc)  # [B], adv2ori, ori2adv
        if self.method == 'adv2ori':
            loss = loss1
        elif self.method == 'ori2adv':
            loss = loss2
        else:
            loss = (loss1 + loss2) / 2.
        weights = weights.float().cuda()
        loss = loss * weights
        if batch_avg:
            return loss.mean()
        return loss


class KNNDist(nn.Module):

    def __init__(self, k=5, alpha=1.05):
        """Compute kNN distance punishment within a point cloud.

        Args:
            k (int, optional): kNN neighbor num. Defaults to 5.
            alpha (float, optional): threshold = mean + alpha * std. Defaults to 1.05.
        """
        super(KNNDist, self).__init__()

        self.k = k
        self.alpha = alpha

    def forward(self, pc, weights=None, batch_avg=True):
        """KNN distance loss described in AAAI'20 paper.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            weights (torch.FloatTensor, optional): [B]. Defaults to None.
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        # build kNN graph
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K], l2^2
        assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)
        # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # d_p, [B, K]
        with torch.no_grad():
            mean = torch.mean(value, dim=-1)  # [B]
            std = torch.std(value, dim=-1)  # [B]
            # [B], penalty threshold for batch
            threshold = mean + self.alpha * std
            weight_mask = (value > threshold[:, None]).\
                float().detach()  # [B, K]
        loss = torch.mean(value * weight_mask, dim=1)  # [B]
        # accumulate loss
        if weights is None:
            weights = torch.ones((B,))
        weights = weights.float().cuda()
        loss = loss * weights
        if batch_avg:
            return loss.mean()
        return loss


class ChamferkNNDist(nn.Module):

    def __init__(self, chamfer_method='adv2ori',
                 knn_k=5, knn_alpha=1.05,
                 chamfer_weight=5., knn_weight=3.):
        """Geometry-aware distance function of AAAI'20 paper.

        Args:
            chamfer_method (str, optional): chamfer. Defaults to 'adv2ori'.
            knn_k (int, optional): k in kNN. Defaults to 5.
            knn_alpha (float, optional): alpha in kNN. Defaults to 1.1.
            chamfer_weight (float, optional): weight factor. Defaults to 5..
            knn_weight (float, optional): weight factor. Defaults to 3..
        """
        super(ChamferkNNDist, self).__init__()

        self.chamfer_dist = ChamferDist(method=chamfer_method)
        self.knn_dist = KNNDist(k=knn_k, alpha=knn_alpha)
        self.w1 = chamfer_weight
        self.w2 = knn_weight

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Adversarial constraint function of AAAI'20 paper.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            ori_pc (torch.FloatTensor): [B, K, 3]
            weights (np.array): weight factors
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        chamfer_loss = self.chamfer_dist(
            adv_pc, ori_pc, weights=weights, batch_avg=batch_avg)
        knn_loss = self.knn_dist(
            adv_pc, weights=weights, batch_avg=batch_avg)
        loss = chamfer_loss * self.w1 + knn_loss * self.w2
        return loss


class FarthestDist(nn.Module):

    def __init__(self):
        """Used in adding cluster attack.
        """
        super(FarthestDist, self).__init__()

    def forward(self, adv_pc, weights=None, batch_avg=True):
        """Compute the farthest pairwise point dist in each added cluster.

        Args:
            adv_pc (torch.FloatTensor): [B, num_add, cl_num_p, 3]
            weights (np.array): weight factors
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        delta_matrix = adv_pc[:, :, None, :, :] - adv_pc[:, :, :, None, :] + 1e-7
        # [B, num_add, num_p, num_p, 3]
        norm_matrix = torch.norm(delta_matrix, p=2, dim=-1)  # [B, na, np, np]
        max_matrix = torch.max(norm_matrix, dim=2)[0]  # take the values of max
        far_dist = torch.max(max_matrix, dim=2)[0]  # [B, num_add]
        far_dist = torch.sum(far_dist, dim=1)  # [B]
        weights = weights.float().cuda()
        loss = far_dist * weights
        if batch_avg:
            return loss.mean()
        return loss


class FarChamferDist(nn.Module):

    def __init__(self, num_add, chamfer_method='adv2ori',
                 chamfer_weight=0.1):
        """Distance function used in generating adv clusters.
        Consisting of a Farthest dist and a chamfer dist.

        Args:
            num_add (int): number of added clusters.
            chamfer_method (str, optional): chamfer. Defaults to 'adv2ori'.
            chamfer_weight (float, optional): weight factor. Defaults to 0.1.
        """
        super(FarChamferDist, self).__init__()

        self.num_add = num_add
        self.far_dist = FarthestDist()
        self.chamfer_dist = ChamferDist(method=chamfer_method)
        self.cd_w = chamfer_weight

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Adversarial constraint function of CVPR'19 paper for adv clusters.

        Args:
            adv_pc (torch.FloatTensor): [B, num_add * cl_num_p, 3],
                                        the added clusters
            ori_pc (torch.FloatTensor): [B, K, 3]
            weights (np.array): weight factors
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        chamfer_loss = self.chamfer_dist(
            adv_pc, ori_pc, weights=weights, batch_avg=batch_avg)
        adv_clusters = adv_pc.view(B, self.num_add, -1, 3)
        far_loss = self.far_dist(
            adv_clusters, weights=weights, batch_avg=batch_avg)
        loss = far_loss + chamfer_loss * self.cd_w
        return loss


class L2ChamferDist(nn.Module):

    def __init__(self, num_add, chamfer_method='adv2ori',
                 chamfer_weight=0.2):
        """Distance function used in generating adv objects.
        Consisting of a L2 dist and a chamfer dist.

        Args:
            num_add (int): number of added objects.
            chamfer_method (str, optional): chamfer. Defaults to 'adv2ori'.
            chamfer_weight (float, optional): weight factor. Defaults to 0.2.
        """
        super(L2ChamferDist, self).__init__()

        self.num_add = num_add
        self.chamfer_dist = ChamferDist(method=chamfer_method)
        self.cd_w = chamfer_weight
        self.l2_dist = L2Dist()

    def forward(self, adv_pc, ori_pc, adv_obj, ori_obj,
                weights=None, batch_avg=True):
        """Adversarial constraint function of CVPR'19 paper for adv objects.

        Args:
            adv_pc (torch.FloatTensor): [B, num_add * obj_num_p, 3],
                                        the added objects after rot and shift
            ori_pc (torch.FloatTensor): [B, K, 3]
            adv_obj (torch.FloatTensor): [B, num_add, obj_num_p, 3],
                                        the added objects after pert
            ori_pc (torch.FloatTensor): [B, num_add, obj_num_p, 3],
                                        the clean added objects
            weights (np.array): weight factors
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        chamfer_loss = self.chamfer_dist(
            adv_pc, ori_pc, weights=weights, batch_avg=batch_avg)
        l2_loss = self.l2_dist(
            adv_obj.view(B, -1, 3), ori_obj.view(B, -1, 3),
            weights=weights, batch_avg=batch_avg)
        loss = l2_loss + self.cd_w * chamfer_loss
        return loss
