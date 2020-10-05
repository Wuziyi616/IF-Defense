from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pytorch_modules import SharedMLP
from .pu_utils import square_distance, index_points, farthest_point_sample, \
    QueryAndGroup, GroupAll


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None,
                npoint=None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        if npoint is not None:
            self.npoint = npoint
        new_features_list = []

        # xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = index_points(
                # xyz_flipped,
                xyz,
                farthest_point_sample(xyz, self.npoint)
            ) if self.npoint is not None else None  # [B, N, C]

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features)  # (B, C, npoint, nsample)
            # (B, mlp[-1], npoint, nsample)
            new_features = self.mlps[i](new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float],
                 nsamples: List[int], mlps: List[List[int]],
                 bn: bool = True, use_xyz: bool = True, use_res=False,
                 pool_method='max_pool', instance_norm=False):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            if use_res:
                raise NotImplementedError
            else:
                self.mlps.append(
                    SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm)
                )
        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None,
                 radius: float = None, nsample: int = None, bn: bool = True,
                 use_xyz: bool = True, use_res=False,
                 pool_method='max_pool', instance_norm=False):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super(PointnetSAModule, self).__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[
                nsample], bn=bn, use_xyz=use_xyz, use_res=use_res,
            pool_method=pool_method, instance_norm=instance_norm
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super(PointnetFPModule, self).__init__()
        self.mlp = SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor,
                unknow_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propagated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        known_feats = known_feats.permute(0, 2, 1)  # [B, m, C2]
        B, N, C = unknown.shape
        _, S, _ = known.shape

        if S == 1:
            interpolated_feats = known_feats.repeat(1, N, 1)
        else:
            dists = square_distance(unknown, known)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            weight = 1.0 / (dists + 1e-8)  # [B, N, 3]
            weight = weight / \
                torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_feats = torch.sum(index_points(
                known_feats, idx) * weight.view(B, N, 3, 1), dim=2)  # [B, N, C2]

        if unknow_feats is not None:
            unknow_feats = unknow_feats.permute(0, 2, 1)  # [B, n, C1]
            new_feats = torch.cat([
                unknow_feats, interpolated_feats
            ], dim=-1)  # [B, n, C1 + C2]
        else:
            new_feats = interpolated_feats

        new_feats = new_feats.permute(0, 2, 1)  # [B, C1 + C2, n]

        new_feats = new_feats.unsqueeze(-1)
        new_features = self.mlp(new_feats)

        return new_features.squeeze(-1)
