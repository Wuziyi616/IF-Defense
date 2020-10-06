import torch
import torch.nn as nn


class ClipPointsL2(nn.Module):

    def __init__(self, budget):
        """Clip point cloud with a given global l2 budget.

        Args:
            budget (float): perturbation budget
        """
        super(ClipPointsL2, self).__init__()

        self.budget = budget

    def forward(self, pc, ori_pc):
        """Clipping every point in a point cloud.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
        """
        with torch.no_grad():
            diff = pc - ori_pc  # [B, 3, K]
            norm = torch.sum(diff ** 2, dim=[1, 2]) ** 0.5  # [B]
            scale_factor = self.budget / (norm + 1e-9)  # [B]
            scale_factor = torch.clamp(scale_factor, max=1.)  # [B]
            diff = diff * scale_factor[:, None, None]
            pc = ori_pc + diff
        return pc


class ClipPointsLinf(nn.Module):

    def __init__(self, budget):
        """Clip point cloud with a given l_inf budget.

        Args:
            budget (float): perturbation budget
        """
        super(ClipPointsLinf, self).__init__()

        self.budget = budget

    def forward(self, pc, ori_pc):
        """Clipping every point in a point cloud.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
        """
        with torch.no_grad():
            diff = pc - ori_pc  # [B, 3, K]
            norm = torch.sum(diff ** 2, dim=1) ** 0.5  # [B, K]
            scale_factor = self.budget / (norm + 1e-9)  # [B, K]
            scale_factor = torch.clamp(scale_factor, max=1.)  # [B, K]
            diff = diff * scale_factor[:, None, :]
            pc = ori_pc + diff
        return pc


class ProjectInnerPoints(nn.Module):

    def __init__(self):
        """Eliminate points shifted inside an object.
        Introduced by AAAI'20 paper.
        """
        super(ProjectInnerPoints, self).__init__()

    def forward(self, pc, ori_pc, normal=None):
        """Clipping "inside" points to the surface of the object.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
            normal (torch.FloatTensor, optional): normals. Defaults to None.
        """
        with torch.no_grad():
            # in case we don't have normals
            if normal is None:
                return pc
            diff = pc - ori_pc
            inner_diff_normal = torch.sum(
                diff * normal, dim=1)  # [B, K]
            inner_mask = (inner_diff_normal < 0.)  # [B, K]

            # clip to surface!
            # 1) vng = Normal x Perturb
            vng = torch.cross(normal, diff, dim=1)  # [B, 3, K]
            vng_norm = torch.sum(vng ** 2, dim=1) ** 0.5  # [B, K]

            # 2) vref = vng x Normal
            vref = torch.cross(vng, normal)  # [B, 3, K]
            vref_norm = torch.sum(vref ** 2, dim=1) ** 0.5  # [B, K]

            # 3) Project Perturb onto vref
            diff_proj = diff * vref / \
                (vref_norm[:, None, :] + 1e-9)  # [B, 3, K]

            # some diff is completely opposite to normal
            # just set them to (0, 0, 0)
            opposite_mask = inner_mask & (vng_norm < 1e-6)
            opposite_mask = opposite_mask.\
                unsqueeze(1).expand_as(diff_proj)
            diff_proj[opposite_mask] = 0.

            # set inner points with projected perturbation
            inner_mask = inner_mask.\
                unsqueeze(1).expand_as(diff)
            diff[inner_mask] = diff_proj[inner_mask]
            pc = ori_pc + diff
        return pc


class ProjectInnerClipLinf(nn.Module):

    def __init__(self, budget):
        """Project inner points to the surface and
        clip the l_inf norm of perturbation.

        Args:
            budget (float): l_inf norm budget
        """
        super(ProjectInnerClipLinf, self).__init__()

        self.project_inner = ProjectInnerPoints()
        self.clip_linf = ClipPointsLinf(budget=budget)

    def forward(self, pc, ori_pc, normal=None):
        """Project to the surface and then clip.

        Args:
            pc (torch.FloatTensor): batch pc, [B, 3, K]
            ori_pc (torch.FloatTensor): original point cloud
            normal (torch.FloatTensor, optional): normals. Defaults to None.
        """
        with torch.no_grad():
            # project
            pc = self.project_inner(pc, ori_pc, normal)
            # clip
            pc = self.clip_linf(pc, ori_pc)
        return pc
