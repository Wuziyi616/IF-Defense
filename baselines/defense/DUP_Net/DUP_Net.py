import os
import numpy as np

import torch
import torch.nn as nn

from .pu_net import PUNet
from ..drop_points import SORDefense


class DUPNet(nn.Module):

    def __init__(self, sor_k=2, sor_alpha=1.1,
                 npoint=1024, up_ratio=4):
        super(DUPNet, self).__init__()

        self.npoint = npoint
        self.sor = SORDefense(k=sor_k, alpha=sor_alpha)
        self.pu_net = PUNet(npoint=self.npoint, up_ratio=up_ratio,
                            use_normal=False, use_bn=False, use_res=False)

    def process_data(self, pc, npoint=None):
        """Process point cloud data to be suitable for
            PU-Net input.
        We do two things:
            sample npoint or duplicate to npoint.

        Args:
            pc (torch.FloatTensor): list input, [(N_i, 3)] from SOR.
                Need to pad or trim to [B, self.npoint, 3].
        """
        if npoint is None:
            npoint = self.npoint
        B = len(pc)
        proc_pc = torch.zeros((B, npoint, 3)).float().cuda()
        for pc_idx in range(B):
            one_pc = pc[pc_idx]
            # [N_i, 3]
            N = len(one_pc)
            if N > npoint:
                # random sample some of them
                idx = np.random.choice(N, npoint, replace=False)
                idx = torch.from_numpy(idx).long().cuda()
                one_pc = one_pc[idx]
            elif N < npoint:
                # just duplicate to the number
                duplicated_pc = one_pc
                num = npoint // N - 1
                for i in range(num):
                    duplicated_pc = torch.cat([
                        duplicated_pc, one_pc
                    ], dim=0)
                num = npoint - len(duplicated_pc)
                # random sample the remaining
                idx = np.random.choice(N, num, replace=False)
                idx = torch.from_numpy(idx).long().cuda()
                one_pc = torch.cat([
                    duplicated_pc, one_pc[idx]
                ], dim=0)
            proc_pc[pc_idx] = one_pc
        return proc_pc

    def forward(self, x):
        with torch.no_grad():
            x = self.sor(x)  # a list of pc
            x = self.process_data(x)  # to batch input
            x = self.pu_net(x)  # [B, N * r, 3]
        return x
