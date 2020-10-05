"""SOR defense proposed by ICCV'19 paper DUP-Net"""
import torch
import torch.nn as nn


class SORDefense(nn.Module):
    """Statistical outlier removal as defense.
    """

    def __init__(self, k=2, alpha=1.1):
        """SOR defense.

        Args:
            k (int, optional): kNN. Defaults to 2.
            alpha (float, optional): \miu + \alpha * std. Defaults to 1.1.
        """
        super(SORDefense, self).__init__()

        self.k = k
        self.alpha = alpha

    def outlier_removal(self, x):
        """Removes large kNN distance points.

        Args:
            x (torch.FloatTensor): batch input pc, [B, K, 3]

        Returns:
            torch.FloatTensor: pc after outlier removal, [B, N, 3]
        """
        pc = x.clone().detach().double()
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]
        assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)  # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # [B, K]
        mean = torch.mean(value, dim=-1)  # [B]
        std = torch.std(value, dim=-1)  # [B]
        threshold = mean + self.alpha * std  # [B]
        bool_mask = (value <= threshold[:, None])  # [B, K]
        sel_pc = [x[i][bool_mask[i]] for i in range(B)]
        return sel_pc

    def forward(self, x):
        with torch.no_grad():
            x = self.outlier_removal(x)
        return x
