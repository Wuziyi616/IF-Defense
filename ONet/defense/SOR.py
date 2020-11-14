import torch
import torch.nn as nn


class SORDefense(nn.Module):
    """Statistical outlier removal as defense.
    """

    def __init__(self, k=2, alpha=1.1, sor_batch=None):
        """SOR defense.

        Args:
            k (int, optional): kNN. Defaults to 2.
            alpha (float, optional): \miu + \alpha * std. Defaults to 1.1.
        """
        super(SORDefense, self).__init__()

        self.k = k
        self.alpha = alpha
        self.sor_batch = sor_batch

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
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)
        # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # [B, K]
        mean = torch.mean(value, dim=-1)  # [B]
        std = torch.std(value, dim=-1)  # [B]
        # [B]
        threshold = mean + self.alpha * std
        bool_mask = (value <= threshold[:, None])  # [B, K]
        sel_pc = [x[i][bool_mask[i]] for i in range(B)]
        return sel_pc

    def forward(self, x):
        if self.sor_batch is None:
            # adaptively find kNN batch_size
            flag = False
            self.sor_batch = x.shape[0]
            while not flag:
                if self.sor_batch < 1:
                    print('CUDA OUT OF MEMORY in kNN of repulsion loss')
                    exit(-1)
                try:
                    with torch.no_grad():
                        sor_x = self.batch_sor(x)
                    flag = True
                except RuntimeError:
                    torch.cuda.empty_cache()
                    self.sor_batch = self.sor_batch // 2
        else:
            with torch.no_grad():
                sor_x = self.batch_sor(x)  # [B, N, k]
        return sor_x

    def batch_sor(self, x):
        out = []
        for i in range(0, x.shape[0], self.sor_batch):
            batch_x = x[i:i + self.sor_batch]
            out += self.outlier_removal(batch_x)
        return out
