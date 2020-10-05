import torch
import torch.nn as nn

from .pn_utils import knn_point, index_points


class RepulsionLoss(nn.Module):

    def __init__(self, nn_size=5, radius=0.07,
                 h=0.03, eps=1e-12, knn_batch=None):
        super(RepulsionLoss, self).__init__()
        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps
        self.knn_batch = knn_batch

    def get_repulsion_loss(self, pred):
        """Push points distribute uniformly.

        Args:
            pred (tensor): batch pc, [B, N (=rK), 3]
        """
        # find kNN
        if self.knn_batch is None:
            # adaptively find kNN batch_size
            flag = False
            self.knn_batch = pred.shape[0]
            while not flag:
                if self.knn_batch < 1:
                    print('CUDA OUT OF MEMORY in kNN of repulsion loss')
                    exit(-1)
                try:
                    with torch.no_grad():
                        idx = self.batch_knn(pred)
                    flag = True
                except RuntimeError:
                    torch.cuda.empty_cache()
                    self.knn_batch = self.knn_batch // 2
        else:
            with torch.no_grad():
                idx = self.batch_knn(pred)  # [B, N, k]
        grouped_points = index_points(pred, idx)  # [B, N, k, 3]
        # calculate l2 norm
        grouped_points = grouped_points - pred.unsqueeze(-2)  # [B, N, k, 3]
        dist2 = torch.sum(grouped_points ** 2, dim=-1)  # [B, N, k]
        dist2 = torch.max(dist2, torch.tensor(self.eps).cuda())
        dist = torch.sqrt(dist2)  # [B, N, k]
        weight = torch.exp(-((dist / self.h) ** 2))  # [B, N, k]
        # compute loss = r - kNN_dist
        # so that points can separate far from each other
        uniform_loss = (self.radius - dist) * weight  # [B, N, k]
        uniform_loss = torch.mean(uniform_loss, dim=[1, 2])  # [B]
        return uniform_loss

    def batch_knn(self, x):
        """Perform kNN on x by batch input.
            Since may encounter CUDA error if feed all x once.

        Args:
            x (tensor): point cloud
        """
        all_idx = []
        for batch_idx in range(0, x.shape[0], self.knn_batch):
            batch_x = x[batch_idx:batch_idx + self.knn_batch]
            all_idx.append(knn_point(self.nn_size, batch_x))
        idx = torch.cat(all_idx, dim=0)
        return idx

    def forward(self, pred):
        return self.get_repulsion_loss(pred)


repulsion_loss = RepulsionLoss()
