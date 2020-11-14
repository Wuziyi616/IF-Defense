import numpy as np

import torch


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, num_point):
    """
    Using FPS to sample N points from a given point cloud.
    Input:
        xyz: point cloud data, [B, N, C]
        num_point: number of samples
    Return:
        centroids: sampled point cloud index, [B, num_points]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, num_point, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(num_point):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def fps_points(xyz, num_point):
    """
    Using FPS to sample N points from a given point cloud.
    Input:
        xyz: point cloud data, [B, N, C]
        num_point: number of samples
    Return:
        points: [B, S, C]
    """
    centroids = farthest_point_sample(xyz, num_point)
    return index_points(xyz, centroids)


def knn_point(k, points):
    """Returns kNN idx for given pointcloud.

    Args:
        k (int): kNN neighbor num
        points (tensor): batch pc, [B, K, 3]
    """
    # no grad
    pc = points.clone().detach()
    # build kNN graph
    B, K = pc.shape[:2]
    pc = pc.transpose(2, 1)  # [B, 3, K]
    inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
    xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
    dist = xx + inner + xx.transpose(2, 1)  # [B, K, K], l2^2
    assert dist.min().item() >= -1e-4
    # the min is self so we take top (k + 1)
    _, top_idx = (-dist).topk(k=k + 1, dim=-1)  # [B, K, k + 1]
    top_idx = top_idx[:, :, 1:]  # [B, K, k]
    return top_idx


def index_points_np(points, idx):
    """
    Input:
        points: input points data, [N, C]
        idx: sample index data, [S]
    Return:
        new_points:, indexed points data, [S, C]
    """
    return points[idx]


def farthest_point_sample_np(xyz, num_point):
    """
    Using FPS to sample N points from a given point cloud.
    Input:
        xyz: point cloud data, [N, C]
        num_point: number of samples
    Return:
        centroids: sampled point cloud index, [num_points]
    """
    N, C = xyz.shape
    centroids = np.zeros((num_point,), dtype=np.int)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(num_point):
        centroids[i] = farthest
        centroid = xyz[farthest]  # [C]
        dist = np.sum((xyz - centroid[None, :]) ** 2, axis=1)  # [N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return centroids


def fps_points_np(xyz, num_point):
    """
    Using FPS to sample N points from a given point cloud.
    Input:
        xyz: point cloud data, [N, C]
        num_point: number of samples
    Return:
        points: [S, C]
    """
    centroids = farthest_point_sample_np(xyz, num_point)
    return index_points_np(xyz, centroids)
