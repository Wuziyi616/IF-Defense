"""Adopted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master/models"""
import numpy as np

import torch


###########################################
# pytorch based functions
###########################################
# processed in batch data


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


def random_sample_points(points, num):
    """points: [B, K, 3]"""
    device = points.device
    batch = points.size(0)
    idx = torch.randint(0, points.size(1), (batch, num)
                        ).to(device)  # [batch, num]
    sampled_points = index_points(points, idx)
    return sampled_points


def normalize_batch_points_torch(points):
    """points: [batch, K, 3]"""
    centroid = torch.mean(points, dim=1)  # [batch, 3]
    points -= centroid[:, None, :]  # center, [batch, K, 3]
    dist = torch.sum(points ** 2, dim=2) ** 0.5  # [batch, K]
    max_dist = torch.max(dist, dim=1)[0]  # [batch]
    points /= max_dist[:, None, None]
    assert torch.sum(torch.isnan(points)) == 0
    return points


def normalize_points_torch(points):
    """points: [K, 3]"""
    points = points - torch.mean(points, dim=0)[None, :]  # center
    dist = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)), dim=0)[0]
    points = points / dist  # scale
    assert torch.sum(torch.isnan(points)) == 0
    return points


###########################################
# numpy based functions
###########################################
# processed in single data


def normalize_points_np(points):
    """points: [K, 3]"""
    points = points - np.mean(points, axis=0)[None, :]  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale
    assert np.sum(np.isnan(points)) == 0
    return points


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


def random_sample_points_np(points, num):
    """points: [K, 3]"""
    idx = np.random.choice(len(points), num, replace=True)
    return points[idx]
