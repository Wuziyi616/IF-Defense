"""Adopted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet.py"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, use_bn=True):
        super(STN3d, self).__init__()
        if use_bn:
            self.conv1 = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.BatchNorm1d(64)
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128)
            )
            self.conv3 = nn.Sequential(
                nn.Conv1d(128, 1024, 1),
                nn.BatchNorm1d(1024)
            )
            self.fc1 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512)
            )
            self.fc2 = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256)
            )
        else:
            self.conv1 = nn.Conv1d(3, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.from_numpy(np.array([
            1, 0, 0, 0, 1, 0, 0, 0, 1
        ]).astype(np.float32)).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))) \
            .view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, use_bn=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(use_bn=use_bn)

        if use_bn:
            self.conv1 = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.BatchNorm1d(64)
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128)
            )
            self.conv3 = nn.Sequential(
                nn.Conv1d(128, 1024, 1),
                nn.BatchNorm1d(1024)
            )
        else:
            self.conv1 = nn.Conv1d(3, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)

        self.relu = nn.ReLU(inplace=True)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.relu(self.conv1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False, use_bn=True):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            global_feat=True, feature_transform=feature_transform, use_bn=use_bn)
        self.use_bn = use_bn

        if use_bn:
            self.fc1 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512)
            )
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
        else:
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(
        torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss
