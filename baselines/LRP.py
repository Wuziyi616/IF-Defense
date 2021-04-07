import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import sys
sys.path.append('../')
sys.path.append('./')

import LRP_utils as utils
from config import BEST_WEIGHTS
from config import MAX_DROP_BATCH as BATCH_SIZE
from dataset import ModelNet40Attack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from util.utils import str2bool, set_seed
from attack import ClipPointsL2


def LRP_scores(model, x, label, target):
    # stn layers
    iden = torch.from_numpy(np.array([
        1, 0, 0, 0, 1, 0, 0, 0, 1
    ]).astype(np.float32)).view(1, 9).repeat(x.shape[0], 1).cuda()
    stn = model.module.feat.stn
    stn_layers = [stn.conv1, nn.ReLU(), stn.conv2, nn.ReLU(), stn.conv3, nn.ReLU(), nn.MaxPool1d(1024),
                  stn.fc1, nn.ReLU(), stn.fc2, nn.ReLU(), stn.fc3, utils.Add_Iden_to_conv(iden), 'MatReshape']
    stn_layers = utils.toconv(stn_layers)
    stn_len = len(stn_layers)
    stn_x = [x] + [None] * stn_len
    for i in range(stn_len):
        if stn_layers[i] == 'MatReshape':
            stn_x[i + 1] = stn_x[i].view(-1, 3, 3)
        else:
            stn_x[i + 1] = stn_layers[i].forward(stn_x[i])

    # feat layers
    feat = model.module.feat
    feat_layers = ['transpose', feat.conv1, nn.ReLU(), feat.conv2, nn.ReLU(), feat.conv3, nn.MaxPool1d(1024)]
    feat_len = len(feat_layers)
    feat_x = [x] + [None] * feat_len
    feat_x[0] = torch.bmm(x.transpose(2, 1), stn_x[-1])
    for i in range(feat_len):
        if feat_layers[i] == 'transpose':
            feat_x[i + 1] = feat_x[i].transpose(2, 1)
        elif feat_layers[i] == 'SqueezeLast':
            feat_x[i + 1] = feat_x[i].view(-1, 1024)
        else:
            feat_x[i + 1] = feat_layers[i].forward(feat_x[i])

    # classifier layers
    cls_layers = [model.module.fc1, nn.ReLU(), nn.Sequential(model.module.fc2, model.module.bn2), nn.ReLU(),
                  model.module.fc3]
    cls_layers = utils.toconv(cls_layers)
    cls_len = len(cls_layers)
    cls_x = [feat_x[-1]] + [None] * cls_len
    for i in range(cls_len):
        cls_x[i + 1] = cls_layers[i].forward(cls_x[i])

    cls_x[-1] = F.softmax(cls_x[-1], dim=1)
    cls_fR = -cls_x[-1][:, target:target+1, :]*cls_x[-1]
    # if target == label:
    #     cls_x[-1][:, target, :].data = (cls_x[-1][:, label, :]*(1-cls_x[-1][:, label, :])).data
    # else:
    #     cls_x[-1][:, target, :].data = (-cls_x[-1][:, label, :] * cls_x[-1][:, target, :]).data
    # ***************************************************LRP*****************************************************
    T = torch.arange(0, 40, dtype=torch.long).view(1, -1, 1).cuda()
    T = target.view(-1, 1, 1) == T

    # cls layers
    cls_R = [None] * cls_len + [cls_x[-1] * T]
    # cls_R[-1] = cls_R[-1] / torch.max(cls_R[-1], dim=1)[0]
    # print(cls_R[-1].max())
    # cls_R[-1].retain_grad()
    for i in range(0, cls_len)[::-1]:
        if isinstance(cls_layers[i], nn.ReLU):
            cls_R[i] = cls_R[i + 1]
        else:
            rho = lambda p: p.clamp(min=0)
            incr = lambda y: y + 1e-9
            z = incr(utils.newlayer(cls_layers[i], rho).forward(cls_x[i]))  # step 1
            W = None
            if isinstance(cls_layers[i], nn.Conv1d):
                W = rho(cls_layers[i].weight)
            else:
                W = rho(cls_layers[i][0].weight) * ((cls_layers[i][1].weight /
                            torch.sqrt(cls_layers[i][1].running_var + cls_layers[i][1].eps)).view(-1, 1, 1))

            W = (W.data).unsqueeze(0)
            s = cls_R[i + 1] / z
            c = torch.sum(W * s.unsqueeze(-2), dim=1)
            cls_R[i] = cls_x[i] * c

    # feat layers
    feat_R = [None] * feat_len + [cls_R[0]]
    for i in range(0, feat_len)[::-1]:
        # feat_x[i].retain_grad()
        # feat_x[i] = (feat_x[i].data).requires_grad_(True)
        if feat_layers[i] == 'transpose':
            feat_R[i] = feat_R[i + 1].transpose(2, 1)
        elif isinstance(feat_layers[i], nn.MaxPool1d):
            max_id = torch.max(feat_x[i], dim=-1)[1]
            feat_R[i] = torch.zeros_like(feat_x[i])
            batch_ids = torch.arange(0, x.shape[0]).view(-1, 1)
            fids = torch.arange(0, 1024).view(1, -1)
            feat_R[i][batch_ids, fids, max_id] = feat_R[i + 1][:, :, 0]
        elif isinstance(feat_layers[i], nn.ReLU):
            feat_R[i] = feat_R[i + 1]
        else:
            rho = lambda p: p.clamp(min=0)
            incr = lambda y: y + 1e-9       # + 0.25 * ((y ** 2).mean() ** .5).data
            z = incr(utils.newlayer(feat_layers[i], rho).forward(feat_x[i]))  # step 1
            if isinstance(feat_layers[i], nn.Conv1d):
                W = rho(feat_layers[i].weight)
            else:
                W = rho(feat_layers[i][0].weight) * ((feat_layers[i][1].weight /
                            torch.sqrt(feat_layers[i][1].running_var + feat_layers[i][1].eps)).view(-1, 1, 1))

            W = (W.data).unsqueeze(0)
            s = feat_R[i + 1] / z
            c = torch.sum(W * s.unsqueeze(-2), dim=1)
            feat_R[i] = feat_x[i] * c

    # for bmm specific
    feat0_R = feat_R[0].transpose(2, 1)
    bmm_xR = torch.zeros_like(feat0_R)
    bmm_convs = utils.bmm_to_conv(stn_x[-1])
    for i in range(x.shape[0]):
        bmm_x = x[i].unsqueeze(0)
        lb = x[i].unsqueeze(0) * 0 - 1
        hb = x[i].unsqueeze(0) * 0 + 1
        z = bmm_convs[i].forward(bmm_x) + 1e-9  # step 1 (a)
        z -= utils.newlayer(bmm_convs[i], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
        z -= utils.newlayer(bmm_convs[i], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)

        W = bmm_convs[i].weight
        rho_plus = lambda p: p.clamp(min=0)
        W_plus = rho_plus(bmm_convs[i].weight)
        rho_minus = lambda p: p.clamp(max=0)
        W_minus = rho_minus(bmm_convs[i].weight)

        W = (W.data).unsqueeze(0)
        W_plus = (W_plus.data).unsqueeze(0)
        W_minus = (W_minus.data).unsqueeze(0)
        s = feat0_R[i].unsqueeze(0) / z
        c = torch.sum(W * s.unsqueeze(-2), dim=1)
        cp = torch.sum(W_plus * s.unsqueeze(-2), dim=1)
        cm = torch.sum(W_minus * s.unsqueeze(-2), dim=1)
        bmm_xR[i] = (c * bmm_x - cp * lb - cm * hb)[0]

    stn_fR = torch.zeros_like(stn_x[-1])
    bmm_convs = utils.bmm_to_conv(x)
    for i in range(x.shape[0]):
        bmm_x = stn_x[-1][i].unsqueeze(0)
        rho = lambda p: p.clamp(min=0)
        incr = lambda y: y + 1e-9
        z = incr(utils.newlayer(bmm_convs[i], rho).forward(bmm_x))  # step 1

        W = rho(bmm_convs[i].weight)
        W = (W.data).unsqueeze(0)

        s = feat_R[0][i].unsqueeze(0) / z
        c = torch.sum(W * s.unsqueeze(-2), dim=1)
        stn_fR[i] = (c * bmm_x)[0]

    # stn layers
    stn_R = [None] * stn_len + [stn_fR]
    for i in range(1, stn_len)[::-1]:
        if stn_layers[i] == 'MatReshape':
            stn_R[i] = stn_R[i + 1].view(-1, 9, 1)
        elif isinstance(stn_layers[i], nn.MaxPool1d):
            max_id = torch.max(stn_x[i], dim=-1)[1]
            stn_R[i] = torch.zeros_like(stn_x[i])
            batch_ids = torch.arange(0, x.shape[0]).view(-1, 1)
            fids = torch.arange(0, 1024).view(1, -1)
            stn_R[i][batch_ids, fids, max_id] = stn_R[i + 1][:, :, 0]
        elif isinstance(stn_layers[i], nn.ReLU) or i == 12:
            stn_R[i] = stn_R[i + 1]
        else:
            rho = lambda p: p.clamp(min=0)
            incr = lambda y: y + 1e-8
            z = incr(utils.newlayer(stn_layers[i], rho).forward(stn_x[i]))  # step 1

            if isinstance(stn_layers[i], nn.Conv1d):
                W = rho(stn_layers[i].weight)
            else:
                W = rho(stn_layers[i][0].weight) * ((stn_layers[i][1].weight /
                            torch.sqrt(stn_layers[i][1].running_var + stn_layers[i][1].eps)).view(-1, 1, 1))

            W = (W.data).unsqueeze(0)
            s = stn_R[i + 1] / z
            c = torch.sum(W * s.unsqueeze(-2), dim=1)
            stn_R[i] = stn_x[i] * c

    # first layer from stn
    lb = stn_x[0] * 0 - 1
    hb = stn_x[0] * 0 + 1

    z = stn_layers[0].forward(stn_x[0]) + 1e-9  # step 1 (a)
    z -= utils.newlayer(stn_layers[0], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
    z -= utils.newlayer(stn_layers[0], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)
    z = z + 1e-8

    if isinstance(stn_layers[0], nn.Conv1d):
        W = stn_layers[0].weight
        rho_plus = lambda p: p.clamp(min=0)
        W_plus = rho_plus(stn_layers[0].weight)
        rho_minus = lambda p: p.clamp(max=0)
        W_minus = rho_minus(stn_layers[0].weight)
    else:
        W = stn_layers[0][0].weight * ((stn_layers[0][1].weight /
                            torch.sqrt(stn_layers[0][1].running_var + stn_layers[0][1].eps)).view(-1, 1, 1))
        rho_plus = lambda p: p.clamp(min=0)
        W_plus = rho_plus(stn_layers[0][0].weight) * ((stn_layers[0][1].weight /
                            torch.sqrt(stn_layers[0][1].running_var + stn_layers[0][1].eps)).view(-1, 1, 1))
        rho_minus = lambda p: p.clamp(max=0)
        W_minus = rho_minus(stn_layers[0][0].weight) * ((stn_layers[0][1].weight /
                            torch.sqrt(stn_layers[0][1].running_var + stn_layers[0][1].eps)).view(-1, 1, 1))

    W = (W.data).unsqueeze(0)
    W_plus = (W_plus.data).unsqueeze(0)
    W_minus = (W_minus.data).unsqueeze(0)
    s = stn_R[1] / z
    c = torch.sum(W * s.unsqueeze(-2), dim=1)
    cp = torch.sum(W_plus * s.unsqueeze(-2), dim=1)
    cm = torch.sum(W_minus * s.unsqueeze(-2), dim=1)
    stn_R[0] = c * stn_x[0] - cp * lb - cm * hb

    R0 = bmm_xR + stn_R[0]
    # R0 = stn_R[0]

    return R0


def AOA_Attack(model, x_adv, label, target):
    # pred, _, _ = model(x)
    # lsec = torch.topk(pred, k=2, dim=-1)[1].squeeze()
    clip_func = ClipPointsL2(budget=0.5)
    x_ori = x_adv.clone().detach()
    yita = 0.08
    eps = 0.04
    N = x_adv.shape[0] * x_adv.shape[1] * x_adv.shape[2]
    alpha = 0.00016
    iter_k = 0
    lmd = 10
    torch.autograd.set_detect_anomaly(True)
    while iter_k < 500:
        R_ori = LRP_scores(model, x_adv, label, label)
        R_sec = LRP_scores(model, x_adv, label, target)
        # log_loss = R_ori.sum() - R_sec.sum()
        log_loss = torch.log(R_ori.abs().sum()) - torch.log(R_sec.abs().sum())
        pred, _, _ = model(x_adv)
        pred = torch.argmax(pred, dim=-1)
        if pred == target:
            break
        # ce_loss = nn.CrossEntropyLoss()(pred, target)
        loss = log_loss      # + lmd * ce_loss
        loss.backward()
        g = x_adv.grad
        perturbation = alpha * g / (g.abs().sum() / N)
        # with torch.no_grad():
        x_adv = x_adv - perturbation
        x_adv = x_ori + (x_adv-x_ori).clamp(min=-eps, max=eps)
        x_adv = clip_func(x_adv, x_ori)
        x_adv = (x_adv.data).requires_grad_(True)
        iter_k += 1
        # if iter_k%40==0:
        #     alpha *= 0.1
        print(loss)

    print("iter time is%d" %iter_k)
    return x_adv


def RMSE(xori, xadv):
    x = xadv-xori
    # N = x.shape[0] * x.shape[1] * x.shape[2]
    return torch.sqrt(torch.sum(x**2))

def main():
    global BATCH_SIZE, BEST_WEIGHTS
    BATCH_SIZE = BATCH_SIZE[1024]
    BEST_WEIGHTS = BEST_WEIGHTS['mn40'][1024]
    cudnn.benchmark = True

    # build model
    model = PointNetCls(k=40, feature_transform=False)
    model = nn.DataParallel(model).cuda()
    model.eval()

    # load model weight
    print('Loading weight {}'.format(BEST_WEIGHTS['pointnet']))
    state_dict = torch.load(BEST_WEIGHTS['pointnet'])
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.module.load_state_dict(state_dict)

    # load dataset
    test_set = ModelNet40Attack('data/attack_data.npz', num_points=1024,
                                normalize=True)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=False)

    ti = 4
    data_iter = iter(test_loader)
    for i in range(ti):
        data = next(data_iter)
    total_num = len(test_set)
    success_num = 0
    i = 0
    for x, label, target in tqdm(test_loader):
        # x, label, target = data
        x, label, target = x.cuda(), label.long().cuda(), target.long().cuda()
        x = x.transpose(2, 1).contiguous()
        x.requires_grad = True
        rx = x.clone()

        x_adv = AOA_Attack(model, x, label, target)
        pred, _, _ = model(x_adv)
        pred = torch.argmax(pred, dim=-1)
        if pred == target:
            success_num += 1
        i += 1
        if i % 50 == 0:
            print("current attack success rate is", success_num / i)

        # R0 = LRP_scores(model, x, label, label)
        # R1 = LRP_scores(model, x, label, target)
        # utils.pc_heatmap(x.transpose(2, 1)[0], R1[0].sum(-2).unsqueeze(-1))

    print("total attack success rate is", success_num/total_num)
    # utils.pc_heatmap(rx.transpose(2, 1)[0], R0[0].sum(-2).unsqueeze(-1))
    # print(x)


if __name__ == '__main__':
    main()
    print("End!!!")