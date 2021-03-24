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


def main():
    global BATCH_SIZE, BEST_WEIGHTS
    BATCH_SIZE = BATCH_SIZE[1024]
    BEST_WEIGHTS = BEST_WEIGHTS['mn40'][1024]
    cudnn.benchmark = True

    # build model
    model = PointNetCls(k=40, feature_transform=False)
    model = nn.DataParallel(model).cuda()
    model.eval()

    # model_vgg = torchvision.models.vgg16(pretrained=True)
    # model_vgg.eval()
    # layers = list(model_vgg._modules['features']) + utils.toconv(list(model_vgg._modules['classifier']))

    # load model weight
    print('Loading weight {}'.format(BEST_WEIGHTS['pointnet']))
    state_dict = torch.load(BEST_WEIGHTS['pointnet'])
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.module.load_state_dict(state_dict)\

    # load dataset
    test_set = ModelNet40Attack('data/attack_data.npz', num_points=1024,
                                normalize=True)
    test_loader = DataLoader(test_set, batch_size=2,
                             shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=False)

    x, label, target = next(iter(test_loader))
    x, label = x.cuda(), label.long().cuda()
    x = x.transpose(2, 1).contiguous()

    # stn layers
    iden = torch.from_numpy(np.array([
        1, 0, 0, 0, 1, 0, 0, 0, 1
    ]).astype(np.float32)).view(1, 9).repeat(x.shape[0], 1).cuda()
    stn = model.module.feat.stn
    stn_layers = [stn.conv1, nn.ReLU(), stn.conv2, nn.ReLU(), stn.conv3, nn.ReLU(), nn.MaxPool1d(1024),
                  stn.fc1, nn.ReLU(), stn.fc2, nn.ReLU(), stn.fc3, utils.Add_Iden_to_conv(iden), 'MatReshape']
    stn_layers = utils.toconv(stn_layers)
    stn_len = len(stn_layers)
    stn_x = [x] + [None]*stn_len
    for i in range(stn_len):
        if stn_layers[i] == 'MatReshape':
            stn_x[i+1] = stn_x[i].view(-1, 3, 3)
        else:
            stn_x[i+1] = stn_layers[i].forward(stn_x[i])

    # feat layers
    feat = model.module.feat
    feat_layers = ['transpose', feat.conv1, nn.ReLU(), feat.conv2, nn.ReLU(), feat.conv3, nn.MaxPool1d(1024)]
    feat_len = len(feat_layers)
    feat_x = [x] + [None]*feat_len
    feat_x[0] = torch.bmm(x.transpose(2, 1), stn_x[-1])
    for i in range(feat_len):
        if feat_layers[i] == 'transpose':
            feat_x[i+1] = feat_x[i].transpose(2, 1)
        elif feat_layers[i] == 'SqueezeLast':
            feat_x[i+1] = feat_x[i].view(-1, 1024)
        else:
            feat_x[i+1] = feat_layers[i].forward(feat_x[i])

    # classifier layers
    cls_layers = [model.module.fc1, nn.ReLU(), nn.Sequential(model.module.fc2, model.module.bn2), nn.ReLU(), model.module.fc3]
    cls_layers = utils.toconv(cls_layers)
    cls_len = len(cls_layers)
    cls_x = [feat_x[-1]] + [None]*cls_len
    for i in range(cls_len):
        cls_x[i+1] = cls_layers[i].forward(cls_x[i])

    # cls_x[-1] = cls_x[-1].squeeze()
    pred, _, _ = model(x)

    # LRP
    T = torch.arange(0, 40, dtype=torch.long).view(1, -1, 1).cuda()
    T = label.view(-1, 1, 1) == T

    # cls layers
    cls_R = [None]*cls_len + [cls_x[-1]*T]
    for i in range(0, cls_len)[::-1]:
        cls_x[i] = (cls_x[i].data).requires_grad_(True)
        if isinstance(cls_layers[i], nn.ReLU):
            cls_R[i] = cls_R[i+1]
        else:
            rho = lambda p: p
            incr = lambda y: y + 1e-9
            z = incr(utils.newlayer(cls_layers[i], rho).forward(cls_x[i]))  # step 1
            s = (cls_R[i + 1] / z).data                                     # step 2
            (z * s).sum().backward()
            c = cls_x[i].grad                                               # step 3
            cls_R[i] = (cls_x[i] * c).data                                  # step 4

    # feat layers
    feat_R = [None]*feat_len + [cls_R[0]]
    for i in range(0, feat_len)[::-1]:
        feat_x[i] = (feat_x[i].data).requires_grad_(True)
        if feat_layers[i] == 'transpose':
            feat_R[i] = feat_R[i+1].transpose(2, 1)
        elif isinstance(feat_layers[i], nn.MaxPool1d):
            max_id = torch.max(feat_x[i], dim=-1)[1]
            feat_R[i] = torch.zeros_like(feat_x[i])
            batch_ids = torch.arange(0, x.shape[0]).view(-1, 1)
            fids = torch.arange(0, 1024).view(1, -1)
            feat_R[i][batch_ids, fids, max_id] = feat_R[i+1][:, :, 0]
        elif isinstance(feat_layers[i], nn.ReLU):
            feat_R[i] = feat_R[i+1]
        else:
            rho = lambda p: p
            incr = lambda y: y+1e-9+0.25*((y**2).mean()**.5).data
            z = incr(utils.newlayer(feat_layers[i], rho).forward(feat_x[i]))    # step 1
            s = (feat_R[i + 1] / z).data                                        # step 2
            (z * s).sum().backward()
            c = feat_x[i].grad                                                  # step 3
            feat_R[i] = (feat_x[i] * c).data                                    # step 4

    # for bmm specific
    feat0_R = feat_R[0].transpose(2, 1)
    bmm_xR = torch.zeros_like(feat0_R)
    bmm_convs = utils.bmm_to_conv(stn_x[-1])
    for i in range(x.shape[0]):
        bmm_x = (x[i].unsqueeze(0).data).requires_grad_(True)
        lb = (x[i].unsqueeze(0).data * 0 - 1).requires_grad_(True)
        hb = (x[i].unsqueeze(0).data * 0 + 1).requires_grad_(True)
        z = bmm_convs[i].forward(bmm_x) + 1e-9               # step 1 (a)
        z -= utils.newlayer(bmm_convs[i], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
        z -= utils.newlayer(bmm_convs[i], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)
        s = (feat0_R[i].unsqueeze(0) / z).data                                                # step 2
        (z * s).sum().backward()
        c, cp, cm = bmm_x.grad, lb.grad, hb.grad                       # step 3
        bmm_xR[i] = (bmm_x * c + lb * cp + hb * cm).squeeze().data               # step 4

    stn_fR = torch.zeros_like(stn_x[-1])
    bmm_convs = utils.bmm_to_conv(x)
    for i in range(x.shape[0]):
        bmm_x = (stn_x[-1][i].unsqueeze(0).data).requires_grad_(True)
        rho = lambda p: p
        incr = lambda y: y + 1e-9 + 0.25 * ((y ** 2).mean() ** .5).data
        z = incr(utils.newlayer(bmm_convs[i], rho).forward(bmm_x))  # step 1
        s = (feat_R[0][i].unsqueeze(0) / z).data  # step 2
        (z * s).sum().backward()
        c = bmm_x.grad  # step 3
        stn_fR[i] = (bmm_x * c).squeeze().data  # step 4

    # stn layers
    stn_R = [None]*stn_len + [stn_fR]
    for i in range(1, stn_len)[::-1]:
        stn_x[i] = (stn_x[i].data).requires_grad_(True)
        if stn_layers[i] == 'MatReshape':
            stn_R[i] = stn_R[i+1].view(-1, 9, 1)
        elif isinstance(stn_layers[i], nn.MaxPool1d):
            max_id = torch.max(stn_x[i], dim=-1)[1]
            stn_R[i] = torch.zeros_like(stn_x[i])
            batch_ids = torch.arange(0, x.shape[0]).view(-1, 1)
            fids = torch.arange(0, 1024).view(1, -1)
            stn_R[i][batch_ids, fids, max_id] = stn_R[i+1][:, :, 0]
        elif isinstance(stn_layers[i], nn.ReLU):
            stn_R[i] = stn_R[i+1]
        else:
            rho = lambda p: p + 0.25 * p.clamp(min=0)
            incr = lambda y: y + 1e-9
            z = incr(utils.newlayer(stn_layers[i], rho).forward(stn_x[i]))  # step 1
            s = (stn_R[i + 1] / z).data                                     # step 2
            (z * s).sum().backward()
            c = stn_x[i].grad                                               # step 3
            stn_R[i] = (stn_x[i] * c).data                                  # step 4

    # first layer from stn
    stn_x[0] = (stn_x[0].data).requires_grad_(True)

    lb = (stn_x[0].data * 0 - 1).requires_grad_(True)
    hb = (stn_x[0].data * 0 + 1).requires_grad_(True)

    z = stn_layers[0].forward(stn_x[0]) + 1e-9                               # step 1 (a)
    z -= utils.newlayer(stn_layers[0], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
    z -= utils.newlayer(stn_layers[0], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)
    s = (stn_R[1] / z).data                                                     # step 2
    (z * s).sum().backward()
    c, cp, cm = stn_x[0].grad, lb.grad, hb.grad                             # step 3
    stn_R[0] = (stn_x[0] * c + lb * cp + hb * cm).data                              # step 4

    R0 = bmm_xR + stn_R[0]
    print(x)
    'SqueezeLast', stn.fc1, nn.ReLU(), stn.fc2, nn.ReLU(), stn.fc3, 'Add_Iden', 'MatReshape'


if __name__ == '__main__':
    main()
    print("End!!!")