"""File for hybrid training on the victim models"""
import os
import pdb
import time
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter

from dataset import ModelNet40Hybrid
from model import DGCNN, PointNetCls, feature_transform_reguliarzer, \
    PointNet2ClsSsg, PointConvDensityClsSsg
from torch.utils.data import DataLoader
from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed


def train(start_epoch):
    best_ori_acc = 0
    best_ori_epoch = 0
    best_def_acc = 0
    best_def_epoch = 0
    best_weight = copy.deepcopy(model.state_dict())

    # training begins
    for epoch in range(start_epoch, args.epochs + 1):
        step_count = 0
        all_loss_save = AverageMeter()
        if args.model.lower() == 'pointnet':
            loss_save = AverageMeter()
            fea_loss_save = AverageMeter()
        acc_save = AverageMeter()
        model.train()

        # one epoch begins
        for data, label in train_loader:
            step_count += 1
            with torch.no_grad():
                data, label = data.float().cuda(), label.long().cuda()
                # to [B, 3, N] point cloud
                data = data.transpose(1, 2).contiguous()

            batch_size = data.size(0)
            opt.zero_grad()

            # calculate loss and BP
            if args.model.lower() == 'pointnet':
                # we may need to calculate feature_transform loss
                logits, trans, trans_feat = model(data)
                loss = criterion(logits, label, False)
                if args.feature_transform:
                    fea_loss = feature_transform_reguliarzer(
                        trans_feat) * 0.001
                else:
                    fea_loss = torch.tensor(0.).cuda()
                all_loss = loss + fea_loss
                all_loss.backward()
                opt.step()

                # calculate training accuracy
                acc = (torch.argmax(logits, dim=-1) ==
                       label).sum().float() / float(batch_size)

                # statistics accumulation
                all_loss_save.update(all_loss.item(), batch_size)
                loss_save.update(loss.item(), batch_size)
                fea_loss_save.update(fea_loss.item(), batch_size)
                acc_save.update(acc.item(), batch_size)
                if step_count % args.print_iter == 0:
                    print('Epoch {}, step {}, lr: {:.6f}\n'
                          'All loss: {:.4f}, loss: {:.4f}, Fea loss: {:.4f}\n'
                          'Train acc: {:.4f}'.
                          format(epoch, step_count, get_lr(opt),
                                 all_loss_save.avg, loss_save.avg,
                                 fea_loss_save.avg, acc_save.avg))
            else:
                logits = model(data)
                all_loss = criterion(logits, label, False)
                all_loss.backward()
                opt.step()

                # calculate training accuracy
                acc = (torch.argmax(logits, dim=-1) ==
                       label).sum().float() / float(batch_size)

                # statistics accumulation
                all_loss_save.update(all_loss.item(), batch_size)
                acc_save.update(acc.item(), batch_size)
                if step_count % args.print_iter == 0:
                    print('Epoch {}, step {}, lr: {:.6f}\n'
                          'All loss: {:.4f}, train acc: {:.4f}'.
                          format(epoch, step_count, get_lr(opt),
                                 all_loss_save.avg, acc_save.avg))
                    torch.cuda.empty_cache()

        # eval
        if epoch % 10 == 0 or epoch > 180:
            ori_acc = test(ori_test_loader)
            def_acc = test(def_test_loader)
            if ori_acc > best_ori_acc:
                best_ori_acc = ori_acc
                best_ori_epoch = epoch
            if def_acc > best_def_acc:
                best_def_acc = def_acc
                best_def_epoch = epoch

            print('Epoch {}, ori acc {:.4f}, def acc {:.4f}\n'
                  'Currently best ori acc {:.4f} at epoch {}\n'
                  'Currently best def acc {:.4f} at epoch {}'.
                  format(epoch, ori_acc, def_acc,
                         best_ori_acc, best_ori_epoch,
                         best_def_acc, best_def_epoch))
            torch.save(model.state_dict(),
                       os.path.join(
                           logs_dir,
                           'model{}_acc_{:.4f}_loss_{:.4f}_lr_{:.6f}.pth'.
                           format(epoch, def_acc, all_loss_save.avg, get_lr(opt))))
            torch.cuda.empty_cache()
            logger.add_scalar('test/ori_acc', ori_acc, epoch)
            logger.add_scalar('test/def_acc', def_acc, epoch)

        logger.add_scalar('train/loss', all_loss_save.avg, epoch)
        logger.add_scalar('train/lr', get_lr(opt), epoch)
        scheduler.step(epoch)

    # save the best model
    torch.save(best_weight,
               os.path.join(logs_dir,
                            'BEST_model{}_acc_{:.4f}.pth'.
                            format(best_def_epoch, best_def_acc)))


def test(loader):
    model.eval()
    acc_save = AverageMeter()
    with torch.no_grad():
        for data, label in loader:
            data, label = data.float().cuda(), label.long().cuda()
            # to [B, 3, N] point cloud
            data = data.transpose(1, 2).contiguous()
            batch_size = data.size(0)
            if args.model.lower() == 'pointnet':
                logits, _, _ = model(data)
            else:
                logits = model(data)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == label).sum().float() / float(batch_size)
            acc_save.update(acc.item(), batch_size)
    return acc_save.avg


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--ori_data', type=str,
                        default='data/MN40_random_2048.npz')
    parser.add_argument('--def_data', type=str,
                        default='data/')
    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str,
                        default='remesh_mn40',
                        choices=['remesh_mn40', 'opt_mn40', 'conv_opt_mn40'])
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=201, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate for the optimizer')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--print_iter', type=int, default=100,
                        help='Print interval')
    args = parser.parse_args()
    set_seed(1)
    print(args)

    # enable cudnn benchmark
    cudnn.benchmark = True

    # build model
    if args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
    else:
        print('Model not recognized')
        exit(-1)

    model = nn.DataParallel(model).cuda()

    # use Adam optimizer, cosine lr decay
    opt = optim.Adam(model.parameters(), lr=args.lr,
                     weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    # prepare data, we meed one hybrid training set and two separate test set
    train_set = ModelNet40Hybrid(args.ori_data, args.def_data,
                                 num_points=args.num_points,
                                 normalize=True,
                                 partition='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=8,
                              pin_memory=True, drop_last=True)

    ori_test_set = ModelNet40Hybrid(args.ori_data, args.def_data,
                                    num_points=args.num_points,
                                    normalize=True,
                                    partition='test', subset='ori')
    ori_test_loader = DataLoader(ori_test_set, batch_size=args.batch_size * 2,
                                 shuffle=False, num_workers=8,
                                 pin_memory=True, drop_last=False)
    def_test_set = ModelNet40Hybrid(args.ori_data, args.def_data,
                                    num_points=args.num_points,
                                    normalize=True,
                                    partition='test', subset='def')
    def_test_loader = DataLoader(def_test_set, batch_size=args.batch_size * 2,
                                 shuffle=False, num_workers=8,
                                 pin_memory=True, drop_last=False)

    # loss function using cross entropy without label smoothing
    criterion = cal_loss

    # save folder
    start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logs_dir = "logs/{}/{}/{}_{}".format(args.dataset, args.model,
                                         start_datetime, args.num_points)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger = SummaryWriter(os.path.join(logs_dir, 'logs'))

    # start training
    train(1)
