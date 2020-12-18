"""Test the victim models"""
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import ModelNet40Attack, ModelNet40
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from util.utils import AverageMeter, str2bool, set_seed
from config import BEST_WEIGHTS
from config import MAX_TEST_BATCH as BATCH_SIZE
from config import MAX_DUP_TEST_BATCH as DUP_BATCH_SIZE


def get_model_name(npz_path):
    """Get the victim model name from npz file path."""
    if 'dgcnn' in npz_path.lower():
        return 'dgcnn'
    if 'pointconv' in npz_path.lower():
        return 'pointconv'
    if 'pointnet2' in npz_path.lower():
        return 'pointnet2'
    if 'pointnet' in npz_path.lower():
        return 'pointnet'
    print('Victim model not recognized!')
    exit(-1)


def test_target():
    """Target test mode.
    Show both classification accuracy and target success rate.
    """
    model.eval()
    acc_save = AverageMeter()
    success_save = AverageMeter()
    with torch.no_grad():
        for data, label, target in test_loader:
            data, label, target = \
                data.float().cuda(), label.long().cuda(), target.long().cuda()
            # to [B, 3, N] point cloud
            data = data.transpose(1, 2).contiguous()
            batch_size = label.size(0)
            # batch in
            if args.model.lower() == 'pointnet':
                logits, _, _ = model(data)
            else:
                logits = model(data)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == label).sum().float() / float(batch_size)
            acc_save.update(acc.item(), batch_size)
            success = (preds == target).sum().float() / float(batch_size)
            success_save.update(success.item(), batch_size)

    print('Overall accuracy: {:.4f}, '
          'attack success rate: {:.4f}'.
          format(acc_save.avg, success_save.avg))


def test_normal():
    """Normal test mode.
    Test on all data.
    """
    model.eval()
    acc_save = AverageMeter()
    with torch.no_grad():
        for data, label in test_loader:
            data, label = \
                data.float().cuda(), label.long().cuda()
            # to [B, 3, N] point cloud
            data = data.transpose(1, 2).contiguous()
            batch_size = label.size(0)
            # batch in
            if args.model.lower() == 'pointnet':
                logits, _, _ = model(data)
            else:
                logits = model(data)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == label).sum().float() / float(batch_size)
            acc_save.update(acc.item(), batch_size)

    print('Overall accuracy: {:.4f}'.format(acc_save.avg))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'target'],
                        help='Testing mode')
    parser.add_argument('--model', type=str, default='', metavar='MODEL',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', ''],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]. '
                             'If not specified, judge from data_root')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40',
                                 'opt_mn40', 'conv_opt_mn40'])
    parser.add_argument('--normalize_pc', type=str2bool, default=False,
                        help='normalize in dataloader')
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',
                        help='Size of batch, use config if not specified')

    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='',
                        help='Model weight to load, use config if not specified')
    args = parser.parse_args()

    # victim model
    if not args.model:
        args.model = get_model_name(args.data_root)

    # random seed
    set_seed(1)

    # in case adding attack
    if 'add' in args.data_root.lower():
        # we add 512 points in adding attacks
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 512
        elif args.num_points == 1024 + 512:
            num_points = 1024
    elif 'cluster' in args.data_root.lower():
        # we add 3*32=96 points in adding cluster attack
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 3 * 32
        elif args.num_points == 1024 + 3 * 32:
            num_points = 1024
    elif 'object' in args.data_root.lower():
        # we add 3*64=192 points in adding object attack
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 3 * 64
        elif args.num_points == 1024 + 3 * 64:
            num_points = 1024
    else:
        num_points = args.num_points

    # determine the weight to use
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][num_points]
    BATCH_SIZE = BATCH_SIZE[num_points]
    DUP_BATCH_SIZE = DUP_BATCH_SIZE[num_points]
    if args.batch_size == -1:  # automatic assign
        args.batch_size = BATCH_SIZE[args.model]
    # add point attack has more points in each point cloud
    if 'ADD' in args.data_root:
        args.batch_size = int(args.batch_size / 1.5)
    # sor processed point cloud has different points in each
    # so batch size only can be 1
    if 'sor' in args.data_root:
        args.batch_size = 1

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

    # load model weight
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(BEST_WEIGHTS[args.model]))

    # prepare data
    if args.mode == 'target':
        test_set = ModelNet40Attack(args.data_root, num_points=args.num_points,
                                    normalize=args.normalize_pc)
    else:
        test_set = ModelNet40(args.data_root, num_points=args.num_points,
                              normalize=args.normalize_pc, partition='test',
                              augmentation=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=8,
                             pin_memory=True, drop_last=False)

    # test
    if args.mode == 'normal':
        test_normal()
    else:
        test_target()
