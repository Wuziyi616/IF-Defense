"""Apply baseline defense methods"""
import os
import tqdm
import argparse
import numpy as np

import torch

from defense import SRSDefense, SORDefense, DUPNet
from config import PU_NET_WEIGHT


def defend(data_root, one_defense):
    # save defense result
    sub_roots = data_root.split('/')
    filename = sub_roots[-1]
    data_folder = data_root[:data_root.rindex(filename)]
    save_folder = os.path.join(data_folder, one_defense)
    save_name = '{}_{}'.format(one_defense, filename)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # data to defend
    batch_size = 128
    npz_data = np.load(data_root)
    test_pc = npz_data['test_pc']
    test_label = npz_data['test_label']
    target_label = npz_data['target_label']

    # defense module
    if one_defense.lower() == 'srs':
        defense_module = SRSDefense(drop_num=args.srs_drop_num)
    elif one_defense.lower() == 'sor':
        defense_module = SORDefense(k=args.sor_k, alpha=args.sor_alpha)
    elif one_defense.lower() == 'dup':
        up_ratio = 4
        defense_module = DUPNet(sor_k=args.sor_k,
                                sor_alpha=args.sor_alpha,
                                npoint=1024, up_ratio=up_ratio)
        defense_module.pu_net.load_state_dict(
            torch.load(PU_NET_WEIGHT))
        defense_module.pu_net = defense_module.pu_net.cuda()

    # defend
    all_defend_pc = []
    for batch_idx in tqdm.trange(0, len(test_pc), batch_size):
        batch_pc = test_pc[batch_idx:batch_idx + batch_size]
        batch_pc = torch.from_numpy(batch_pc)[..., :3]
        batch_pc = batch_pc.float().cuda()
        defend_batch_pc = defense_module(batch_pc)

        # sor processed results have different number of points in each
        if isinstance(defend_batch_pc, list) or \
                isinstance(defend_batch_pc, tuple):
            defend_batch_pc = [
                pc.detach().cpu().numpy().astype(np.float32) for
                pc in defend_batch_pc
            ]
        else:
            defend_batch_pc = defend_batch_pc.\
                detach().cpu().numpy().astype(np.float32)
            defend_batch_pc = [pc for pc in defend_batch_pc]

        all_defend_pc += defend_batch_pc

    all_defend_pc = np.array(all_defend_pc)
    np.savez(os.path.join(save_folder, save_name),
             test_pc=all_defend_pc,
             test_label=test_label.astype(np.uint8),
             target_label=target_label.astype(np.uint8))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str, default='',
                        help='the npz data to defend')
    parser.add_argument('--defense', type=str, default='',
                        choices=['', 'srs', 'sor', 'dup'],
                        help='Defense method for input processing, '
                             'apply all if not specified')
    parser.add_argument('--srs_drop_num', type=int, default=500,
                        help='Number of point dropping in SRS')
    parser.add_argument('--sor_k', type=int, default=2,
                        help='KNN in SOR')
    parser.add_argument('--sor_alpha', type=float, default=1.1,
                        help='Threshold = mean + alpha * std')
    args = parser.parse_args()

    # defense method
    if args.defense == '':
        all_defense = ['srs', 'sor', 'dup']
    else:
        all_defense = [args.defense]

    # apply defense
    for one_defense in all_defense:
        print('{} defense'.format(one_defense))
        # if data_root is a folder
        # then apply defense to all the npz file in it
        if os.path.isdir(args.data_root):
            all_files = os.listdir(args.data_root)
            for one_file in all_files:
                data_path = os.path.join(args.data_root, one_file)
                if os.path.isfile(data_path):
                    defend(data_path, one_defense=one_defense)
        else:
            defend(args.data_root, one_defense=one_defense)
