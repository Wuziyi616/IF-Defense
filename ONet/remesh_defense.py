import os
import pdb
import tqdm
import trimesh
import argparse
import numpy as np

import torch

from im2mesh import config

from defense import SORDefense


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('--config', type=str,
                    default='configs/onet_mn40.yaml',
                    help='Path to config file.')
parser.add_argument('--sample_npoint', type=int, default=1024,
                    help='Re-sample points number per mesh.')
parser.add_argument('--padding_scale', type=float, default=0.9,
                    help='Used in pre-processing point clouds,'
                         'padding in unit cube')

parser.add_argument('--data_root', type=str, default='',
                    help='Path to point cloud npz file.')
parser.add_argument('--train', type=str2bool, default=False,
                    help='whether defend training data')

parser.add_argument('--sor', type=str2bool, default=True,
                    help='whether use SOR before reconstruction')
parser.add_argument('--sor_k', type=int, default=2,
                    help='KNN in SOR')
parser.add_argument('--sor_alpha', type=float, default=1.1,
                    help='Threshold = mean + alpha * std')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
args.input_npoint = cfg['data']['pointcloud_n']
args.threshold = cfg['test']['threshold']

device = torch.device("cuda")

# Model
model = config.get_model(cfg, device=device, dataset=None)
model.load_state_dict(torch.load(cfg['test']['model_file']))

# Generator
generator = config.get_generator(model, cfg, device=device)

# Generate
model.eval()


def normalize_pc(points):
    """points: [K, 3]"""
    points = points - np.mean(points, axis=0)[None, :]  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale
    return points


def sor_process(pc):
    """Use SOR to pre-process pc.
    Inputs:
        pc: [N, K, 3]

    Returns list of [K_i, 3]
    """
    N = len(pc)
    batch_size = 32
    sor_pc = []
    sor_defense = SORDefense(k=args.sor_k, alpha=args.sor_alpha)
    for i in range(0, N, batch_size):
        input_pc = pc[i:i + batch_size]  # [B, K, 3]
        input_pc = torch.from_numpy(input_pc).float().cuda()
        output_pc = sor_defense(input_pc)
        # to np array list
        output_pc = [
            one_pc.detach().cpu().numpy().
            astype(np.float32) for one_pc in output_pc
        ]
        sor_pc.append(output_pc)
    pc = []
    for i in range(len(sor_pc)):
        pc += sor_pc[i]  # list of [k, 3]
    assert len(pc[0].shape) == 2 and pc[0].shape[1] == 3
    return pc


def preprocess_pc(pc, num_points=None, padding_scale=1.):
    """Center and scale to be within unit cube.
    Inputs:
        pc: np.array of [K, 3]
        num_points: pick a subset of points as OccNet input.
        padding_scale: padding ratio in unit cube.
    """
    # normalize into unit cube
    center = np.mean(pc, axis=0)  # [3]
    centered_pc = pc - center
    max_dim = np.max(centered_pc, axis=0)  # [3]
    min_dim = np.min(centered_pc, axis=0)  # [3]
    scale = (max_dim - min_dim).max()
    scaled_centered_pc = centered_pc / scale * padding_scale

    # select a subset as ONet input
    if num_points is not None and \
            scaled_centered_pc.shape[0] > num_points:
        idx = np.random.choice(
            scaled_centered_pc.shape[0], num_points,
            replace=False)
        pc = scaled_centered_pc[idx]
    else:
        pc = scaled_centered_pc

    # to torch tensor
    torch_pc = torch.from_numpy(pc).\
        float().cuda().unsqueeze(0)
    return torch_pc


def reconstruct_mesh(pc,
                     padding_scale=args.padding_scale):
    '''Reconstruct a mesh from input point cloud.
    With potentially pre-processing and post-processing.
    '''
    # pre-process
    # only use coordinates information
    pc = pc[:, :3]
    pc = preprocess_pc(pc, num_points=args.input_npoint,
                       padding_scale=padding_scale)

    # ONet mesh generation
    # shape latent code, [B, c_dim (typically 512)]
    c = generator.model.encode_inputs(pc)

    # z is of no use
    z = generator.model.get_z_from_prior(
        (1,), sample=generator.sample).cuda()

    mesh = generator.generate_from_latent(z, c)
    return mesh


def resample_points(ori_pc,
                    num_points=args.sample_npoint):
    '''Apply reconstruction and re-sampling.'''
    re_mesh = reconstruct_mesh(ori_pc)
    # sample points from it
    try:
        pc, _ = trimesh.sample.sample_surface(
            re_mesh, count=num_points)
    # reconstruction might fail
    # random sample some points as defense results
    except IndexError:
        pc = np.zeros((num_points, 3), dtype=np.float32)
        if ori_pc.shape[0] > num_points:
            # apply SRS
            idx = np.random.choice(
                ori_pc.shape[0], num_points,
                replace=False)
            pc = ori_pc[idx]
        else:
            pc[:ori_pc.shape[0]] = ori_pc
    return pc


def get_save_name(path, train=args.train):
    """Saving name containing model settings."""
    sub_path = path.split('/')
    save_name = sub_path[-1]
    save_name = 'onet_remesh-' + save_name
    save_folder = path[:path.rindex(sub_path[-1])]
    save_folder = os.path.join(save_folder, 'ONet-Mesh')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, save_name)
    return save_path


def defend_npz_train_test_data(path=args.data_root):
    """Process both train and test data.
    """
    # load data
    npz = np.load(path)
    train_pc = npz['train_pc'][..., :3]
    test_pc = npz['test_pc'][..., :3]  # [..., K, 3]
    re_train_pc = np.zeros(
        (len(train_pc), args.sample_npoint, 3),
        dtype=np.float32)
    re_test_pc = np.zeros(
        (len(test_pc), args.sample_npoint, 3),
        dtype=np.float32)

    # possible SOR preprocess
    if args.sor:
        with torch.no_grad():
            train_pc = sor_process(train_pc)
            test_pc = sor_process(test_pc)
    torch.cuda.empty_cache()

    # remesh and resample
    for i in tqdm.trange(len(train_pc)):
        one_pc = train_pc[i]
        re_pc = resample_points(one_pc)
        re_train_pc[i] = re_pc
    for i in tqdm.trange(len(test_pc)):
        one_pc = test_pc[i]
        re_pc = resample_points(one_pc)
        re_test_pc[i] = normalize_pc(re_pc)

    # save
    save_path = get_save_name(path, train=True)
    np.savez(save_path,
             train_pc=re_train_pc.astype(np.float32),
             test_pc=re_test_pc.astype(np.float32),
             train_label=npz['train_label'],
             test_label=npz['test_label'])
    print('defense result saved to {}'.format(save_path))


def defend_npz_test_data(path=args.data_root):
    '''Apply defense to all pc in a npz file.
    Then save it for evaluation.
    '''
    # load data
    npz = np.load(path)
    test_pc = npz['test_pc'][..., :3]  # [..., K, 3]
    test_label = npz['test_label']
    try:
        target_label = npz['target_label']
    except KeyError:
        target_label = None

    # possible SOR preprocess
    if args.sor:
        with torch.no_grad():
            test_pc = sor_process(test_pc)
    torch.cuda.empty_cache()

    # reconstruct, re-sample
    re_test_pc = np.zeros(
        (len(test_pc), args.sample_npoint, 3),
        dtype=np.float32)
    for i in tqdm.trange(len(test_pc)):
        one_pc = test_pc[i]
        re_pc = resample_points(one_pc)
        re_test_pc[i] = normalize_pc(re_pc)

    # save new npz file
    save_path = get_save_name(path)
    if target_label is None:
        np.savez(save_path,
                 test_pc=re_test_pc.astype(np.float32),
                 test_label=test_label.astype(np.uint8))
    else:
        np.savez(save_path,
                 test_pc=re_test_pc.astype(np.float32),
                 test_label=test_label.astype(np.uint8),
                 target_label=target_label.astype(np.uint8))
    print('defense result saved to {}'.format(save_path))


if __name__ == '__main__':
    if args.train:
        # defend both train and test data
        # for hybrid training usage
        defend_npz_train_test_data(args.data_root)
    else:
        # if given a dir, then defend all the files in it
        data_root = args.data_root
        if os.path.isdir(data_root):
            all_files = os.listdir(data_root)
            for file in all_files:
                one_file = os.path.join(data_root, file)
                if os.path.isfile(one_file):
                    defend_npz_test_data(one_file)
        else:
            defend_npz_test_data(args.data_root)
