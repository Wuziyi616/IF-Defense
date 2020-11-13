"""Optimizing input init points to object surface."""

import os
import pdb
import tqdm
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from src import config

from defense import SORDefense
from defense import repulsion_loss


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('--config', type=str,
                    default='configs/convonet_3plane_mn40.yaml',
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

parser.add_argument('--init_sigma', type=float, default=0.01,
                    help='sigma for normal dist used in ori_init')
parser.add_argument('--iterations', type=int, default=200,
                    help='Optimization iterations.')
parser.add_argument('--batch_size', type=int, default=192,
                    help='Batch process points.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='lr in optimization')
parser.add_argument('--rep_weight', type=float, default=500.,
                    help='loss weight for repulsion term')

parser.add_argument('--sor', type=str2bool, default=True,
                    help='whether use SOR before reconstruction')
parser.add_argument('--sor_k', type=int, default=2,
                    help='KNN in SOR')
parser.add_argument('--sor_alpha', type=float, default=1.1,
                    help='Threshold = mean + alpha * std')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

device = torch.device("cuda")

args.threshold = cfg['test']['threshold']
args.input_npoint = cfg['data']['pointcloud_n']

# Model
model = config.get_model(cfg, device=device, dataset=None)
model.load_state_dict(torch.load(cfg['test']['model_file']))

# Generator
generator = config.get_generator(model, cfg, device=device)

# model and generator not updated
model.eval()
for p in model.parameters():
    p.requires_grad = False


def normalize_batch_pc(points):
    """points: [batch, K, 3]"""
    centroid = torch.mean(points, dim=1)  # [batch, 3]
    points -= centroid[:, None, :]  # center, [batch, K, 3]
    dist = torch.sum(points ** 2, dim=2) ** 0.5  # [batch, K]
    max_dist = torch.max(dist, dim=1)[0]  # [batch]
    points /= max_dist[:, None, None]
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
    # torch_pc is ONet input, have fixed number of points
    # torch_all_pc is for initializing the defense point cloud
    torch_pc = torch.from_numpy(pc).\
        float().cuda().unsqueeze(0)
    torch_all_pc = torch.from_numpy(scaled_centered_pc).\
        float().cuda().unsqueeze(0)
    return torch_all_pc, torch_pc


def init_points(pc):
    """Initialize points to be optimized.

    Args:
        pc (tensor): input (adv) pc, [B, N, 3]
    """
    with torch.no_grad():
        B = len(pc)
        # init points from ori_points with noise
        # if not enough points in input pc, randomly duplicate
        # else select a subset
        if isinstance(pc, list):  # after SOR
            idx = [
                torch.randint(
                    0, len(one_pc),
                    (args.sample_npoint,)).long().cuda() for one_pc in pc
            ]
        else:
            idx = torch.randint(
                0, pc.shape[1], (B, args.sample_npoint)).long().cuda()
        points = torch.stack([
            pc[i][idx[i]] for i in range(B)
        ], dim=0).float().cuda()

        # add noise
        noise = torch.randn_like(points) * args.init_sigma
        points = torch.clamp(
            points + noise,
            min=-0.5 * args.padding_scale,
            max=0.5 * args.padding_scale)
    return points


def optimize_points(opt_points, z, c,
                    rep_weight=1.,
                    iterations=1000,
                    printing=False):
    """Optimization process on point coordinates.

    Args:
        opt_points (tensor): input init points to be optimized
        z (tensor): latent code
        c (tensor): feature vector
        iterations (int, optional): opt iter. Defaults to 1000.
        printing (bool, optional): print info. Defaults to False.
    """
    # 2 losses in total
    # Geo-aware loss enforces occ_value = occ_threshold by BCE
    # Dist-aware loss pushes points uniform by repulsion loss
    opt_points = opt_points.float().cuda()
    opt_points.requires_grad_()
    B, K = opt_points.shape[:2]

    # GT occ for surface
    with torch.no_grad():
        occ_threshold = torch.ones(
            (B, K)).float().cuda() * args.threshold

    opt = torch.optim.Adam([opt_points], lr=args.lr)

    # start optimization
    for i in range(iterations + 1):
        # 1. occ = threshold
        occ_value = generator.model.decode(opt_points, c).logits
        occ_loss = F.binary_cross_entropy_with_logits(
            occ_value, occ_threshold, reduction='none')  # [B, K]
        occ_loss = torch.mean(occ_loss)
        occ_loss = occ_loss * K

        # 2. repulsion loss
        rep_loss = torch.tensor(0.).float().cuda()
        if rep_weight > 0.:
            rep_loss = repulsion_loss(opt_points)  # [B]
            rep_loss = torch.mean(rep_loss)
            rep_loss = rep_loss * rep_weight

        loss = occ_loss + rep_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if printing and i % 100 == 0:
            print('iter {}, loss {:.4f}'.format(i, loss.item()))
            print('occ loss: {:.4f}, '
                  'rep loss: {:.4f}\n'
                  'occ value mean: {:.4f}'.
                  format(occ_loss.item(),
                         rep_loss.item(),
                         torch.sigmoid(occ_value).mean().item()))
    opt_points.detach_()
    opt_points = normalize_batch_pc(opt_points)
    return opt_points.detach().cpu().numpy()


def get_save_name(path, train=args.train):
    """Saving name containing model settings."""
    sub_path = path.split('/')
    save_name = sub_path[-1]
    save_name = 'convonet_opt-' + save_name
    save_folder = path[:path.rindex(sub_path[-1])]
    save_folder = os.path.join(save_folder, 'ConvONet-Opt')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, save_name)
    return save_path


def defend_point_cloud(pc):
    """Apply defense to input point clouds.

    Args:
        pc (tensor): [num_data, K, 3]
    """
    # possible SOR preprocessor
    if args.sor:
        with torch.no_grad():
            pc = sor_process(pc)
    torch.cuda.empty_cache()

    opt_pc = np.zeros(
        (len(pc), args.sample_npoint, 3),
        dtype=np.float32)

    # batch process
    for idx in tqdm.trange(0, len(pc), args.batch_size):
        # prepare for input
        with torch.no_grad():
            batch_pc = pc[idx:idx + args.batch_size]  # [B, K, 3]
            # preprocess
            batch_proc_pc = [
                preprocess_pc(
                    one_pc, num_points=args.input_npoint,
                    padding_scale=args.padding_scale) for one_pc in batch_pc
            ]
            # the selected input_n points from batch_pc after preprocess
            # sel_pc are for ONet input and have fixed number of points
            batch_proc_sel_pc = torch.cat([
                one_pc[1] for one_pc in batch_proc_pc
            ], dim=0).float().cuda()
            # proc_pc may have different num_points because of SOR
            # they're used for initializing the defense point clouds
            try:
                batch_proc_pc = torch.cat([
                    one_pc[0] for one_pc in batch_proc_pc
                ], dim=0).float().cuda()
            except RuntimeError:
                batch_proc_pc = [
                    one_pc[0][0] for one_pc in batch_proc_pc
                ]  # list of [num, 3]

            # get latent feature vector c
            # c is [B, c_dim (typically 512)]
            c = generator.model.encode_inputs(batch_proc_sel_pc)

            # z is of no use
            z = None

        # init points and optimize
        points = init_points(batch_proc_pc)
        points.requires_grad_()
        points = optimize_points(points, z, c,
                                 rep_weight=args.rep_weight,
                                 iterations=args.iterations,
                                 printing=True)
        opt_pc[idx:idx + args.batch_size] = points

    return opt_pc


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

    # defend
    def_test_pc = defend_point_cloud(test_pc)

    # save new npz file
    save_path = get_save_name(path)
    if target_label is None:
        np.savez(save_path,
                 test_pc=def_test_pc.astype(np.float32),
                 test_label=test_label.astype(np.uint8))
    else:
        np.savez(save_path,
                 test_pc=def_test_pc.astype(np.float32),
                 test_label=test_label.astype(np.uint8),
                 target_label=target_label.astype(np.uint8))
    print('defense result saved to {}'.format(save_path))


def defend_npz_train_test_data(path=args.data_root):
    '''Apply defense to all pc in a npz file.
    Then save it for evaluation.
    '''
    # load data
    npz = np.load(path)
    train_pc = npz['train_pc'][..., :3]  # [num_data, K, 3]
    test_pc = npz['test_pc'][..., :3]
    train_label = npz['train_label']
    test_label = npz['test_label']

    # defend
    def_train_pc = defend_point_cloud(train_pc)
    def_test_pc = defend_point_cloud(test_pc)

    # save new npz file
    save_path = get_save_name(path)
    np.savez(save_path,
             train_pc=def_train_pc.astype(np.float32),
             train_label=train_label.astype(np.uint8),
             test_pc=def_test_pc.astype(np.float32),
             test_label=test_label.astype(np.uint8))
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
