import numpy as np

from torch.utils.data import Dataset

from util.pointnet_utils import normalize_points_np, random_sample_points_np
from util.augmentation import rotate_point_cloud, jitter_point_cloud


def load_data(data_root, partition='train'):
    npz = np.load(data_root, allow_pickle=True)
    if partition == 'train':
        return npz['train_pc'], npz['train_label']
    elif partition == 'attack':
        return npz['test_pc'], npz['test_label'], npz['target_label']
    else:
        return npz['test_pc'], npz['test_label']


class ModelNet40(Dataset):
    """General ModelNet40 dataset class."""

    def __init__(self, data_root, num_points, normalize=True,
                 partition='train', augmentation=None):
        assert partition in ['train', 'test']
        self.data, self.label = load_data(data_root, partition=partition)
        self.num_points = num_points
        self.normalize = normalize
        self.partition = partition
        self.augmentation = (partition == 'train') if \
            augmentation is None else augmentation

    def __getitem__(self, item):
        """Returns: point cloud as [N, 3] and its label as a scalar."""
        pc = self.data[item][:, :3]
        if self.partition == 'test':
            pc = pc[:self.num_points]
        else:
            pc = random_sample_points_np(pc, self.num_points)

        label = self.label[item]

        if self.normalize:
            pc = normalize_points_np(pc)

        if self.augmentation:
            pc = rotate_point_cloud(pc)
            pc = jitter_point_cloud(pc)

        return pc, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40Hybrid(ModelNet40):
    """ModelNet40 dataset class.
    Add defense point clouds for hybrid training.
    """

    def __init__(self, ori_data, def_data, num_points,
                 normalize=True, partition='train',
                 augmentation=None, subset='ori'):
        assert partition in ['train', 'test']
        ori_data, ori_label = load_data(ori_data, partition=partition)
        ori_data = ori_data[..., :3]
        def_data, def_label = load_data(def_data, partition=partition)
        def_data = def_data[..., :3]
        # concatenate two data
        if partition == 'train':
            self.data = np.concatenate([
                ori_data, def_data], axis=0)
            self.label = np.concatenate([
                ori_label, def_label], axis=0)
        else:  # only take subset data for testing
            if subset == 'ori':
                self.data = ori_data
                self.label = ori_label
            elif subset == 'def':
                self.data = def_data
                self.label = def_label
            else:
                print('Subset not recognized!')
                exit(-1)
        # shuffle real and defense data
        if partition == 'train':
            idx = list(range(len(self.label)))
            np.random.shuffle(idx)
            self.data = self.data[idx]
            self.label = self.label[idx]
        self.num_points = num_points
        self.normalize = normalize
        self.partition = partition
        self.augmentation = (partition == 'train') if \
            augmentation is None else augmentation


class ModelNet40Normal(Dataset):
    """Modelnet40 dataset with point normals.
    This is used in kNN attack which requires normal in projection operation.
    """

    def __init__(self, data_root, num_points, normalize=True):
        self.data, self.label = \
            load_data(data_root, partition='test')
        self.num_points = num_points
        # not for training, so no need to consider augmentation
        self.normalize = normalize

    def __getitem__(self, item):
        """Returns: point cloud as [N, 6] and its label as a scalar."""
        pc = self.data[item][:self.num_points, :6]
        label = self.label[item]

        if self.normalize:
            pc[:, :3] = normalize_points_np(pc[:, :3])

        return pc, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40Attack(Dataset):
    """Modelnet40 dataset for target attack evaluation.
    We return an additional target label for an example.
    """

    def __init__(self, data_root, num_points, normalize=True):
        self.data, self.label, self.target = \
            load_data(data_root, partition='attack')
        self.num_points = num_points
        self.normalize = normalize

    def __getitem__(self, item):
        """Returns: point cloud as [N, 3], its label as a scalar
            and its target label for attack as a scalar.
        """
        pc = self.data[item][:self.num_points, :3]
        label = self.label[item]
        target = self.target[item]

        if self.normalize:
            pc = normalize_points_np(pc)

        return pc, label, target

    def __len__(self):
        return self.data.shape[0]


class ModelNet40NormalAttack(Dataset):
    """Modelnet40 dataset with point normals and target label."""

    def __init__(self, data_root, num_points, normalize=True):
        self.data, self.label, self.target = \
            load_data(data_root, partition='attack')
        self.num_points = num_points
        self.normalize = normalize

    def __getitem__(self, item):
        """Returns: point cloud as [N, 6], its label as a scalar
            and its target label for attack as a scalar.
        """
        pc = self.data[item][:self.num_points, :6]
        label = self.label[item]
        target = self.target[item]

        if self.normalize:
            pc[:, :3] = normalize_points_np(pc[:, :3])

        return pc, label, target

    def __len__(self):
        return self.data.shape[0]
