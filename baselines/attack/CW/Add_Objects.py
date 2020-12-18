"""Implementation of optimization based attack,
    CW Attack for object adding.
Based on CVPR'19: Generating 3D Adversarial Point Clouds.
"""

import pdb
import copy
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from sklearn.cluster import DBSCAN

from util.pointnet_utils import normalize_points_np


def get_critical_points(model, pc, label, num):
    """Get top num important point coordinates for given model and pc.

    Args:
        model (torch.nn.Module): model to evaluate
        pc (torch.FloatTensor): input batch pc, [B, 3, K]
        label (torch.LontTensor): batch label, [B]
        num (int): number we want
    """
    B = label.shape[0]
    input_pc = pc.clone().detach().float().cuda()
    input_pc.requires_grad_()
    label = label.long().cuda()
    model.eval()
    # input and calculate gradient
    logits = model(input_pc)
    if isinstance(logits, tuple):  # PointNet
        logits = logits[0]
    loss = F.cross_entropy(logits, label)
    loss.backward()
    with torch.no_grad():
        grad = input_pc.grad.data  # [B, 3, K]
        grad = torch.sum(grad ** 2, dim=1)  # [B, K]
        # get top index of [B, num]
        _, idx = grad.topk(k=num, dim=-1)
        critical_points = torch.stack([
            pc[i, :, idx[i]] for i in range(B)
        ], dim=0).clone().detach()  # [B, 3, num]
    return critical_points


class CWAddObjects:
    """Class for CW attack.
    """

    def __init__(self, model, adv_func, dist_func, object_pc, attack_lr=1e-2,
                 init_weight=5., max_weight=40., binary_step=5,
                 num_iter=500, num_add=3, obj_num_p=64, scaling=0.3):
        """CW attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            dist_func (function): distance metric
            object_pc (np.ndarry): the point cloud of the added object
            attack_lr (float, optional): lr for optimization. Defaults to 1e-2.
            init_weight (float, optional): weight factor init. Defaults to 10.
            max_weight (float, optional): max weight factor. Defaults to 80.
            binary_step (int, optional): binary search step. Defaults to 10.
            num_iter (int, optional): max iter num in every search step. Defaults to 500.
            num_add (int, optional): number of adding clusters. Defaults to 3.
            obj_num_p (int, optional): number of points in an object. Defaults to 64.
            scaling (float, optional): scaling factor of the added object. Defaults to 0.3.
        """

        self.model = model.cuda()
        self.model.eval()

        self.adv_func = adv_func
        self.dist_func = dist_func
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.num_add = num_add
        self.obj_num_p = obj_num_p
        object_pc = self.process_object(object_pc, scaling)

        # [num_add, obj_num_p, 3]
        self.object_pc = np.zeros((self.num_add, self.obj_num_p, 3))
        for i in range(self.num_add):
            np.random.shuffle(object_pc)
            self.object_pc[i] = copy.deepcopy(object_pc[:self.obj_num_p])

    def process_object(self, pc, scaling):
        """Normalize and scale the added object point cloud."""
        pc = normalize_points_np(pc)
        pc = pc * scaling
        return pc

    def _init_centers(self, pc, label):
        """Clustering critical points for centers init.

        Args:
            pc (torch.FloatTensor): input batch pc, [B, 3, K]
            label (torch.LontTensor): batch label, [B]
        """
        batch = len(pc)
        num_cri = 128
        cri_points = get_critical_points(
            self.model, pc, label, num_cri)  # [B, 3, num_cri]
        batch_cri = [[] for _ in range(batch)]
        # perform DBSCAN clustering
        eps = 0.2
        min_number = 3
        for i in range(batch):
            points = cri_points[i].detach().cpu().numpy()  # [3, num_cri]
            points = np.transpose(points, [1, 0])  # [num_cri, 3]
            dbscan = DBSCAN(eps, min_samples=min_number)
            result = dbscan.fit_predict(points)
            filter_idx = (result > -0.5)  # get the index of non-outlier point
            result = result[filter_idx]
            points = points[filter_idx]
            labels, counts = np.unique(result, return_counts=True)
            sel_idx = np.argsort(counts)[-self.num_add:]
            # get the label idx for the top num_add clusters
            labels = labels[sel_idx]
            for one_label in labels:
                cluster_points = points[result == one_label]  # [num, 3]
                cluster_center = np.mean(cluster_points, axis=0)  # [3]
                # find the "center" points on the object surface
                # use the point with the min_dist to points.mean()
                dist = np.sum(
                    (cluster_points - cluster_center[None, :]) ** 2,
                    axis=1)  # [num]
                min_idx = np.argmin(dist)
                one_center = cluster_points[min_idx]
                batch_cri[i].append(copy.deepcopy(one_center))
            # in case not enough center points
            # just randomly select one
            while len(batch_cri[i]) < self.num_add:
                rand_idx = np.random.choice(len(points), 1)[0]
                rand_point = points[rand_idx]
                batch_cri[i].append(copy.deepcopy(rand_point))

        # batch_cri is np.array of shape [B, num_add, 3]
        return np.array(batch_cri)

    def _rotate_shift(self, points, angles, shifts):
        """Transform clusters to desired positions.

        Args:
            points (torch.FloatTensor): [B, num_add, obj_num_p, 3]
            angles (torch.FloatTensor): [B, num_add, 3]
            shifts (torch.FloatTensor): [B, num_add, 3]
        """
        batch = len(points)
        # [B, num_add]
        # TODO: different from the original paper
        # TODO: I only rotate along the y axis!
        angle = angles[..., 0]
        cosval, sinval = torch.cos(angle), torch.sin(angle)
        zeros = torch.zeros_like(cosval).float().cuda().detach()
        ones = torch.ones_like(cosval).float().cuda().detach()
        rot_mat = torch.stack([
            cosval,  zeros, sinval,
            zeros,   ones,   zeros,
            -sinval, zeros, cosval,
        ], dim=-1).view(batch * self.num_add, 3, 3)
        '''
        cos1, sin1 = torch.cos(angles[..., 0]), torch.sin(angles[..., 0])
        cos2, sin2 = torch.cos(angles[..., 1]), torch.sin(angles[..., 1])
        cos3, sin3 = torch.cos(angles[..., 2]), torch.sin(angles[..., 2])
        rot_mat = torch.stack([
            cos1*cos3 - cos2*sin1*sin3, -cos2*cos3*sin1 - cos1*sin3,  sin1*sin2,
            cos3*sin1 + cos1*cos2*sin3,  cos1*cos2*cos3 - sin1*sin3, -cos1*sin2,
            sin2*sin3,                        cos3*sin2,          cos2
        ], dim=-1).view(batch * self.num_add, 3, 3)
        '''
        points = points.view(batch * self.num_add, self.obj_num_p, 3)
        rot_points = torch.bmm(
            points, rot_mat).view(batch, self.num_add, self.obj_num_p, 3)
        rot_shift_points = rot_points + shifts[:, :, None, :]

        # [B, num_add, obj_num_p, 3]
        return rot_shift_points

    def attack(self, data, target):
        """Attack on given data to target.

        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.LongTensor): target output, [B]
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        data = data.transpose(1, 2).contiguous()  # [B, 3, K]
        ori_data = data.clone().detach()  # [B, 3, K]
        ori_data.requires_grad = False
        target = target.long().cuda().detach()  # [B]
        label_val = target.detach().cpu().numpy()  # [B]

        # weight factor for budget regularization
        lower_bound = np.zeros((B,))
        upper_bound = np.ones((B,)) * self.max_weight
        current_weight = np.ones((B,)) * self.init_weight

        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, self.num_add * self.obj_num_p))

        # centers is np.array of shape [B, num_add, 3]
        centers = self._init_centers(ori_data, target)
        shifts = torch.from_numpy(centers).float().cuda()
        shifts.requires_grad = False

        # angles init from zeros
        angles = torch.zeros((B, self.num_add, 3)).float().cuda()
        angles.requires_grad = False

        # duplicate from [num_add, obj_num_p, 3] to [B, na, np, 3]
        objects = np.tile(self.object_pc, (B, 1, 1, 1))
        objects = torch.from_numpy(objects).float().cuda()
        objects.requires_grad = False

        # perform binary search
        for binary_step in range(self.binary_step):
            # init adv objects, shifts and angles!
            adv_objects = objects + torch.randn(
                (B, self.num_add, self.obj_num_p, 3)).cuda() * 1e-7
            adv_objects.requires_grad_()  # [B, num_add, obj_num_p, 3]

            adv_shifts = shifts + torch.randn(
                (B, self.num_add, 3)).cuda() * 1e-7
            adv_shifts.requires_grad_()

            # init with different angles
            adv_angles = angles.detach().clone()
            adv_angles = torch.rand_like(
                adv_angles).float().cuda() * np.pi
            adv_angles.requires_grad_()

            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            opt = optim.Adam([adv_objects, adv_shifts, adv_angles],
                             lr=self.attack_lr, weight_decay=0.)

            adv_loss = torch.tensor(0.).cuda()
            dist_loss = torch.tensor(0.).cuda()

            total_time = 0.
            forward_time = 0.
            backward_time = 0.
            update_time = 0.

            for iteration in range(self.num_iter):
                t1 = time.time()

                # forward passing
                # rotate and shift, [B, num_add, obj_num_p, 3]
                adv_data = self._rotate_shift(
                    adv_objects, adv_angles, adv_shifts)
                adv_data = adv_data.view(
                    B, self.num_add * self.obj_num_p, 3)
                adv_data = adv_data.transpose(1, 2).contiguous()
                # now adv_data is [B, 3, num_add * obj_num_p]

                # concat added objects with real pc!
                cat_data = torch.cat([ori_data, adv_data], dim=-1)
                logits = self.model(cat_data)  # [B, num_classes]
                if isinstance(logits, tuple):  # PointNet
                    logits = logits[0]

                t2 = time.time()
                forward_time += t2 - t1

                # print
                pred = torch.argmax(logits, dim=-1)  # [B]
                success_num = (pred == target).sum().item()
                if iteration % (self.num_iter // 5) == 0:
                    print('Step {}, iteration {}, success {}/{}\n'
                          'adv_loss: {:.4f}, dist_loss: {:.4f}'.
                          format(binary_step, iteration, success_num, B,
                                 adv_loss.item(), dist_loss.item()))

                # record values!
                dist_val = self.dist_func(
                    adv_data.transpose(1, 2).contiguous(),
                    ori_data.transpose(1, 2).contiguous(),
                    adv_objects,
                    objects,
                    batch_avg=False).detach().cpu().numpy()  # [B]
                pred_val = pred.detach().cpu().numpy()  # [B]
                input_val = adv_data.detach().cpu().numpy()  # [B, 3, K]

                # update binary search
                for e, (dist, pred, label, ii) in \
                        enumerate(zip(dist_val, pred_val, label_val, input_val)):
                    if dist < bestdist[e] and pred == label:
                        bestdist[e] = dist
                        bestscore[e] = pred
                    if dist < o_bestdist[e] and pred == label:
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred
                        o_bestattack[e] = ii

                t3 = time.time()
                update_time += t3 - t2

                # compute loss and backward
                adv_loss = self.adv_func(logits, target).mean()
                dist_loss = self.dist_func(
                    adv_data.transpose(1, 2).contiguous(),
                    ori_data.transpose(1, 2).contiguous(),
                    adv_objects,
                    objects,
                    weights=torch.from_numpy(current_weight)).mean()
                loss = adv_loss + dist_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

                t4 = time.time()
                backward_time += t4 - t3
                total_time += t4 - t1

                if iteration % 100 == 0:
                    print('total: {:.2f}, for: {:.2f}, '
                          'back: {:.2f}, update: {:.2f}'.
                          format(total_time, forward_time,
                                 backward_time, update_time))
                    total_time = 0.
                    forward_time = 0.
                    backward_time = 0.
                    update_time = 0.
                    torch.cuda.empty_cache()

                # clip angle to be within [0, 2pi]
                with torch.no_grad():
                    adv_angles.data = adv_angles.data % (2. * np.pi)

            # adjust weight factor
            for e, label in enumerate(label_val):
                if bestscore[e] == label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                    # success
                    lower_bound[e] = max(lower_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                else:
                    # failure
                    upper_bound[e] = min(upper_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.

            torch.cuda.empty_cache()

        # end of CW attack
        # fail to attack some examples
        # just assign them with last time attack data
        fail_idx = (lower_bound == 0.)
        o_bestattack[fail_idx] = input_val[fail_idx]  # [B, 3, num]
        # return final results
        success_num = (lower_bound > 0.).sum()
        print('Successfully attack {}/{}'.format(success_num, B))

        # concatenate add and ori data
        ori_data = ori_data.detach().cpu().numpy()  # [B, 3, K]
        o_bestattack = np.concatenate([ori_data, o_bestattack], axis=-1)
        return o_bestdist, o_bestattack.transpose((0, 2, 1)), success_num
