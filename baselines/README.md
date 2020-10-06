# Basic Codebase

This folder contains some basic files for studying 3D adversarial attack and defense in point cloud. We implement training and testing of victim models, several baseline attack and defense methods. This codebase is highly extendable. You can easily add more victim model architectures, attacks and defenses based on it.

## Preliminary

### Data Storage

In order to simplify the process of data loading, I convert all the data (e.g. point cloud, GT label, target label) into a NumPy npz file. For example, for the ModelNet40 npz file ```data/MN40_random_2048.npz```, it contains the following data:

- **train/test_pc**: point clouds for training and testing, shape [num_data, num_points, 3]
- **train/test_label**: ground-truth labels for training and testing data
- **target_label**: pre-assigned target labels for targeted attack

### Configurations

Some commonly-used settings are hard-coded in ```config.py```. For example, the pre-trained weights of the victim models, the batch size used in model evaluation and attacks. **Please change this file if you want to use your own settings.**

## Requirements

### Data Preparation

Please download the point cloud data used for training, testing and attacking [here](https://drive.google.com/file/d/1o47ZvVcNvwBGv55xibEFw6KAOLaRF1IF/view?usp=sharing). Uncompress it to ```data/```.

- ```MN40_random_2048.npz``` is randomly sampled ModelNet40 data, which has 2048 points per point cloud
- ```attack_data.npz``` is the data used in all the attacks, which contains only test data and each point cloud has 1024 points, the pre-assigned target label is also in it
- ```onet_remesh/onet_opt/convonet_opt-MN40_random_2048.npz``` are the defended data from ```MN40_random_2048.npz``` using three variants of IF-Defense. They are used for hybrid training of the victim models (see below and the appendix of our paper for more details)

### Pre-trained Victim Models

We provided the pre-trained weights for the victim models used in our experiments. Download from [here](https://drive.google.com/file/d/1n9bRWyjPWSMyQktodCP2fnKpmzDFEf3e/view?usp=sharing) and uncompress them into ```pretrain/```. You can also train your victim models by yourself and put them into the folder.

**Note that**, if you want to use your own model/weight, please modify the variable called 'BEST_WEIGHTS' in ```config.py```.

## Usage

We briefly introduce the commands for different tasks here. Please refer to ```command.txt``` for more details about the parameters in each commands.

### Training Victim Models

We provide four types victim models for the experiments, **PointNet, PointNet++, DGCNN and PointConv**, where *Single Scale Grouping (SSG)* is adopted for PointNet++ and PointConv.

#### Train on Clean Data

Train a model on the randomly sampled ModelNet40 dataset:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --model={$MODEL} --num_points=1024
```

This will train the model for 200 epochs using the Adam optimizer with learning rate starting from 1e-3 and cosine decay to 1e-5.

The pre-trained weights we provided using this command achieve slightly lower accuracy than reported in their paper. This is because we do not use any tricks (e.g. label smoothing in the official TensorFlow implementation of [DGCNN](https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/models/dgcnn.py#L105)), and we train/test on randomly sampled point clouds while some official implementations are train/tested on Farthest Point Sampling (FPS) sampled point clouds.

#### Hybrid Training

The victim models trained purely on clean data degrade the accuracy by up to 5% on the IF-Defense defended data, which is unacceptable for a defense. So we add the defense data as a simple data augmentation to achieve negligible degradation:

```shell
CUDA_VISIBLE_DEVICES=0 python hybrid_train.py --model={$MODEL} --num_points=1024 --dataset={$DATASET} --def_data=path/to/defense_data.npz
```

Please refer to the appendix of our paper for more details about hybrid training. And see ```command.txt``` for the usage of argument {$DATASET}.

### Attacks

We implement **Perturb**, **Add**, **kNN**, variants of **FGM** and **Drop** attack. The attack scripts are in ```attack_scripts/``` folder. Except for Drop method that cannot targetedly attack the victim model, we perform targeted attack according to pre-assigned and fixed target labels.

The attacks are very time-consuming (e.g. 10-step binary search in Perturb method), so we use multi-GPU support provided by DistributedDataParallel in PyTorch to accelerate the process. For detailed usage of each attack method, please refer to ```command.txt```.

### Defenses

We implement **SRS**, **SOR** and **DUP-Net** defense. To apply a defense, simply run

```shell
CUDA_VISIBLE_DEVICES=0 python defend_npz.py --data_root=path/to/adv_data.npy --defense=srs/sor/dup/''
```

The defense result will still be a NumPy npz file saved in the same directory as the adv_data.

### Evaluating Victim Models

You can test the attacks/defenses performance using ```inference.py```:

```shell
CUDA_VISIBLE_DEVICES=0 python inference.py --num_points=1024 --mode=normal/target --model={$MODEL} --normalize_pc=True/False --dataset={$DATASET} --data_root=path/to/test_data.npz
```

Please refer to ```command.txt``` for details about different parameters.

