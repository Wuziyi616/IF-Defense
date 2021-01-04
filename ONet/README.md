# ONet based IF-Defense

This folder contains code for ONet based IF-Defense, mainly adopted from [ONet](https://github.com/autonomousvision/occupancy_networks). We only provide the inference used in defense. The training code of ONet is nearly the same as their original repo so we do not provide them here.

## Requirements

### Extension Compilation

You need to compile some extension modules to run the code. The compiler on my server are gcc 5.5.0 and g++ 5.5.0. Simply run

```shell
python setup.py build_ext --inplace
```

### Pre-trained ONet Weight

We provide the ONet weight used in our experiments [here](https://drive.google.com/file/d/1GnSisZGcN_G38YzauWiYDLN3_gZq03hi/view?usp=sharing). Please download it and put it into the ```pretrain/``` folder.

## Usage

### ONet-Mesh Defense

```shell
CUDA_VISIBLE_DEVICES=0 python remesh_defense.py --sample_npoint=1024 --train=False --data_root=path/to/adv_data.npz
```

The defense result will be saved in the ```ONet-Mesh/``` folder in the adv_data's directory.

### ONet-Opt Defense

```shell
CUDA_VISIBLE_DEVICES=0 python opt_defense.py --sample_npoint=1024 --train=False --rep_weight=500.0 --data_root=path/to/adv_data.npz
```

The defense result will be saved in the ```ONet-Opt/``` folder in the adv_data's directory.

## Dataset Preparation

As required by [this issue](https://github.com/Wuziyi616/IF-Defense/issues/2), I upload the scripts I used to generate ModelNet40 training data to train implicit function networks in ```data_proc/``` folder. They are adopted from the original [ONet](https://github.com/autonomousvision/occupancy_networks#building-the-dataset) repo but I made some modifications (e.g. use Manifold to generate watertight meshes). Please refer to the issue for more details about training on ModelNet40 dataset.



