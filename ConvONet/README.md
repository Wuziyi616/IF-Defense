# ConvONet based IF-Defense

This folder contains code for ConvONet based IF-Defense, mainly adopted from [ConvONet](https://github.com/autonomousvision/convolutional_occupancy_networks). We only provide the inference used in defense. The training code of ConvONet is nearly the same as their original repo so we do not provide them here.

## Requirements

### Extension Compilation

You need to compile some extension modules to run the code. The compiler on my server are gcc 5.5.0 and g++ 5.5.0. Simply run

```shell
python setup.py build_ext --inplace
```

### Pre-trained ConvONet Weight

We provide the ConvONet weight used in our experiments [here](https://drive.google.com/file/d/1x2N-g31a9ZiAPYiALL0dyJ9EtRlOQlt7/view?usp=sharing). Please download it and put it into the ```pretrain/``` folder.

## Usage

### ConvONet-Opt Defense

```shell
CUDA_VISIBLE_DEVICES=0 python opt_defense.py --sample_npoint=1024 --train=False --rep_weight=500.0 --data_root=path/to/adv_data.npz
```

The defense result will be saved in the ```ConvONet-Opt/``` folder in the adv_data's directory.

