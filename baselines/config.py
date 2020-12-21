"""Config file for automatic code running
Assign some hyper-parameters, e.g. batch size for attack
"""
BEST_WEIGHTS = {
    # trained on standard mn40 dataset
    'mn40': {
        1024: {
            'pointnet': 'pretrain/mn40/pointnet.pth',
            'pointnet2': 'pretrain/mn40/pointnet2.pth',
            'pointconv': 'pretrain/mn40/pointconv.pth',
            'dgcnn': 'pretrain/mn40/dgcnn.pth',
        },
    },
    # trained on mn40 + ONet remesh-resampled mn40
    'remesh_mn40': {
        1024: {
            'pointnet': 'pretrain/remesh_mn40/pointnet.pth',
            'pointnet2': 'pretrain/remesh_mn40/pointnet2.pth',
            'pointconv': 'pretrain/remesh_mn40/pointconv.pth',
            'dgcnn': 'pretrain/remesh_mn40/dgcnn.pth',
        },
    },
    # trained on mn40 + ONet optimized mn40
    'opt_mn40': {
        1024: {
            'pointnet': 'pretrain/opt_mn40/pointnet.pth',
            'pointnet2': 'pretrain/opt_mn40/pointnet2.pth',
            'pointconv': 'pretrain/opt_mn40/pointconv.pth',
            'dgcnn': 'pretrain/opt_mn40/dgcnn.pth',
        },
    },
    # trained on mn40 + ConvONet optimized mn40
    'conv_opt_mn40': {
        1024: {
            'pointnet': 'pretrain/conv_opt_mn40/pointnet.pth',
            'pointnet2': 'pretrain/conv_opt_mn40/pointnet2.pth',
            'pointconv': 'pretrain/conv_opt_mn40/pointconv.pth',
            'dgcnn': 'pretrain/conv_opt_mn40/dgcnn.pth',
        },
    },
}

# PU-Net trained on Visionair with 1024 input point number, up rate 4
PU_NET_WEIGHT = 'defense/DUP_Net/pu-in_1024-up_4.pth'

# Note: the following batch sizes are tested on a RTX 2080 Ti GPU
# you may need to slightly adjust them to fit in your device

# max batch size used in testing model accuracy
MAX_TEST_BATCH = {
    1024: {
        'pointnet': 512,
        'pointnet2': 256,
        'dgcnn': 96,
        'pointconv': 320,
    },
}

# max batch size used in testing model accuracy with DUP-Net defense
# since there will be 4x points in DUP-Net defense results
MAX_DUP_TEST_BATCH = {
    1024: {
        'pointnet': 160,
        'pointnet2': 80,
        'dgcnn': 26,
        'pointconv': 48,
    },
}

# max batch size used in Perturb attack
MAX_PERTURB_BATCH = {
    1024: {
        'pointnet': 384,
        'pointnet2': 78,
        'dgcnn': 52,
        'pointconv': 57,
    },
}

# max batch size used in kNN attack
MAX_KNN_BATCH = {
    1024: {
        'pointnet': 248,
        'pointnet2': 74,
        'dgcnn': 42,
        'pointconv': 54,
    },
}

# max batch size used in Add attack
MAX_ADD_BATCH = {
    1024: {
        'pointnet': 256,
        'pointnet2': 78,
        'dgcnn': 35,
        'pointconv': 57,
    },
}

# max batch size used in Add Cluster attack
MAX_ADD_CLUSTER_BATCH = {
    1024: {
        'pointnet': 320,
        'pointnet2': 88,
        'dgcnn': 45,
        'pointconv': 60,
    },
}

# max batch size used in Add Object attack
MAX_ADD_OBJECT_BATCH = {
    1024: {
        'pointnet': 320,
        'pointnet2': 88,
        'dgcnn': 42,
        'pointconv': 58,
    },
}

# max batch size used in Drop attack
MAX_DROP_BATCH = {
    1024: {
        'pointnet': 360,
        'pointnet2': 80,
        'dgcnn': 52,
        'pointconv': 57,
    },
}

MAX_FGM_PERTURB_BATCH = {
    1024: {
        'pointnet': 360,
        'pointnet2': 76,
        'dgcnn': 52,
        'pointconv': 58,
    },
}
