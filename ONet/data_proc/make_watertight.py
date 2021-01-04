"""Process to get watertight meshes.
Use Manifold: https://github.com/hjwdzh/Manifold
    to generate watertight meshes.
First download Manifold and build it, then specify the data root to
    ModelNet40 meshes and run this file.
"""

import os
import numpy as np
import trimesh
from tqdm import tqdm


def postprocess_mesh(mesh, num_faces=None):
    """Post processing mesh by removing small isolated pieces.

    Args:
        mesh (trimesh.Trimesh): input mesh to be processed
        num_faces (int, optional): min face num threshold. Defaults to 4096.
    """
    total_num_faces = len(mesh.faces)
    if num_faces is None:
        num_faces = total_num_faces // 100
    cc = trimesh.graph.connected_components(
        mesh.face_adjacency, min_len=3)
    mask = np.zeros(total_num_faces, dtype=np.bool)
    cc = np.concatenate([
        c for c in cc if len(c) > num_faces
    ], axis=0)
    mask[cc] = True
    mesh.update_faces(mask)
    return mesh


data_root = '/home/wzw/WZY/SUMMER/occ_net/data/MN40.build'
save_root = 'data/MN40_watertight'
interval = 5
count = 0
POST = True

if not os.path.exists(save_root):
    os.makedirs(save_root)
all_class = os.listdir(data_root)
all_class.sort()
for one_class in all_class:
    class_root = os.path.join(data_root, one_class, '0_in')
    all_file = os.listdir(class_root)
    all_file.sort()
    # save dir
    save_class_root = os.path.join(save_root, one_class)
    if not os.path.exists(save_class_root):
        os.makedirs(save_class_root)
    for one_file in tqdm(all_file):
        file_root = os.path.join(class_root, one_file)
        save_file_root = os.path.join(
            save_class_root, one_file.replace('off', 'obj'))
        # Manifold can only process obj file
        # so first convert off to obj
        mesh = trimesh.load(file_root)
        mesh.export(save_file_root)
        # Call Manifold!
        os.system('./build/manifold {} {}'.
                  format(save_file_root, save_file_root))
        # convert back to off format
        mesh = trimesh.load(save_file_root)
        # post-processing
        if POST:
            try:
                mesh = postprocess_mesh(mesh)
            except ValueError:
                pass
        # Manifold may fail to process some meshes
        # filter out non-watertight meshes
        if mesh.is_watertight:
            mesh.export(save_file_root.replace('obj', 'off'))
        else:  # simply discard them
            print('Not watertight!')
            count += 1
        os.remove(save_file_root)
print('Total {} not watertight'.format(count))
