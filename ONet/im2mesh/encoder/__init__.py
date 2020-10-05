from im2mesh.encoder import pointnet

encoder_dict = {
    'pointnet_simple': pointnet.SimplePointnet,
    'pointnet_resnet': pointnet.ResnetPointnet,  # onet uses this
}
