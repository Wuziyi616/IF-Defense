from torch import nn
from src.encoder import encoder_dict
from src.conv_onet import models
from src.conv_onet import generation
from src.common import decide_total_volume_range, update_reso


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']

    # for pointcloud_crop
    try:
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except KeyError:
        pass
    # local positional encoding
    if 'local_coord' in cfg['model'].keys():
        encoder_kwargs['local_coord'] = cfg['model']['local_coord']
        decoder_kwargs['local_coord'] = cfg['model']['local_coord']
    if 'pos_encoding' in cfg['model']:
        encoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']

    # update the feature volume/plane resolution
    if cfg['data']['input_type'] == 'pointcloud_crop':
        fea_type = cfg['model']['encoder_kwargs']['plane_type']
        if (dataset.split == 'train') or (cfg['generation']['sliding_window']):
            recep_field = 2**(cfg['model']['encoder_kwargs']
                              ['unet3d_kwargs']['num_levels'] + 2)
            reso = cfg['data']['query_vol_size'] + recep_field - 1
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = update_reso(
                    reso, dataset.depth)
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = update_reso(
                    reso, dataset.depth)
        # if dataset.split == 'val': #TODO run validation in room level during training
        else:
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = dataset.total_reso
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = dataset.total_reso

    decoder = models.decoder_dict[decoder](
        dim=dim, c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    model = models.ConvolutionalOccupancyNetwork(
        decoder, encoder, device=device
    )

    return model


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    if cfg['data']['input_type'] == 'pointcloud_crop':
        # calculate the volume boundary
        query_vol_metric = cfg['data']['padding'] + 1
        unit_size = cfg['data']['unit_size']
        recep_field = 2**(cfg['model']['encoder_kwargs']
                          ['unet3d_kwargs']['num_levels'] + 2)
        if 'unet' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet_kwargs']['depth']
        elif 'unet3d' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels']

        vol_info = decide_total_volume_range(
            query_vol_metric, recep_field, unit_size, depth)

        grid_reso = cfg['data']['query_vol_size'] + recep_field - 1
        grid_reso = update_reso(grid_reso, depth)
        query_vol_size = cfg['data']['query_vol_size'] * unit_size
        input_vol_size = grid_reso * unit_size
        # only for the sliding window case
        vol_bound = None
        if cfg['generation']['sliding_window']:
            vol_bound = {'query_crop_size': query_vol_size,
                         'input_crop_size': input_vol_size,
                         'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                         'reso': grid_reso}

    else:
        vol_bound = None
        vol_info = None

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type=cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info=vol_info,
        vol_bound=vol_bound,
    )
    return generator
