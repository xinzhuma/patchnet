from lib.models.fpointnet import FPointNet
from lib.models.patchnet import PatchNet


def build_model(cfg_model, dataset_helper, logger):
    assert cfg_model['name'] in ['fpointnet', 'patchnet']
    if cfg_model['name'] =='fpointnet':
        return FPointNet(cfg=cfg_model,
                         num_heading_bin=dataset_helper.num_heading_bin,
                         num_size_cluster=dataset_helper.num_size_cluster,
                         mean_size_arr=dataset_helper.mean_size_arr)
    elif cfg_model['name'] == 'patchnet':
        return PatchNet(cfg=cfg_model,
                        num_heading_bin=dataset_helper.num_heading_bin,
                        num_size_cluster=dataset_helper.num_size_cluster,
                        mean_size_arr=dataset_helper.mean_size_arr)
    else:
        raise NotImplementedError("%s model is not supported" % cfg_model['name'])
