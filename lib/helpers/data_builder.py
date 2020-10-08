import numpy as np

from torch.utils.data import DataLoader

from lib.datasets.frustum_dataset import FrustumDataset
from lib.datasets.patch_dataset import PatchDataset


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_dataloader(cfg, dataset_helper,logger):
    # ------------------  build frustum dataset -------------------
    if cfg['type'] == 'frustum':   # for FPointNet and Pseudo LiDAR
        train_loader = None
        if cfg['train']['enable']:
            train_dataset = FrustumDataset(dataset_helper=dataset_helper,
                                           npoints=cfg['npoints'],
                                           rotate_to_center=cfg['rotate_to_center'],
                                           random_flip=cfg['train']['random_flip'],
                                           random_shift=cfg['train']['random_flip'],
                                           pickle_file=cfg['train']['pickle_file'],
                                           logger=logger)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=cfg['batch_size'],
                                      num_workers=cfg['workers'],
                                      worker_init_fn=my_worker_init_fn,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
        test_loader = None
        if cfg['val']['enable']:
            test_dataset = FrustumDataset(dataset_helper=dataset_helper,
                                          npoints=cfg['npoints'],
                                          rotate_to_center=cfg['rotate_to_center'],
                                          random_flip=cfg['val']['random_flip'],
                                          random_shift=cfg['val']['random_flip'],
                                          pickle_file=cfg['val']['pickle_file'],
                                          from_rgb_detection=cfg['val']['from_rgb_detection'],
                                          logger=logger)
            test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=cfg['batch_size'],
                                      num_workers=cfg['workers'],
                                      worker_init_fn=my_worker_init_fn,
                                      shuffle=False,
                                      pin_memory=True,
                                      drop_last=False)

    # ------------------  build patch dataset -------------------
    elif cfg['type'] == 'patch':   # for PatchNet
        train_loader = None
        if cfg['train']['enable']:
            train_dataset = PatchDataset(dataset_helper=dataset_helper,
                                         pickle_file=cfg['train']['pickle_file'],
                                         patch_size=cfg['patch_size'],
                                         random_flip=cfg['train']['random_flip'],
                                         random_shift=cfg['train']['random_flip'],
                                         add_rgb=cfg['add_rgb'],
                                         rotate_to_center=['rotate_to_center'],
                                         logger=logger)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=cfg['batch_size'],
                                      num_workers=cfg['workers'],
                                      worker_init_fn=my_worker_init_fn,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
        testloader = None
        if cfg['val']['enable']:
            test_dataset = PatchDataset(dataset_helper=dataset_helper,
                                        pickle_file=cfg['val']['pickle_file'],
                                        patch_size=cfg['patch_size'],
                                        random_flip=cfg['val']['random_flip'],
                                        random_shift=cfg['val']['random_flip'],
                                        add_rgb=cfg['add_rgb'],
                                        rotate_to_center=['rotate_to_center'],
                                        from_rgb_detection=cfg['val']['from_rgb_detection'],
                                        logger=logger)
            test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=cfg['batch_size'],
                                      num_workers=cfg['workers'],
                                      worker_init_fn=my_worker_init_fn,
                                      shuffle=False,
                                      pin_memory=True,
                                      drop_last=False)


    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    return train_loader, test_loader
