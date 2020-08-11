import os
import sys
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse
import torch
import numpy as np

from lib.helpers.data_builder import build_dataloader
from lib.helpers.model_builder import build_model
from lib.helpers.optimizer_builder import build_optimizer
from lib.helpers.scheduler_builder import build_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.kitti_helper import Kitti_Config


parser = argparse.ArgumentParser(description='implementation of PyTorch 3D Object Detection')
parser.add_argument('-e', '--evaluation', dest='evaluation', action='store_true', help='evaluate model on validation set')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    log_file = 'train.log'
    logger = create_logger(log_file)

    # load cfg
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    set_random_seed(cfg['random_seed'])

    # build dataset helper, temporally only support kitti dataset
    dataset_helper = Kitti_Config()

    #  build dataset
    if args.evaluation:
        cfg['dataset']['train']['enable'] = False         # avoid loading trainset when testing
        cfg['dataset']['val'] = cfg['dataset']['test']    # using test data to replace val data
        cfg['dataset']['val']['enable'] = True            # make sure the enable tag is 'True'

    train_loader, test_loader = build_dataloader(cfg['dataset'], dataset_helper, logger)

    # build model
    model = build_model(cfg['model'], dataset_helper, logger)

    if args.evaluation:
        tester = Tester(cfg['tester'], model, test_loader, logger)
        tester.test()
        return

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model, logger)

    # build scheduler
    last_epoch = -1
    cfg['scheduler']['lr_scheduler'].update({'lr': cfg['optimizer']['lr'], 'last_epoch': last_epoch})
    cfg['scheduler']['bnm_scheduler'].update({'last_epoch': last_epoch})
    lr_scheduler, bnm_scheduler = build_scheduler(cfg['scheduler'], optimizer, model, logger)

    trainer = Trainer(cfg['trainer'],
                      model,
                      optimizer,
                      train_loader,
                      test_loader,
                      lr_scheduler,
                      bnm_scheduler,
                      logger)
    trainer.train()


if __name__ == '__main__':
    main()
