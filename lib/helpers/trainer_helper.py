import os
import tqdm
import random

import torch
import numpy as np

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.decorator_helper import decorator



class Trainer(object):
    def __init__(self, cfg_trainer, model, optimizer, train_loader, test_loader,
                 lr_scheduler, bnm_scheduler, logger):
        self.cfg = cfg_trainer
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.bnm_scheduler = bnm_scheduler
        self.decorator = decorator
        self.logger = logger
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def train(self):
        for epoch in range(self.cfg['max_epoch']):
            # update lr & bnm
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.bnm_scheduler is not None:
                self.bnm_scheduler.step()

            # train one epoch
            self.logger.info('------ TRAIN EPOCH %03d ------' %(epoch + 1))
            self.logger.info('Learning Rate: %f' % self.lr_scheduler.get_lr()[0])
            self.logger.info('BN Momentum: %f' % (self.bnm_scheduler.lmbd(self.bnm_scheduler.last_epoch)))

            # reset numpy seed.
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            self.train_one_epoch()
            trained_epoch = epoch + 1

            if (trained_epoch % self.cfg['eval_frequency']) == 0:
                self.logger.info('------ EVAL EPOCH %03d ------' % (trained_epoch))
                self.eval_one_epoch()

            # save trained model
            if (trained_epoch % self.cfg['save_frequency']) == 0:
                os.makedirs('checkpoints', exist_ok=True)
                ckpt_name = os.path.join('checkpoints', 'checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, trained_epoch), ckpt_name, self.logger)

        return None


    def train_one_epoch(self):
        self.model.train()
        disp_dict = {}
        for batch_idx, batch_data in enumerate(self.train_loader):
            batch_data = [item.to(self.device) for item in batch_data]

            # train one batch
            self.optimizer.zero_grad()
            loss, stat_dict = self.decorator(self.model, batch_data, self.cfg['decorator'])
            loss.backward()
            self.optimizer.step()
            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in stat_dict.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                disp_dict[key] += stat_dict[key]

            # display statistics
            if trained_batch % self.cfg['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / self.cfg['disp_frequency']
                    log_str += ' %s:%.4f,' %(key, disp_dict[key])
                    disp_dict[key] = 0  # reset statistics
                self.logger.info(log_str)


    def eval_one_epoch(self):
        self.model.eval()
        disp_dict = {}  # collect statistics
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_loader):
                batch_data = [item.to(self.device) for item in batch_data]
                loss, stat_dict = self.decorator(self.model, batch_data, self.cfg['decorator'])

                for key in stat_dict.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += stat_dict[key]

                progress_bar.update()
            progress_bar.close()

            # display & log
            log_str = ''
            for key in sorted(disp_dict.keys()):
                disp_dict[key] /= len(self.test_loader)
                log_str += ' %s:%.4f,' %(key, disp_dict[key])
            self.logger.info(log_str)

