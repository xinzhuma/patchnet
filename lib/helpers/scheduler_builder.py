import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched


def build_scheduler(cfg_sheduler, optimizer, model, logger):
    lr_scheduler_cfg = cfg_sheduler['lr_scheduler']
    lr_scheduler = build_lr_scheduler(optimizer=optimizer,
                                      lr=lr_scheduler_cfg['lr'],
                                      lr_clip=lr_scheduler_cfg['clip'],
                                      lr_decay_list=lr_scheduler_cfg['decay_list'],
                                      lr_decay_rate=lr_scheduler_cfg['decay_rate'],
                                      last_epoch=lr_scheduler_cfg['last_epoch'])
    bnm_scheduler = None
    bnm_scheduler_cfg = cfg_sheduler['bnm_scheduler']
    if bnm_scheduler_cfg['enable']:
        bnm_scheduler = build_bnm_scheduler(model=model,
                                            bnm=bnm_scheduler_cfg['momentum'],
                                            bnm_clip=bnm_scheduler_cfg['clip'],
                                            bnm_decay_list=bnm_scheduler_cfg['decay_list'],
                                            bnm_decay_rate=bnm_scheduler_cfg['decay_rate'],
                                            last_epoch=bnm_scheduler_cfg['last_epoch'])

    return lr_scheduler, bnm_scheduler



def build_lr_scheduler(optimizer, lr, lr_clip, lr_decay_list, lr_decay_rate, last_epoch):

    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in lr_decay_list:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * lr_decay_rate
        return max(cur_decay,  lr_clip / lr)

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    return lr_scheduler


def build_bnm_scheduler(model, bnm, bnm_clip, bnm_decay_list, bnm_decay_rate, last_epoch):

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in bnm_decay_list:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * bnm_decay_rate
        return max(bnm*cur_decay, bnm_clip)

    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return bnm_scheduler


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))