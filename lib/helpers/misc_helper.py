import torch.nn as nn


def init_weights(net, type='xavier'):
    assert type in ['kaiming', 'xavier', 'gaussian', 'uniform', 'none']
    if type == 'none': return
    for m in net.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            # init kernel weights
            if type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight)
            elif type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif type == 'gaussian':
                nn.init.normal_(m.weight, mean=0, std=1)
            elif type == 'uniform':
                nn.init.uniform_(m.weight, 0, 1)

            # init kernel bias
            if m.bias is not None:
                nn.init.zeros_(m.bias)

