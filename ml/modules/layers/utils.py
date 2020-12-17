from torch import nn
from torch.nn import init


def init_weights(m, init_w='normal'):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if init_w == 'normal':
            m.weight.data.normal_(0, 1e-3)
        elif init_w == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if init_w == 'normal':
            m.weight.data.normal_(0, 1e-3)
        elif init_w == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
        if init_w == 'normal':
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif init_w == 'kaiming':
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)