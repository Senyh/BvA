import torch
import torch.nn as nn
from models.vnet import VNetorg


#######################################################################################
'''
VNet
'''
########################################################################################
class VNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, normalization='batchnorm'):
        super(VNet, self).__init__()
        self.net = VNetorg(in_channels=in_channels, out_channels=out_channels, normalization=normalization)

    def forward(self, x, perturbation=False, eval=False):
        x = self.net(x, perturbation)
        return x

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn([2, 1, 96, 96, 96]).to(device)
    viseg3d = VNet(in_channels=1, out_channels=4).to(device)
    vnet = VNet(in_channels=1, out_channels=4).to(device)
    result = viseg3d(tensor)
    print('#parameters:', sum(param.numel() for param in viseg3d.parameters()), sum(param.numel() for param in vnet.parameters()))
    print('output: ', result['out'].shape, result['rec'].shape, result['noise'].shape)