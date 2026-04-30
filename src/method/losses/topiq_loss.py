import torch
from torch import nn as nn
from basicsr.utils.registry import LOSS_REGISTRY
import pyiqa

@LOSS_REGISTRY.register()
class TopiqLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(TopiqLoss, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.topiq_loss_fn = pyiqa.create_metric('topiq_fr', device=device, as_loss=True)
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * self.topiq_loss_fn(pred, target), None
