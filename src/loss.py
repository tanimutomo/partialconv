import torch
import torch.nn as nn


def tvloss(input):
    return torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
           torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, :, 1:]))


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super(InpantingLoss, self).__init__()
        self.l1 = nn.L1Loss()
        # default extractor is VGG16
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        pass


