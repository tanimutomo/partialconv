import torch
import torch.nn as nn


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super(InpantingLoss, self).__init__()
        self.l1 = nn.L1Loss()
        # default extractor is VGG16
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        # Non-hole pixels directly set to ground truth
        comp = mask * input + (1 - mask) * output

        tv_loss = torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) \
                  + torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        hole_loss = l1((1 - mask) * (output - gt))
        valid_loss = l1(mask * (output - gt))
        perc_loss = 


        total_loss = valid_loss + 6*hole_loss + 0.05*perceptual_loss \
                     + 120*(style_out_loss+style_comp_loss) + 0.1*tv_loss

        return total_loss




