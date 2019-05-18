import torch
import torch.nn as nn
from torchvision import models


class InpaintingLoss(nn.Module):
    def __init__(self, coef, extractor):
        super(InpaintingLoss, self).__init__()
        self.coef = coef
        self.l1 = nn.L1Loss()
        # default extractor is VGG16
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        # Non-hole pixels directly set to ground truth
        comp = mask * input + (1 - mask) * output

        # Total Vision Regularization
        tv_loss = torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) \
                  + torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))

        # Hole Pixel Loss
        hole_loss = self.l1((1-mask) * output, (1-mask) * gt)

        # Valid Pixel Loss
        valid_loss = self.l1(mask * output, mask * gt)

        # Perceptual Loss and Style Loss
        feats_out = self.extractor(output)
        feats_comp = self.extractor(comp)
        feats_gt = self.extractor(gt)
        perc_loss = 0.0
        style_loss = 0.0
        # Calculate the L1Loss for each feature map
        for i in range(3):
            perc_loss += self.l1(feats_out[i], feats_gt[i])
            perc_loss += self.l1(feats_comp[i], feats_gt[i])
            style_loss += self.l1(gram_matrix(feats_out[i]), gram_matrix(feats_gt[i]))
            style_loss += self.l1(gram_matrix(feats_comp[i]), gram_matrix(feats_gt[i]))

        total_loss = self.coef['valid'] * valid_loss \
                     + self.coef['hole'] * hole_loss \
                     + self.coef['perc'] * perc_loss \
                     + self.coef['style'] * style_loss \
                     + self.coef['tv'] * tv_loss
        return total_loss


# The network of extracting the feature for perceptual and style loss
class VGG16FeatureExtractor(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        normalization = Normalization(self.MEAN, self.STD)
        # Define the each feature exractor
        self.enc_1 = nn.Sequential(normalization, *vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{}'.format(i+1)).parameters():
                param.requires_grad = False

    def forward(self, input):
        feature_maps = [input]
        for i in range(3):
            feature_map = getattr(self, 'enc_{}'.format(i+1))(feature_maps[-1])
            feature_maps.append(feature_map)
        return feature_maps[1:]


# Normalization Layer for VGG
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# Calcurate the Gram Matrix of feature maps
def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


if __name__ == '__main__':
    from config import get_config
    config = get_config()
    vgg = VGG16FeatureExtractor()
    criterion = InpaintingLoss(config['loss_coef'], vgg)

    img = torch.randn(1, 3, 500, 500)
    mask = torch.ones((1, 1, 500, 500))
    mask[:, :, 250:, :][:, :, :, 250:] = 0
    input = img * mask
    out = torch.randn(1, 3, 500, 500)
    loss = criterion(input, mask, out, img)

