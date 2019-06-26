import argparse
import os
from PIL import Image
import torch
from torchvision import transforms

from src.model import PConvUNet

parser = argparse.ArgumentParser(description="Specify the inputs")
parser.add_argument('--img', type=str, default="examples/img0.jpg")
parser.add_argument('--mask', type=str, default="examples/mask0.png")
parser.add_argument('--model', type=str, default="pretrained_pconv.pth")
args = parser.parse_args()

# Define the used device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
print("Loading the Model...")
model = PConvUNet(finetune=False, layer_size=7)
model.load_state_dict(torch.load(args.model, map_location=device)['model'])
model.to(device)

# Data Transformation
img_tf = transforms.Compose([
            transforms.ToTensor()
            ])
mask_tf = transforms.Compose([
            transforms.ToTensor()
            ])

print("Loading the inputs...")
img = Image.open(args.img)
img = img_tf(img.convert('RGB'))
mask = Image.open(args.mask)
mask = mask_tf(mask.convert('RGB'))
gt = img
img = img * mask


# model prediction
print("Model Prediction...")
model.eval()
with torch.no_grad():
    output, _ = model(img.unsqueeze(0).to(device),
                      mask.unsqueeze(0).to(device))
output = output.to(torch.device('cpu')).squeeze()
output_comp = mask * img + (1 - mask) * output

# save the output image
print("Saving the output...")
to_pil = transforms.ToPILImage()
out = to_pil(output_comp)
img_name = args.img.split('/')[-1]
out.save(os.path.join("examples", "out_{}".format(img_name)))
