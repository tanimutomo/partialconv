import os
from PIL import Image
import torch
from torchvision import transforms

from src.model import PConvUNet
from src.utils import create_ckpt_dir

# create the checkpoint directory
ckpt = create_ckpt_dir()

# Define the used device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
print("Loading the Model...")
MODEL_PATH = "pretrained_pconv.pth"
model = PConvUNet(finetune=False, layer_size=7)
model.load_state_dict(torch.load(MODEL_PATH)['model'])
model.to(device)

# Data Transformation
img_tf = transforms.Compose([
            transforms.ToTensor()
            ])
mask_tf = transforms.Compose([
            transforms.ToTensor()
            ])

IMAGE_PATH = "./example/img01.png"
MASK_PATH = "./example/mask01.png"
img = Image.open(IMAGE_PATH)
img = img_tf(img.convert('RGB'))
mask = Image.open(MASK_PATH)
mask = mask_tf(mask.convert('RGB'))
gt = img
img = img * mask


# model prediction
model.eval()
with torch.no_grad():
    output, _ = model(img.to(device), mask.to(device))
output = output.to(torch.device('cpu'))
output_comp = mask * img + (1 - mask) * output

# save the output image
out = output_comp.numpy()
to_pil = transforms.TOPILImage()
out = to_pil(out)
img_name = IMAGE_PATH.split('/')[-1]
out.save(os.path.join(ckpt, "out_{}".format(img_name)))
