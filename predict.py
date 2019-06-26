import torch
from torchvision import transforms
from PIL import Image

from src.model import PConvUNet
from src.utils import Config, load_ckpt, create_ckpt_dir

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
output_comp = mask * image + (1 - mask) * output

