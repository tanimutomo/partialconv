from comet_ml import Experiment

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from src.config import get_config
from src.dataset import Places2
from src.model import PConvUNet
from src.train import train
from src.utils import Config, init_xavier
from src import const


# the information of comet-ml
API_KEY = 
PROJECT_NAME = 
WORKSPACE =

# data root path
IMG_ROOT = './data'
MASK_ROOT = './data/mask'

# set the config
config = Config(get_config())

# Define the used device
device = torch.device('cuda:{}'.format(config.cuda_num)
                      if torch.cuda.is_available() else 'cpu')

# Define the model
model = PConvUNet(layer_size=config.layer_size).to(device)

# Data Transformation
img_tf = transforms.ToTensor()
mask_tf = transforms.ToTensor()

# Define the Validation set
dataset_val = Places2(IMG_ROOT,
                      MASK_ROOT,
                      img_tf,
                      mask_tf,
                      data='val').to(device)

# Set the configuration for training
if config.mode == 'train':
    # set the comet-ml
    if config.comet:
        experiment = Experiment(api_key=API_KEY,
                                project_name=PROJECT_NAME,
                                workspace=WORKSPACE)
        experiment.log_parameters(config)
    else:
        experiment = None

    # Define the Places2 Dataset and Data Loader
    dataset_train = Places2(IMG_ROOT,
                            MASK_ROOT,
                            img_tf,
                            mask_tf,
                            data='train').to(device)

    # Define the model initializer
    model.apply(init_xavier)

    # Define the Loss fucntion
    criterion = InpaintingLoss()
    # Define the Optimizer
    if config.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.lr,
                                     weight_decay=config.weight_decay)
    elif config.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)

    trainer = Trainer(config, device, model, dataloader_train,
                      dataset_val, criterion, optimizer, experiment=experiment)
    if config.comet:
        with experiment.train()
            trainer.iterate(config.num_iter)
    else:
        trainer.iterate(config.num_iter)

# Set the configuration for testing
elif config.mode == 'test':
    # <model load the trained weights>
    evaluate(model, dataset_val)

