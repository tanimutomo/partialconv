import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import save_ckpt, to_items
from .evaluate import evaluate


class Trainer:
    def __init__(self, step, config, device, model, dataset_train,
                 dataset_val, criterion, optimizer, experiment):
        self.stepped = step
        self.config = config
        self.device = device
        self.model = model
        self.dataloader_train = DataLoader(dataset_train,
                                           batch_size=config.batch_size,
                                           shuffle=True)
        self.dataset_val = dataset_val
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluate = evaluate
        self.experiment = experiment
        
    def iterate(self, num_iter):
        print('Start the training')
        for step, (input, mask, gt) in enumerate(self.dataloader_train):
            loss_dict = self.train(step+self.stepped, input, mask, gt)
            # report the loss
            if step % self.config.log_interval == 0:
                self.report(step+self.stepped, loss_dict)

            # evaluation
            if (step+self.stepped + 1) % self.config.vis_interval == 0:
                # set the model to evaluation mode
                self.model.eval()
                self.evaluate(self.model, self.dataset_val, self.device,
                              '{}/train_out/test_{}.png'.format(self.config.ckpt, step+self.stepped),
                              self.experiment)

            # save the model
            if (step+self.stepped + 1) % self.config.save_model_interval == 0 \
                    or (step + 1) == self.config.max_iter:
                save_ckpt('{}/models/{}.pth'.format(self.config.ckpt, step+self.stepped + 1),
                          [('model', self.model)],
                          [('optimizer', self.optimizer)],
                          step+self.stepped + 1)

            if step >= self.config.max_iter:
                break

    def train(self, step, input, mask, gt):
        # set the model to training mode
        self.model.train()

        # send the input tensors to cuda
        input = input.to(self.device)
        mask = mask.to(self.device)
        gt = gt.to(self.device)

        # model forward
        output, _ = self.model(input, mask)
        loss_dict = self.criterion(input, mask, output, gt)
        loss = 0.0
        for key, coef in self.config.loss_coef.items():
            value = coef * loss_dict[key]
            loss += value

        # updates the model's params
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict['total'] = loss
        return to_item(loss_dict)
        
    def report(self, step, loss_dict):
        self.experiment.log_metrics(loss_dict, step)
        print('STEP:', step,
              ' / Valid Loss:', loss_dict['valid'],
              ' / Hole Loss:', loss_dict['hole'],
              ' / TV Loss:', loss_dict['tv'],
              ' / Perc Loss:', loss_dict['perc'],
              ' / Style Loss:', loss_dict['style'],
              ' / TOTAL LOSS:', loss_dict['total'])


