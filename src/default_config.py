
def get_config():
    return  {
            # The using GPU ID
            'cuda_id': 0,
            # whether you use comet-ml for visualizing the training procedure
            'comet': False,
            # The Running mode ('train' or 'test')
            'mode': 'train',
            # fine-tuning
            'finetune': False,
            # The state path which you want to resume the training
            'resume': False,
            # The interval step for viaulizing the output images
            'vis_interval': 5000,
            # The interval step for saving the model
            'save_model_interval': 50000,
            # The interval step for printing the losses to standard output line
            'log_interval': 100,

            # training parameters
            'max_iter': 1000000,
            'batch_size': 4,
            # UNet layer size
            'layer_size': 6,
            # the loss coefficients
            'loss_coef': {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'perc': 0.05, 'style': 120.0},
            # Optimization Setting
            'optim': 'Adam',
            'initial_lr': 2e-4,
            'finetune_lr': 5e-5,
            'momentum': 0,
            'weight_decay': 0,

            # The directory settings
            'data_root': 'data',
            'ckpt': 'ckpt',

            # the information of comet-ml
            'api_key': '',
            'project_name': '',
            'workspace': ''
            }
