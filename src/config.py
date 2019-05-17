
def get_config():
    return  {
            # training parameters
            'epoch': 100
            'batch_size': 4,

            # network configurations
            'layer_size': 6,

            # data normalization parameters
            'MEAN': [0.485, 0.456, 0.406],
            'STD': [0.229, 0.224, 0.225],

            # the loss coefficients
            'loss_coef': {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'perc': 0.05, 'style': 120.0}

            }
