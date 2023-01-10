from torch.optim import Adam, SGD


class ArealOptim:
    def __init__(self, cfg, parameters):
        super(ArealOptim, self).__init__()
        if cfg.optimizer['type'] == 'Adam':
            self.optimizer = Adam(parameters,
                                  lr=cfg.optimizer['lr'],
                                  weight_decay=cfg.optimizer['weight_decay'])
        elif cfg.optimizer['type'] == 'SGD':
            self.optimizer = SGD(parameters,
                                 lr=cfg.optimizer['lr'],
                                 momentum=cfg.optimizer['momentum'],
                                 weight_decay=cfg.optimizer['weight_decay'])

    def get_optim(self):
        return self.optimizer

    net_optimizer = property(get_optim)