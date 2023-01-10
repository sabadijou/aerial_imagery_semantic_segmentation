from torch.optim import Adam, SGD, lr_scheduler


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

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             step_size=cfg.scheduler['step_size'],
                                             gamma=cfg.scheduler['gamma'])

    def get_optim(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler

    net_optimizer = property(get_optim)
    lr_scheduler = property(get_scheduler)