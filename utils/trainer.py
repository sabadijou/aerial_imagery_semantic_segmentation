from dataset.dataset import ArealDataset


class Trainer:
    def __init__(self, model, ds_root, world_size=1, epochs=100, batch_size=4):
        super(Trainer, self).__init__()
        self.dataset = ds_root
        self.world_size = world_size
        self.epochs = epochs
        self.model = model

    def run_train(self):
        self.model.train()
        for epoch in range(self.epochs):
            loss_epoch = self.train_step()

    def train_step(self):
        for batch_idx, (x, y) in enumerate(self.train_loader):
            pass