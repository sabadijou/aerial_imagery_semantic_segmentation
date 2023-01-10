from dataset.dataset import ArealDataset
from dataset_builder import Builder


class Trainer:
    def __init__(self, model, ds_root, world_size=1, epochs=100,
                 batch_size=4, train_test_split=0.1, workers=1):
        super(Trainer, self).__init__()
        self.dataset = Builder(dataset_root=ds_root,
                               batch_size=batch_size,
                               train_test_split_p=train_test_split,
                               workers=workers)

        self.world_size = world_size
        self.epochs = epochs
        self.model = model
        self.train_dataloader = self.dataset.train_dataloader
        self.test_dataloader = self.dataset.test_dataloader

    def run_train(self):
        self.model.train()
        for epoch in range(self.epochs):
            loss_epoch = self.train_step()

    def train_step(self):
        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            pass