from dataset.dataset import ArealDataset
from dataset_builder import Builder
from losses import ArealLoss
from optimizer import ArealOptim
from evaluation import Metrics


class Trainer:
    def __init__(self, cfg, model, ds_root, world_size=1, epochs=100,
                 batch_size=4, train_test_split=0.1, workers=1,
                 num_classes=6):
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
        self.criterion = ArealLoss(num_classes=num_classes)
        self.optimizer_builder = ArealOptim(cfg, model.parameters())
        self.optimizer = self.optimizer_builder.net_optimizer
        self.metrics = Metrics()

    def run_train(self):
        self.model.train()
        for epoch in range(self.epochs):
            loss_epoch = self.train_step()

    def train_step(self):
        batch_loss_list = []
        auc_list = []
        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            y_pred = self.model(x)
            loss = self.criterion(y, y_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_loss_list.append(loss.cpu().detach().numpy())
            auc_list.append(self.metrics.accuracy(y, y_pred).numpy())
