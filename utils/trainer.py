import sys

from dataset.dataset import ArealDataset
from utils.dataset_builder import Builder
from utils.losses import ArealLoss
from utils.optimizer import ArealOptim
from utils.evaluation import Metrics


class Trainer:
    def __init__(self, cfg, model):
        super(Trainer, self).__init__()
        self.dataset = Builder(dataset_root=cfg.dataset_path,
                               batch_size=cfg.batch_size,
                               train_test_split_p=cfg.dataset['train_test_split'],
                               workers=cfg.workers)

        self.epochs = cfg.epochs
        self.model = model
        self.train_dataloader = self.dataset.train_dataloader
        self.test_dataloader = self.dataset.test_dataloader
        self.criterion = ArealLoss(num_classes=cfg.num_classes)
        self.optimizer_builder = ArealOptim(cfg, model.parameters())
        self.optimizer = self.optimizer_builder.net_optimizer
        self.lr_scheduler = self.optimizer_builder.lr_scheduler
        self.metrics = Metrics()


    def run_train(self):
        self.model.train()
        for epoch in range(self.epochs):
            self.train_step()

            if epoch % 5 == 0 and epoch > 0:
                self.lr_scheduler.step()

    def train_step(self):
        batch_loss_list = []
        auc_list = []
        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(loss)
            batch_loss_list.append(loss.cpu().detach().numpy())
            auc_list.append(self.metrics.accuracy(y, y_pred).numpy())
