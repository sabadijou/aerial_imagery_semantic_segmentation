
from torch.utils.tensorboard import SummaryWriter
from utils.dataset_builder import Builder
from utils.optimizer import ArealOptim
from utils.evaluation import Metrics
from utils.losses import ArealLoss
import numpy as np
import torch
import sys
import os

writer = SummaryWriter(comment='Metrics')


class Trainer:
    def __init__(self, cfg, model):
        super(Trainer, self).__init__()
        self.dataset = Builder(dataset_root=cfg.dataset_path,
                               batch_size=cfg.batch_size,
                               train_test_split_p=cfg.dataset['train_test_split'],
                               workers=cfg.workers)
        self.cfg = cfg
        self.epochs = cfg.epochs
        self.model = model
        self.train_dataloader = self.dataset.train_dataloader
        self.test_dataloader = self.dataset.test_dataloader
        self.criterion = ArealLoss(cfg, num_classes=cfg.num_classes)
        self.optimizer_builder = ArealOptim(cfg, model.parameters())
        self.optimizer = self.optimizer_builder.net_optimizer
        self.lr_scheduler = self.optimizer_builder.lr_scheduler
        self.metrics = Metrics()
        self.best_auc = 0
        self.total_batch_counter = 0

    def run_train(self):
        for epoch in range(self.epochs):
            self.cfg.current_epoch = epoch
            self.model.train()
            batch_loss_list_ = self.train_step()
            val_loss_list_ = self.evaluation()
            if epoch % 5 == 0 and epoch > 0:
                self.lr_scheduler.step()
            writer.add_scalar('Lr/Epoch', self.optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Train Loss/Epoch', np.mean(batch_loss_list_), epoch)
            writer.add_scalar('Validation Loss/Epoch', np.mean(val_loss_list_), epoch)

    def train_step(self):
        batch_loss_list = []
        auc_list = []
        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_loss_list.append(loss.cpu().detach().numpy())
            auc_list.append(self.metrics.accuracy(y, y_pred).numpy())
            writer.add_scalar('Loss/Batch', loss, self.total_batch_counter)
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)] [Learning Rate: %f]"
                % (
                    self.cfg.current_epoch,
                    self.cfg.epochs,
                    batch_idx + 1,
                    len(self.train_dataloader),
                    loss.cpu().detach().numpy(),
                    np.mean(auc_list),
                    self.optimizer.param_groups[0]['lr']
                )
            )
            self.total_batch_counter += 1
        return batch_loss_list

    def evaluation(self):
        self.model.eval()
        val_loss_list = []
        val_acc_list = []
        for batch_idx, (x, y) in enumerate(self.test_dataloader):
            with torch.no_grad():
                y_pred = self.model(x)
            val_loss = self.criterion(y_pred, y)
            val_loss_list.append(val_loss.item())
            val_acc_list.append(self.metrics.accuracy(y, y_pred).item())
        print('Validation loss : {:.5f} - Validation Accuracy : {:.2f}'.format(np.mean(val_loss_list),
                                                                               np.mean(val_acc_list)))

        if self.best_auc < np.mean(val_acc_list):
            self.best_auc = np.mean(val_acc_list)
            self.save_checkpoint()
        return val_loss_list

    def save_checkpoint(self):
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(self.model.state_dict(), 'checkpoints/best.pth')
