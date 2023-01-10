
import torch
import torch.nn as nn
import torch.nn.functional as F


class JaccardLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(JaccardLoss, self).__init__ ()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, self.classes, h, w).to(tensor.device).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, inputs, target):
        N = inputs.size ()[0]
        inputs = F.softmax(inputs, dim=1)
        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        inter = inter.view(N, self.classes, -1).sum(2)
        union = inputs + target_oneHot - (inputs * target_oneHot)
        union = union.view(N, self.classes, -1).sum(2)
        loss = inter / union
        return 1 - loss.mean()


class ArealLoss(nn.Module):
    def __init__(self, cfg, num_classes=6):
        super(ArealLoss, self).__init__()
        weights = torch.ones(num_classes)
        weights = weights.to(cfg.device)
        self.criterion_1 = torch.nn.CrossEntropyLoss(weight=weights)
        self.criterion_2 = JaccardLoss(n_classes=num_classes)

    def forward(self, x, y):
        loss_1 = self.criterion_1(x, self.criterion_2.to_one_hot(y))
        loss_2 = self.criterion_2(x, y)
        return loss_1 + loss_2

