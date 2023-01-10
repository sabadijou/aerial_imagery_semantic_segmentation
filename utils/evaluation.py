import torch


class Metrics:
    def __init__(self):
        super(Metrics, self).__init__()

    @staticmethod
    def accuracy(y, y_pred):
        acc = (y.cpu() == torch.argmax(y_pred, axis=1).cpu()).sum() / torch.numel(y.cpu())
        return acc
