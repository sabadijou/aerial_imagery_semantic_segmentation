import torch
from dataset.dataset import ArealDataset


class Builder:
    def __init__(self, dataset_root, batch_size=4, train_test_split_p=0.1, workers=1):
        super(Builder, self).__init__()
        self.dataset = ArealDataset(root=dataset_root, training=True)
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset,
                                                                               [len(self.dataset)-int(train_test_split_p *
                                                                                                      len(self.dataset)),
                                                                                int(train_test_split_p * len(self.dataset))],
                                                                               generator=torch.Generator().manual_seed(101))
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=True)

        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False)

    def get_test_dataloader(self):
        return self.test_dataloader

    def get_train_dataloader(self):
        return self.train_dataloader

    train_loader = property(get_train_dataloader)
    test_loader = property(get_test_dataloader)