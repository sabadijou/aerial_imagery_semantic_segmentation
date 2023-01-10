import torch
from dataset.dataset import ArealDataset
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=4)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=1)

train_dataset, test_dataset =

class Builder:
    def __init__(self, dataset_root, batch_size=4, train_test_split_p=0.1):
        super(Builder, self).__init__()
        self.dataset  = ArealDataset(root=dataset_root, training=True)
        self.train_dataset, self.test_dataset =  torch.utils.data.random_split(self.dataset,
                                                                               [len(self.dataset)-int(train_test_split_p *
                                                                                                      len(self.dataset)),
                                                                                int(train_test_split_p * len(self.dataset))],
                                                                               generator=torch.Generator().manual_seed(101))
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            num_workers=4)