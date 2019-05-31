from torch.utils.data import Dataset, DataLoader
import torch


class SpectraDataset(Dataset):
    def __init__(self, x, y=None):
        super(SpectraDataset, self).__init__()
        if y is None:
            assert 'y' not in return_form

        if isinstance(x, list):
            self.x = [torch.FloatTensor(x_) for x_ in x]
        else:
            self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y) if y is not None else y

        # self.return_in, self.return_out = return_form.split('_')

    def __len__(self):
        
        return self.x[0].size(0) if isinstance(self.x, list) else self.x.size(0)

    def __getitem__(self, idx):
        if isinstance(self.x, list):
            return_x = [x_[idx] for x_ in self.x]
        else:
            return_x = self.x[idx]
        if self.y is not None:
            return return_x, self.y[idx]
        else:
            return return_x


def create_loader(x, y, batch_size, shuffle, return_form='x_x',num_workers=0):
    dataset = SpectraDataset(x, y)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader

