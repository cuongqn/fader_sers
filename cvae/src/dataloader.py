from torch.utils.data import Dataset, DataLoader
import torch


class SpectraDataset(Dataset):
    def __init__(self, x, y=None, return_form='xy'):
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
        return self.x.size(0)

    def __getitem__(self, idx):
        # return_in = [getattr(self, i)[idx] for i in self.return_in]
        # return_in = return_in[0] if len(return_in) == 1 else return_in
        # return_out = [getattr(self, o)[idx] for o in self.return_out]
        # return_out = return_out[0] if len(return_out) == 1 else return_out

        # return return_in, return_out

        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]


def create_loader(x, y, batch_size, shuffle, return_form='x_x',num_workers=0):
    dataset = SpectraDataset(x, y)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader

