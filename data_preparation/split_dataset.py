import numpy as np
from torch.utils.data import random_split

# Split test set into validation and test sets
class SplitDataset:
    def __init__(self,  val_fraction=0.2, seed=42):
        self.val_fraction = val_fraction
        self.seed = seed

    def split(self, train_x, train_y):
        assert len(train_x) == len(train_y), "train_x and train_y must have same length"

        n = len(train_x)
        indices = np.arange(n)

        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)

        val_size = int(n * self.val_fraction)

        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        val_x = train_x[val_idx]
        val_y = train_y[val_idx]

        train_x = train_x[train_idx]
        train_y = train_y[train_idx]

        return train_x, train_y, val_x, val_y