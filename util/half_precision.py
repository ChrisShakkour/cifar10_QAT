import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class HalfPrecisionDataLoader(DataLoader):
    """
    Wraps a DataLoader and converts all floating-point tensors in each batch to half precision.
    """
    def __iter__(self):
        for batch in super().__iter__():
            if isinstance(batch, (tuple, list)):
                yield tuple(
                    b.half() if torch.is_tensor(b) and b.dtype.is_floating_point else b
                    for b in batch
                )
            elif torch.is_tensor(batch) and batch.dtype.is_floating_point:
                yield batch.half()
            else:
                yield batch