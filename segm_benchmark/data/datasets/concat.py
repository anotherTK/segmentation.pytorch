import bisect
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

class ConcatDataset(_ConcatDataset):
    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx