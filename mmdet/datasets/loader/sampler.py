from __future__ import division
import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler

from .curriculum_func import CurriculumFunc


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class DistributedRepeatedRandomSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 sampling_cfg,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 shuffle=True):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # override num_samples and total_size so samples in all batches are
        # divisible by samples_per_gpu.
        self.num_samples = int(
            math.ceil(
                len(self.dataset) * 1.0 / samples_per_gpu /
                self.num_replicas)) * samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

        # repeated sampling config
        self.thres = sampling_cfg['thres']
        self.max_epochs = sampling_cfg['max_epochs']
        self.curriculum_func = CurriculumFunc(sampling_cfg['func'])
        self.repeat_factors = self._compute_repeat_factors()

    def __iter__(self):
        if self.shuffle:
            # after impelemntation, check all process have same indices
            indices = self._get_epoch_indices()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def _compute_repeat_factors(self):
        # 1. compute category repeat factors
        cats = self.dataset.lvis.load_cats(self.dataset.cat_ids)
        cat_freqs = np.zeros(len(cats))
        for i, cat in enumerate(cats):
            cat_freqs[i] = cat['image_count'] / len(self.dataset.img_ids)
        cat_repeat_factors = np.maximum(1, np.sqrt(self.thres / cat_freqs))

        # 2. compute image repeat factors
        img_repeat_factors = []
        for i, img_info in enumerate(self.dataset.img_infos):
            img_id = img_info['id']
            ann_ids = self.dataset.lvis.get_ann_ids(img_ids=[img_id])
            anns = self.dataset.lvis.load_anns(ids=ann_ids)

            cats_in_img = {ann['category_id'] for ann in anns}
            cat_repeat_factors_in_img = []
            for unique_cat_id in list(cats_in_img):
                cat_repeat_factors_in_img.append(
                    cat_repeat_factors[unique_cat_id - 1])
            img_repeat_factors.append(np.max(cat_repeat_factors_in_img))

        return torch.tensor(img_repeat_factors, dtype=torch.float32)

    def _get_epoch_indices(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # apply curriculum on repeat factors
        phase = self.epoch / (self.max_epochs - 1)
        alpha = self.curriculum_func(phase)
        rep_factors = (1 - alpha) + alpha * self.repeat_factors.clone()

        # stochastic rounding on repeat factors so that repeat factors slightly
        # differ for every epoch.
        rep_int_part = torch.trunc(rep_factors)
        rep_frac_part = rep_factors - rep_int_part
        rands = torch.rand(len(rep_frac_part), generator=g)
        stochastic_rep_factors = rep_int_part + (rands < rep_frac_part).float()

        # Construct a list of indices in which we repeat images as specified
        img_indices = []
        for dataset_index, rep_factor in enumerate(stochastic_rep_factors):
            img_indices.extend([dataset_index] * int(rep_factor.item()))

        # image index list for every epoch reflected repeatation
        rand_indices = torch.randperm(
            len(img_indices), generator=g).tolist()[:len(self.dataset)]
        indices = np.asarray(img_indices)[rand_indices].tolist()

        if self.rank == 0:
            log_str = 'Epoch: {}/{}, '.format(self.epoch + 1, self.max_epochs)
            log_str += 'total copied indices: {}, net indices: {}, '.format(
                len(img_indices), len(set(indices)))
            log_str += 'len(dataset): {}'.format(len(self.dataset))
            print(log_str)
        return indices


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
