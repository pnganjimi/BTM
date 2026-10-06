"""Dataset loading utilities for eICU and MIMIC-III experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import os
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def loaders_eicu(path="../data/eicu/", train_batch=64, val_batch=64, test_batch=64, 
                 random_seed=42):

    # ----- Load pickled data -----
    with open(os.path.join(path, 'X_train.pkl'), "rb") as f:
        X = pickle.load(f)

    with open(os.path.join(path, 'y_train.pkl'), "rb") as f:
        y = pickle.load(f)

    # Convert to numpy arrays if needed
    X = np.array(X)
    y = np.array(y)

    # ----- First split: train (65%) vs temp (35%) -----
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.65, stratify=y, random_state=random_seed
    )

    # ----- Second split: validation (15%) vs test (20%) from temp -----
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.34, stratify=y_temp, random_state=random_seed
    ) 

    # Data has already been pre-processed

    # ----- Wrap into PyTorch datasets -----
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.float32))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch)
    test_loader = DataLoader(test_dataset, batch_size=test_batch)

    return train_loader, val_loader, test_loader


def get_loaders_mimic3(
        path="../data/mimic3/", train_batch=64, val_batch=64, test_batch=64, sampler=True,
        combine_val=False, pre_process="minmax",
        ds_half=0,):
    
    # The feature dimensions without duplications for mimic3
    mimic3_fea_dim_no_dup = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 16, 17, 
                                    19, 24, 25, 26,27, 28, 29, 30, 31, 32, 33, 34, 
                                    35, 36, 37, 39, 44, 40, 41] + list(range(49, 76)))

    # TRAINING set
    with open(os.path.join(path,'train_raw.p'), 'rb') as f:
        tr_x = pickle.load(f)

    tr_data, tr_lb = tr_x[0], tr_x[1]
    tr_data = tr_data[..., mimic3_fea_dim_no_dup]   # remove duplicated feature dimensions

    ### half train set exp
    assert ds_half in (0,1,2,)
    if ds_half:
        mid=len(tr_data)//2
        if ds_half==1:
            # use first half as the training data
            print("Using first half as the training set")
            tr_data, tr_lb = tr_data[0:mid], tr_lb[0:mid]
        else:
            print("Using second half as the training set")
            tr_data, tr_lb = tr_data[mid:], tr_lb[mid:]

    prpr = DataNormalization(tr_data, pre_process=pre_process)
    tr_data = prpr(tr_data)
    
    train_dataset= TensorDataset(torch.FloatTensor(tr_data),torch.LongTensor(tr_lb))
    
    if sampler:
        class_sample_count = np.array([len(np.where(tr_lb == t)[0]) for t in np.unique(tr_lb)])
        weight = 1. / class_sample_count
        # weight[1]=weight[1]*2
        samples_weight = np.array([weight[t] for t in tr_lb])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight)) 
        train_loader = DataLoader(train_dataset, batch_size=train_batch, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
    
    # VALIDATION set
    with open(os.path.join(path,'val_raw.p'), 'rb') as f:
        x= pickle.load(f)

    val_data, val_lb = x[0], x[1]
    val_data = val_data[..., mimic3_fea_dim_no_dup]  # remove duplicated feature dimensions
    val_data = prpr(val_data)

    val_dataset= TensorDataset(torch.FloatTensor(val_data),torch.LongTensor(val_lb))
    val_loader = DataLoader(val_dataset, batch_size=val_batch)

    # TEST set
    with open(os.path.join(path,'test_raw.p'), 'rb') as f:
        x= pickle.load(f)

    test = x["data"][0]
    test = test[..., mimic3_fea_dim_no_dup]  # remove duplicated feature dimensions

    test = prpr(test)
    test_labels = np.array(x["data"][1])

    test_dataset= TensorDataset(torch.FloatTensor(test),torch.LongTensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=test_batch)

    if combine_val:
        # combine validation with train set
        res_data=np.concatenate([tr_data, val_data], axis=0)
        res_lb=np.concatenate([tr_lb, val_lb], axis=0)
        return train_loader, val_loader, test_loader, res_data, res_lb, prpr
    else:
        return train_loader, val_loader, test_loader, tr_data, tr_lb, prpr


def get_loaders_mimic3_multilabel(
        path="../data/mimic3/pheno/",
        train_batch=64,
        val_batch=64,
        test_batch=64,
        sampler=True,
        combine_val=False,
        pre_process="minmax",
        ds_half=0,
        threshold=48,
):
    # Feature dimensions without duplicated features for MIMIC-III
    mimic3_fea_dim_no_dup = np.asarray(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 16, 17,
         19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
         35, 36, 37, 39, 44, 40, 41] + list(range(49, 76))
    )

    # ------------------
    # TRAINING SET
    # ------------------
    tr_data, tr_lb = load_split(path, 'train', mimic3_fea_dim_no_dup)
    tr_data, tr_lb = filter_and_truncate(tr_data, tr_lb, threshold)

    # Ensure labels are numpy array of shape [N, C]
    tr_lb = np.asarray(tr_lb)
    if tr_lb.ndim == 1:
        raise ValueError(
            "For multilabel training, labels must have shape [N, num_labels], "
            f"but got shape {tr_lb.shape}."
        )

    # Optional half-train experiments
    assert ds_half in (0, 1, 2)
    if ds_half:
        mid = len(tr_data) // 2
        if ds_half == 1:
            print("Using first half as the training set")
            tr_data, tr_lb = tr_data[:mid], tr_lb[:mid]
        else:
            print("Using second half as the training set")
            tr_data, tr_lb = tr_data[mid:], tr_lb[mid:]

    # Fit preprocessing only on train
    prpr = DataNormalization(tr_data, pre_process=pre_process)
    tr_data = prpr(tr_data)

    # Multilabel targets should be float
    train_dataset = TensorDataset(
        torch.FloatTensor(tr_data),
        torch.FloatTensor(tr_lb)
    )

    if sampler:
        # Multilabel weighted sampling:
        # class_counts[c] = number of positives for class c
        class_counts = tr_lb.sum(axis=0).astype(np.float64)

        # Avoid division by zero for classes with no positives
        class_weights = np.zeros_like(class_counts, dtype=np.float64)
        nonzero = class_counts > 0
        class_weights[nonzero] = 1.0 / class_counts[nonzero]

        # Per-sample weight:
        # sum weights of all positive labels for that sample
        samples_weight = (tr_lb * class_weights).sum(axis=1)

        # Give some small weight to all-negative samples if they exist
        all_negative = (tr_lb.sum(axis=1) == 0)
        if all_negative.any():
            neg_weight = 1.0 / all_negative.sum()
            samples_weight[all_negative] = np.maximum(samples_weight[all_negative], neg_weight)

        samples_weight = torch.as_tensor(samples_weight, dtype=torch.double)

        train_sampler = WeightedRandomSampler(
            weights=samples_weight,
            num_samples=len(samples_weight),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=train_batch, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)

    # ------------------
    # VALIDATION SET
    # ------------------
    val_data, val_lb = load_split(path, 'val', mimic3_fea_dim_no_dup)
    val_data, val_lb = filter_and_truncate(val_data, val_lb, threshold)

    val_data = prpr(val_data)

    if val_lb.ndim == 1:
        raise ValueError(
            "For multilabel validation, labels must have shape [N, num_labels], "
            f"but got shape {val_lb.shape}."
        )

    val_dataset = TensorDataset(
        torch.FloatTensor(val_data),
        torch.FloatTensor(val_lb)
    )
    val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=False)

    # ------------------
    # TEST SET
    # ------------------
    test_data, test_lb = load_split(path, 'test', mimic3_fea_dim_no_dup)
    test_data, test_lb = filter_and_truncate(test_data, test_lb, threshold)

    test_data = prpr(test_data)


    if test_lb.ndim == 1:
        raise ValueError(
            "For multilabel test, labels must have shape [N, num_labels], "
            f"but got shape {test_lb.shape}."
        )

    test_dataset = TensorDataset(
        torch.FloatTensor(test_data),
        torch.FloatTensor(test_lb)
    )
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False)

    num_classes = tr_lb.shape[1]

    if combine_val:
        res_data = np.concatenate([tr_data, val_data], axis=0)
        res_lb = np.concatenate([tr_lb, val_lb], axis=0)
        return train_loader, val_loader, test_loader, num_classes, res_data, res_lb, prpr
    else:
        return train_loader, val_loader, test_loader, num_classes, tr_data, tr_lb, prpr


class DataNormalization(object):
    # do normalization along each feature dimension
    def __init__(self, X_train, pre_process="std", eps=1e-9, prpr_ax=(0,1,)):
        # X_train is a N*T*F ndarray or a list of len N where X_train[i] has a shape T_i*F
        # pre_process: "minmax": min-max normalization, "std": z-normalization, "none": do nothing

        self.eps=eps # be added to denominator to avoid divided by zero

        assert pre_process in ("none", "std", "minmax"), \
            "Invalid pre_process: {}, needs to be one of: none/std/minmax".format(pre_process)

        self.pre_process = pre_process  # pre-processing method

        if self.pre_process!="none":

            # only compute statistical information when necessary
            if type(X_train) is list and len(X_train[0].shape)==2:
                # X_train is a list with len N, X_train[i] has a shape T_i*F
                X_source = np.concatenate(X_train)  # concatenate list into T_{ALL}*F
                ax = (0,)  # the axis to compute statistical information
            elif type(X_train) is np.ndarray and len(X_train.shape)==3:
                # X_train is a N*T*F matrix
                X_source = X_train
                ax = prpr_ax   # the axis to compute statistical information
            else:
                raise NotImplementedError("Normalization method not implemented!")

            self.mean = np.mean(X_source, axis=ax, keepdims=True)
            self.std = np.std(X_source, axis=ax, keepdims=True)+self.eps
            self.max = np.amax(X_source, axis=ax, keepdims=True)
            self.min = np.amin(X_source, axis=ax, keepdims=True)
            self.dif = (self.max-self.min)+self.eps

    def norm_std(self, data):
        if type(data) is list:
            norm_data = [(item - self.mean) / self.std for item in data]
        else:
            norm_data = (data - self.mean) / self.std
        return norm_data

    def norm_minmax(self, data):
        if type(data) is list:
            norm_data = [(item-self.min)/self.dif for item in data]
        else:
            norm_data = (data-self.min)/self.dif
        return norm_data

    def recover_minmax(self, data):
        if type(data) is list:
            rec_data = [(item*self.dif)+self.min for item in data]
        else:
            rec_data = (data * self.dif) + self.min
        return rec_data

    def recover_std(self, data):
        if type(data) is list:
            rec_data = [(item*self.std)+self.mean for item in data]
        else:
            rec_data = (data*self.std)+self.mean
        return rec_data

    def __call__(self, data):
        # data is a N*T*F ndarray or a list of len N where X_train[i] has a shape T_i*F
        if self.pre_process == "none":
            return data
        else:
            if self.pre_process == "minmax":
                # min-max normalization
                norm_data = self.norm_minmax(data)
            elif self.pre_process == "std":
                # standard normalization
                norm_data = self.norm_std(data)
            else:
                raise NotImplementedError("Pre-processing method: {} not implemented".format(self.pre_process))
            return norm_data

    def recover(self, data):
        # recover the data before normalization
        if self.pre_process == "none":
            return data
        else:
            if self.pre_process == "minmax":
                rec_data = self.recover_minmax(data)
            elif self.pre_process == "std":
                rec_data = self.recover_std(data)
            else:
                raise NotImplementedError("Pre-processing method: {} not implemented".format(self.pre_process))
            return rec_data


def load_split(data_path, split, feature_idx):
    with open(os.path.join(data_path, f"{split}_data_pheno.p"), "rb") as f:
        data = pickle.load(f)

    with open(os.path.join(data_path, f"{split}_labels_pheno.p"), "rb") as f:
        labels = pickle.load(f)

    data = [x[:, feature_idx] for x in data]
    labels = np.asarray(labels)
    
    return data, labels


def filter_and_truncate(data, labels, min_t=48):
    """
    data: list of arrays [T_i, F]
    labels: np.ndarray [N, C]

    returns:
        data_out: np.ndarray [N_new, min_t, F]
        labels_out: np.ndarray [N_new, C]
    """
    keep_idx = [i for i, x in enumerate(data) if x.shape[0] >= min_t]

    data_out = [data[i][:min_t] for i in keep_idx]
    labels_out = labels[keep_idx]

    data_out = np.stack(data_out)  # now safe: all are [48, F]
    return data_out, labels_out


@dataclass
class DatasetBundle:
    """Named container returned by :func:`load_dataset`."""
    name: str
    task: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_features: int
    output_dim: int
    train_data: Optional[np.ndarray] = None
    train_labels: Optional[np.ndarray] = None
    normaliser: Optional[Any] = None

    @property
    def is_multilabel(self) -> bool:
        return self.task == "multilabel"


DATASET_CHOICES = ("eicu", "mimic3_ihm", "mimic3_ph")


def load_dataset(
    dataset: str,
    *,
    data_root: str | os.PathLike = "data",
    train_batch: int = 256,
    val_batch: Optional[int] = None,
    test_batch: Optional[int] = None,
    pre_process: str = "std",
    sampler: bool = True,
    random_seed: int = 42,
    threshold: int = 48,
) -> DatasetBundle:
    """
    dataset:
        ``"eicu"``, ``"mimic3_ihm"`` or ``"mimic3_ph"``.
    data_root:
        ``eicu/``, ``mimic3/`` and ``mimic3/pheno/``.
    """
    dataset = dataset.lower()
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {DATASET_CHOICES}.")

    val_batch = val_batch or (16 * train_batch)
    test_batch = test_batch or (16 * train_batch)
    root = Path(data_root)

    if dataset == "eicu":
        train_loader, val_loader, test_loader = loaders_eicu(
            path=str(root / "eicu"),
            train_batch=train_batch,
            val_batch=val_batch,
            test_batch=test_batch,
            random_seed=random_seed,
        )
        return DatasetBundle(
            name=dataset,
            task="binary",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_features=402,
            output_dim=1,
        )

    if dataset == "mimic3_ihm":
        train_loader, val_loader, test_loader, tr_data, tr_lb, prpr = get_loaders_mimic3(
            path=str(root / "mimic3"),
            train_batch=train_batch,
            val_batch=val_batch,
            test_batch=test_batch,
            sampler=sampler,
            pre_process=pre_process,
        )
        return DatasetBundle(
            name=dataset,
            task="binary",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_features=60,
            output_dim=1,
            train_data=tr_data,
            train_labels=np.asarray(tr_lb),
            normaliser=prpr,
        )

    train_loader, val_loader, test_loader, num_classes, tr_data, tr_lb, prpr = (
        get_loaders_mimic3_multilabel(
            path=str(root / "mimic3" / "pheno"),
            train_batch=train_batch,
            val_batch=val_batch,
            test_batch=test_batch,
            sampler=sampler,
            pre_process=pre_process,
            threshold=threshold,
        )
    )
    return DatasetBundle(
        name=dataset,
        task="multilabel",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_features=60,
        output_dim=num_classes,
        train_data=tr_data,
        train_labels=np.asarray(tr_lb),
        normaliser=prpr,
    )
