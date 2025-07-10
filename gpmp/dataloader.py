"""
Data helpers

Components
----------
Dataset
    Lightweight container for covariates *x* and observations *z*.
    Accepts either a single array or a list of array “shards”.
    Provides random train/validation/test split, k-fold and repeated
    k-fold index generation, row sub-selection, and safe length checks
    across shards.

DataLoader
    Iterator that returns successive subsamples (“mini-batches”).
    Features deterministic shuffling via ``set_epoch()``, optional
    infinite cycling, and controlled last-batch handling
    (``drop_last``).

Normaliser, RobustScaler, MinMaxScaler, ObservationScaler
    Column-wise location/scale transforms commonly used before
    numerical estimation.  All ``transform`` / ``inverse_transform``
    methods accept ``copy: bool = True`` to avoid unintended in-place
    modification.

collate_batches
    Helper to stack a list of mini-batches back into full arrays.

Design note
-----------
* Sharded datasets are concatenated lazily; consistency across shards
  is asserted at construction.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2025, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

import bisect
import gpmp.num as gnp
from typing import Tuple, List, Iterator, Optional, Union


Array = gnp.ndarray
Idx = gnp.ndarray
ArrayOrList = Union[Array, List[Array]]
_EPS = 1e-8


# ======================================================================
#                               Dataset
# ======================================================================
class Dataset:
    """Dataset storing covariates *x* and observations *z*.

    *x* and *z* may each be a single array or a list of arrays (shards)
    that share the same first-dimension length.

    Shards are retained at construction and indexing is performed
    lazily, without concatenation, with O(log(#shards)) index lookup.

    """

    def __init__(self, x: ArrayOrList, z: ArrayOrList) -> None:
        """
        Parameters
        ----------
        x : Array or list of Array
            Covariates, ``(n_samples, n_features)`` or list of such arrays.
        z : Array or list of Array
            Observations, ``(n_samples, …)`` or list of such arrays.
        """
        # normalise to lists
        self.x_list = x if isinstance(x, list) else [x]
        self.z_list = z if isinstance(z, list) else [z]

        # ensure all shards are gpmp.num arrays
        self.x_list = [gnp.asarray(xi) for xi in self.x_list]
        self.z_list = [gnp.asarray(zi) for zi in self.z_list]

        # consistency check
        assert len(self.x_list) == len(self.z_list), "x and z shard counts differ"
        for xi, zi in zip(self.x_list, self.z_list):
            assert xi.shape[0] == zi.shape[0], "shard length mismatch"

        self.size = sum(xi.shape[0] for xi in self.x_list)
        self._shard_bounds = self._compute_shard_bounds()

    def _compute_shard_bounds(self) -> List[int]:
        """Compute cumulative sample bounds across shards."""
        bounds = []
        cumsum = 0
        for xi in self.x_list:
            cumsum += xi.shape[0]
            bounds.append(cumsum)
        return bounds

    # ------------------------------------------------------------- special methods
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Array, Array]:
        """Return ``(x[idx], z[idx])`` without full-array concatenation."""
        shard_idx = bisect.bisect_right(self._shard_bounds, idx)
        start = 0 if shard_idx == 0 else self._shard_bounds[shard_idx - 1]
        local_idx = idx - start
        return self.x_list[shard_idx][local_idx], self.z_list[shard_idx][local_idx]

    def __repr__(self) -> str:
        shards = len(self.x_list)
        return (
            f"{self.__class__.__name__}(size={self.size}, "
            f"shards={shards}, "
            f"x_shape={[x.shape for x in self.x_list]}, "
            f"z_shape={[z.shape for z in self.z_list]})"
        )

    # ------------------------------------------------------------- slice
    def subset(self, indices: Idx) -> "Dataset":
        """Return a dataset restricted to *indices*.

        Shard structure is preserved if possible.
        """
        if indices.ndim != 1:
            raise ValueError("Subset indices must be 1D")

        indices = gnp.sort(indices)  # enforce increasing order
        xs = []
        zs = []

        shard_starts = [0] + self._shard_bounds[:-1]
        shard_ends = self._shard_bounds

        for shard_idx, (start, end) in enumerate(zip(shard_starts, shard_ends)):
            mask = (indices >= start) & (indices < end)
            if mask.any():
                local_idx = indices[mask] - start
                xi = gnp.index_select(self.x_list[shard_idx], 0, local_idx)
                zi = gnp.index_select(self.z_list[shard_idx], 0, local_idx)
                xs.append(xi)
                zs.append(zi)

        return Dataset(xs, zs)

    # ------------------------------------------------------------- split
    @staticmethod
    def split(
        dataset: "Dataset",
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: Optional[int] = None,
    ) -> Tuple["Dataset", "Dataset", "Dataset"]:
        """Return (train, val, test) datasets according to *ratios*.

        Samples are randomly shuffled before splitting.
        """
        assert gnp.isclose(
            gnp.asarray(sum(ratios)), gnp.asarray(1.0)
        ), "Ratios must sum to 1"
        if seed is not None:
            gnp.set_seed(seed)

        n = len(dataset)
        idx = gnp.permutation(n)
        n_tr = int(ratios[0] * n)
        n_va = int(ratios[1] * n)
        return (
            dataset.subset(idx[:n_tr]),
            dataset.subset(idx[n_tr : n_tr + n_va]),
            dataset.subset(idx[n_tr + n_va :]),
        )

    # .............................................................. k-fold
    @staticmethod
    def k_fold_indices(
        n_samples: int, n_splits: int, seed: Optional[int] = None
    ) -> List[Tuple[Idx, Idx]]:
        """Return exactly *k* (train, val) index tuples for k-fold CV.

        Each split has approximately the same size.
        """
        if seed is not None:
            gnp.set_seed(seed)

        idx = gnp.permutation(n_samples)
        base, r = divmod(n_samples, n_splits)
        sizes = gnp.concatenate(
            [
                gnp.full((r,), base + 1, dtype=int),
                gnp.full((n_splits - r,), base, dtype=int),
            ]
        )
        bounds = gnp.cumsum(sizes, axis=0)[:-1]
        folds = gnp.split(idx, bounds)

        out: List[Tuple[Idx, Idx]] = []
        for k in range(n_splits):
            val = folds[k]
            train = gnp.concatenate([folds[j] for j in range(n_splits) if j != k], 0)
            out.append((train, val))
        return out

    # .................................................... repeated k-fold
    @staticmethod
    def repeated_k_fold_indices(
        n_samples: int, n_splits: int, n_repeats: int, seed: Optional[int] = None
    ) -> List[Tuple[Idx, Idx]]:
        """Return *n_repeats × k* shuffled k-fold splits.

        Each repetition is independently shuffled.
        """
        out: List[Tuple[Idx, Idx]] = []
        for r in range(n_repeats):
            out += Dataset.k_fold_indices(
                n_samples, n_splits, None if seed is None else seed + r
            )
        return out

    # ................................................... internal methods
    def _reduce_min(self, x_or_z: str) -> Array:
        lst = getattr(self, f"{x_or_z}_list")
        first = True
        for data in lst:
            shard_min = gnp.min(data, axis=0)
            if first:
                global_min = shard_min
                first = False
            else:
                global_min = gnp.minimum(global_min, shard_min)
        return global_min

    def _reduce_max(self, x_or_z: str) -> Array:
        lst = getattr(self, f"{x_or_z}_list")
        first = True
        for data in lst:
            shard_max = gnp.max(data, axis=0)
            if first:
                global_max = shard_max
                first = False
            else:
                global_max = gnp.maximum(global_max, shard_max)
        return global_max

    def _reduce_mean(self, x_or_z: str) -> Array:
        lst = getattr(self, f"{x_or_z}_list")
        total_sum = None
        n = 0
        for data in lst:
            shard_sum = gnp.sum(data, axis=0)
            if total_sum is None:
                total_sum = shard_sum
            else:
                total_sum += shard_sum
            n += data.shape[0]
        return total_sum / n

    def _reduce_var(self, x_or_z: str) -> Array:
        mean = self._reduce_mean(x_or_z)
        lst = getattr(self, f"{x_or_z}_list")
        total_var = None
        n = 0
        for data in lst:
            x_centered = data - mean
            shard_var = gnp.sum(x_centered**2, axis=0)
            if total_var is None:
                total_var = shard_var
            else:
                total_var += shard_var
            n += data.shape[0]
        return total_var / (n - 1)

    def _reduce_std(self, x_or_z: str) -> Array:
        return gnp.sqrt(self._reduce_var(x_or_z))

    def _reduce_quantile(self, x_or_z: str, q: float) -> Array:
        if not (0.0 <= q <= 1.0):
            raise ValueError("quantile level q must be between 0 and 1")
        lst = getattr(self, f"{x_or_z}_list")
        data_full = gnp.concatenate(lst, axis=0)
        return gnp.percentile(data_full, q * 100.0, axis=0)

    def _reduce_quantile_approx(
        self, x_or_z: str, q: float, sample_size: int = 1000
    ) -> Array:
        if not (0.0 <= q <= 1.0):
            raise ValueError("quantile level q must be between 0 and 1")
        lst = getattr(self, f"{x_or_z}_list")
        subsamples = []
        for data in lst:
            n = data.shape[0]
            idx = gnp.choice(n, min(n, sample_size // len(lst)), replace=False)
            subsamples.append(data[idx])
        sample = gnp.concatenate(subsamples, axis=0)
        return gnp.percentile(sample, q * 100.0, axis=0)

    def _reduce_median(self, x_or_z: str) -> Array:
        return self._reduce_quantile(x_or_z, 0.5)


# -----------------------------------------------------------------------
# Auto-generate x_* and z_* methods
# -----------------------------------------------------------------------

for field in ("x", "z"):
    for stat in ("min", "max", "mean", "var", "std", "quantile", "median"):

        def make_method(field=field, stat=stat):
            def method(self, *args, **kwargs):
                return getattr(self, f"_reduce_{stat}")(field, *args, **kwargs)

            method.__name__ = f"{field}_{stat}"
            return method

        setattr(Dataset, f"{field}_{stat}", make_method())


# ======================================================================
#                               DataLoader
# ======================================================================
class DataLoader:
    """
    Mini-batch generator over a Dataset.

    Supports optional shuffling, infinite iteration,
    and deterministic seeding across epochs for reproducibility
    (especially useful with distributed training).

    Fetching is shard-aware and avoids full dataset concatenation.

    Behaviour when the last batch is incomplete:

    * ``drop_last=True``  – discard it.
    * ``drop_last=False`` – yield a smaller batch.

    Notes
    -----
    * Call :py:meth:`set_epoch` before each epoch to ensure deterministic
      shuffling across distributed workers.
    * If ``infinite=True`` the iterator cycles forever (useful for GANs).
    """

    def __init__(
        self,
        dataset: "Dataset",
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
        infinite: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        dataset : Dataset
            Dataset instance to sample from.
        batch_size : int
            Number of samples per batch.
        shuffle : bool, optional
            Whether to shuffle indices at each epoch (default: True).
        drop_last : bool, optional
            Whether to drop last incomplete batch (default: False).
        seed : int, optional
            Base random seed for reproducible shuffling (default: None).
        infinite : bool, optional
            Whether to cycle infinitely (default: False).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._base_seed = seed
        self._epoch = 0
        self._infinite = infinite

    # ------------------------------------------------------------- epoch control
    def set_epoch(self, epoch: int) -> None:
        """Manually set current epoch (affects shuffling)."""
        self._epoch = epoch

    # ------------------------------------------------------------- iteration
    def __iter__(self) -> Iterator[Tuple[Array, Array]]:
        """Yield successive ``(x_batch, z_batch)``; infinite if requested."""
        while True:
            if self._base_seed is not None:
                gnp.set_seed(self._base_seed + self._epoch)

            n = len(self.dataset)
            idx = gnp.permutation(n) if self.shuffle else gnp.arange(n)

            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                if end > n and self.drop_last:
                    break
                batch_idx = idx[start:end]
                yield self._fetch_batch(batch_idx)

            self._epoch += 1
            if not self._infinite:
                break

    def _fetch_batch(self, batch_idx: Idx) -> Tuple[Array, Array]:
        """Select batch across shards without full concatenation."""
        xs = []
        zs = []

        shard_starts = [0] + self.dataset._shard_bounds[:-1]
        shard_ends = self.dataset._shard_bounds

        for shard_idx, (start, end) in enumerate(zip(shard_starts, shard_ends)):
            mask = (batch_idx >= start) & (batch_idx < end)
            if mask.any():
                local_idx = batch_idx[mask] - start
                xi = gnp.index_select(self.dataset.x_list[shard_idx], 0, local_idx)
                zi = gnp.index_select(self.dataset.z_list[shard_idx], 0, local_idx)
                xs.append(xi)
                zs.append(zi)

        return gnp.concatenate(xs, 0), gnp.concatenate(zs, 0)

    # -------------------------------------------------- special methods
    def __len__(self) -> int:
        """Return number of batches in one finite epoch."""
        n = len(self.dataset)
        full = n // self.batch_size
        return full if (self.drop_last or n % self.batch_size == 0) else full + 1

    def __getitem__(self, index: Union[int, slice]) -> Union[Tuple[Array, Array], List[Tuple[Array, Array]]]:
        """
        Enable batch indexing or slicing when shuffle=False and infinite=False.

        Parameters
        ----------
        index : int or slice
            Batch index or range of batch indices.

        Returns
        -------
        (x_batch, z_batch) or list of such tuples

        Raises
        ------
        RuntimeError if shuffling or infinite mode is active.
        IndexError for out-of-bounds access.
        """
        if self.shuffle or self._infinite:
            raise RuntimeError("Batch indexing requires shuffle=False and infinite=False.")

        n_batches = len(self)
        if isinstance(index, int):
            if index < 0:
                index += n_batches
            if not (0 <= index < n_batches):
                raise IndexError("Batch index out of range")

            start = index * self.batch_size
            end = min(start + self.batch_size, len(self.dataset))
            if end > len(self.dataset) and self.drop_last:
                raise IndexError("Index corresponds to dropped last batch")
            batch_idx = gnp.arange(start, end)
            return self._fetch_batch(batch_idx)

        elif isinstance(index, slice):
            indices = range(*index.indices(n_batches))
            return [self[i] for i in indices]

        else:
            raise TypeError("Index must be int or slice")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last}, "
            f"infinite={self._infinite}, dataset_size={len(self.dataset)})"
        )
    
    # ------------------------------------------------------------- mean
    def reduce_mean(self, func) -> Array:
        """
        Compute the weighted mean of `func(x_batch, z_batch)` over batches.

        Each batch output is weighted by its batch size.

        Parameters
        ----------
        func : callable
            A function (x_batch, z_batch) -> scalar or array (batch result).

        Returns
        -------
        mean : Array
            Weighted mean over all samples.
        """
        total = 0.0
        total_weight = 0

        for x_batch, z_batch in self:
            batch_size = x_batch.shape[0]
            value = func(x_batch, z_batch)
            total += value * batch_size
            total_weight += batch_size

        return total / total_weight


# ----------------------------------------------------------------
# Auto-generate dataset_* properties
# ----------------------------------------------------------------

for stat in ("min", "max", "mean", "var", "std", "quantile", "median"):
    for field in ("x", "z"):
        prop_name = f"dataset_{field}_{stat}"
        method_name = f"{field}_{stat}"

        def make_property(method_name=method_name):
            @property
            def prop(self):
                return getattr(self.dataset, method_name)

            return prop

        setattr(DataLoader, prop_name, make_property())


# ======================================================================
#                 Normalization / Standardization helpers
# ======================================================================
# (Normaliser, RobustScaler, MinMaxScaler unchanged)
class Normalizer:
    """Standardize covariates to zero mean and unit variance."""

    def __init__(self, mean: Array, std: Array) -> None:
        self.mean = mean
        self.std = gnp.where(std < _EPS, 1.0, std)

    def transform(self, x: Array, copy: bool = True) -> Array:
        if copy:
            x = gnp.copy(x)
        return (x - self.mean) / self.std

    def inverse_transform(self, x_std: Array, copy: bool = True) -> Array:
        if copy:
            x_std = gnp.copy(x_std)
        return x_std * self.std + self.mean

    @staticmethod
    def fit(x: Array) -> "Normalizer":
        return Normalizer(x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True))


class RobustScaler:
    """Scale covariates by median and interquartile range."""

    def __init__(self, median: Array, iqr: Array) -> None:
        self.median = median
        self.iqr = gnp.where(iqr < _EPS, 1.0, iqr)

    def transform(self, x: Array, copy: bool = True) -> Array:
        if copy:
            x = gnp.copy(x)
        return (x - self.median) / self.iqr

    def inverse_transform(self, x_rb: Array, copy: bool = True) -> Array:
        if copy:
            x_rb = gnp.copy(x_rb)
        return x_rb * self.iqr + self.median

    @staticmethod
    def fit(x: Array, q_low: float = 25.0, q_high: float = 75.0) -> "RobustScaler":
        q_low = gnp.percentile(x, q_low, axis=0, keepdims=True)
        q_high = gnp.percentile(x, q_high, axis=0, keepdims=True)
        median = gnp.percentile(x, 50.0, axis=0, keepdims=True)
        iqr = q_high - q_low
        return RobustScaler(median, iqr)


class MinMaxScaler:
    """Rescale covariates to lie between 0 and 1."""

    def __init__(self, x_min: Array, x_max: Array) -> None:
        self.x_min = x_min
        self.range = gnp.where((x_max - x_min) < _EPS, 1.0, x_max - x_min)

    def transform(self, x: Array, copy: bool = True) -> Array:
        if copy:
            x = gnp.copy(x)
        return (x - self.x_min) / self.range

    def inverse_transform(self, x_mm: Array, copy: bool = True) -> Array:
        if copy:
            x_mm = gnp.copy(x_mm)
        return x_mm * self.range + self.x_min

    @staticmethod
    def fit(x: Array) -> "MinMaxScaler":
        return MinMaxScaler(
            gnp.min(x, axis=0, keepdims=True), gnp.max(x, axis=0, keepdims=True)
        )


class ObservationScaler:
    """Standardise scalar observations to zero mean and unit variance."""

    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = 1.0 if std < _EPS else std

    def transform(self, z: Array, copy: bool = True) -> Array:
        if copy:
            z = gnp.copy(z)
        return (z - self.mean) / self.std

    def inverse_transform(self, z_std: Array, copy: bool = True) -> Array:
        if copy:
            z_std = gnp.copy(z_std)
        return z_std * self.std + self.mean

    @staticmethod
    def fit(z: Array) -> "ObservationScaler":
        return ObservationScaler(z.mean(), z.std())


# ======================================================================
#                            Collate helper
# ======================================================================
def collate_batches(batches: List[Tuple[Array, Array]]) -> Tuple[Array, Array]:
    """Concatenate a list of ``(x_batch, z_batch)``.

    Raises
    ------
    ValueError
        If batch list is empty.
    """
    if not batches:
        raise ValueError("Cannot collate an empty list of batches.")

    xs, zs = zip(*batches)
    return gnp.concatenate(xs, 0), gnp.concatenate(zs, 0)


# ----------------------------------------------------------------------
#                              TODO (roadmap)
# ----------------------------------------------------------------------
# * Sampling & shuffling: class-weighted / probability sampling
# * Dataset: map / filter hooks for online transforms
# * Dataloader: Multi-worker prefetch, pinned-memory, device copy helpers
# ----------------------------------------------------------------------
