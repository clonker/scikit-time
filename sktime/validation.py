from typing import List, Union

import numpy as np
from sktime.markov._base import argblocksplit_trajs

from . import _base_bindings as _bindings


class TimeSeriesContainer(list):
    r"""A container for multiple trajectories which can be accessed through two-dimensional slicing.
    This enables it to be used with scikit-learn's cross validation implementations.
    """

    def __init__(self, data: Union[np.ndarray, List[np.ndarray]]):
        r"""Creates a new time series container from either a single trajectory or a list of trajectories.
        The dtype must be either float32 or float64 and the dimension must match. This is not a requirement
        for the length of each trajectory, i.e., the container can be 'jagged'.

        Parameters
        ----------
        data : ndarray or list of ndarray
            Trajectory or list of trajectories.
        """
        super().__init__()
        if not isinstance(data, (list, tuple)):
            data = [data]

        n_trajectories = len(data)
        if not n_trajectories > 0:
            raise ValueError("A TimeSeriesContainer requires at least one trajectory.")
        if data[0].ndim < 2:
            raise ValueError("All trajectories must be two-dimensional, but the first element in the "
                             "collection was not.")
        dim = data[0].shape[1]
        dtype = data[0].dtype
        if dtype not in (np.float32, np.float64):
            raise ValueError(f"Only dtypes float32 and float64 supported, but got {dtype}")
        min_n_frames = float('inf')
        max_n_frames = 0

        self._min_n_frames = min_n_frames
        self._max_n_frames = max_n_frames
        self._n_trajectories = n_trajectories
        self._dim = dim
        self._dtype = dtype

        for d in data:
            self.append(d)

    def __getitem__(self, item):
        return self._handle_slice(item)

    def append(self, trajectory: np.ndarray):
        if not isinstance(trajectory, np.ndarray):
            raise ValueError("Container only supports ndarrays.")
        if trajectory.ndim != 2:
            raise ValueError(f"Each trajectory can only be two-dimensional, "
                             f"but trajectory got {trajectory.ndim} dimensions.")
        if trajectory.shape[1] != self._dim:
            raise ValueError(f"All trajectories in a time series container must "
                             f"have the same dimension (shape[1]).")
        if trajectory.dtype != self._dtype:
            raise ValueError("All trajectories must have the same dtype.")
        if trajectory.shape[0] < self._min_n_frames:
            self._min_n_frames = trajectory.shape[0]
        if trajectory.shape[0] > self._max_n_frames:
            self._max_n_frames = trajectory.shape[0]
        super().append(trajectory)

    @staticmethod
    def _from_tuple(tup: tuple, idx: int, default: np.ndarray = None) -> np.ndarray:
        r""" Yields a value from a tuple at a particular index. If out of bounds, returns default.

        Parameters
        ----------
        tup : tuple
            The input tuple
        idx : int
            Index of the element
        default : ndarray
            Default value to return if index is out of bounds

        Returns
        -------
        element : ndarray
            Element at :code:`tup[idx]` or default.
        """
        if idx < len(tup):
            item = tup[idx]
            if not isinstance(item, (list, tuple, slice)):
                item = [item]
            return np.array(item)
        else:
            return default

    def _handle_slice(self, item):
        if isinstance(item, int):
            return super(TimeSeriesContainer, self).__getitem__(item)
        if len(item) != 2:
            raise ValueError(f"Can only slice over two axes (trajectory and time) but got {len(item)} axes.")
        trajectories = np.atleast_1d(item[0])
        frames = np.atleast_1d(item[1])

        if trajectories.ndim != 1 or frames.ndim != 1 or len(trajectories) != len(frames):
            raise ValueError("Can only slice with one-dimensional arrays of equal length.")
        if np.any(trajectories >= len(self)):
            raise ValueError("Requested trajectories which are out of bounds for this collection.")

        out = np.empty((len(trajectories), self._dim), dtype=self._dtype)
        if self._dtype == np.float32:
            _bindings.gather_frames_f(self, trajectories, frames, out)
        else:
            _bindings.gather_frames_d(self, trajectories, frames, out)
        return out


class TimeSeriesCVSplitter(object):
    def __init__(self, n_splits=10, lagtime=1, sliding=True, *, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.lagtime = lagtime
        self.sliding = sliding

    def split(self, X, y=None, groups=None):
        from sklearn.utils import check_random_state
        random_state = check_random_state(self.random_state)
        for fold in range(self.n_splits):
            split = argblocksplit_trajs(X, lagtime=self.lagtime, sliding=self.sliding, random_state=self.random_state)

            I0 = random_state.choice(len(split), int(len(split) / 2), replace=False)
            I1 = np.array(list(set(list(np.arange(len(split)))) - set(list(I0))))
            dtrajs_train = (tuple(np.concatenate([split[i][0] for i in I0]).tolist()),
                            tuple(np.concatenate([split[i][1] for i in I0]).tolist()))
            dtrajs_test = (tuple(np.concatenate([split[i][0] for i in I1]).tolist()),
                           tuple(np.concatenate([split[i][1] for i in I1]).tolist()))

            yield dtrajs_train, dtrajs_test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
