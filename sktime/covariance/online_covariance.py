import numpy as np

from sktime.base import Estimator
from .util.running_moments import running_covar as running_covar

__all__ = ['OnlineCovariance']

__author__ = 'paul, nueske, marscher, clonker'


class OnlineCovarianceModel(object):
    def __init__(self, compute_c00, compute_c0t, compute_ctt, remove_data_mean, reversible, bessel, sparse_mode,
                 lagtime, ncov, column_selection, diag_only):

        self.compute_c00 = compute_c00
        self.compute_c0t = compute_c0t
        self.compute_ctt = compute_ctt
        self.remove_data_mean = remove_data_mean
        self.reversible = reversible
        self.bessel = bessel
        self.sparse_mode = sparse_mode
        self.lagtime = lagtime
        self.ncov = ncov
        self.column_selection = column_selection
        self.diag_only = diag_only

        self._cov_00 = None
        self._cov_01 = None
        self._cov_11 = None
        self._mean_0 = None
        self._mean_1 = None

    @property
    def cov_00(self):
        return self._cov_00

    @property
    def cov_01(self):
        return self._cov_01

    @property
    def cov_11(self):
        return self._cov_11

    @property
    def mean_0(self):
        return self._mean_0

    @property
    def mean_1(self):
        return self._mean_1

    @property
    def lagged(self):
        return self.compute_c0t or self.compute_ctt


class OnlineCovariance(Estimator):
    r"""Compute (potentially lagged) covariances between data in an online fashion.

     Parameters
     ----------
     c00 : bool, optional, default=True
         compute instantaneous correlations over the first part of the data. If lag==0, use all of the data.
         Makes the C00_ attribute available.
     c0t : bool, optional, default=False
         compute lagged correlations. Does not work with lag==0.
         Makes the C0t_ attribute available.
     ctt : bool, optional, default=False
         compute instantaneous correlations over the time-shifted chunks of the data. Does not work with lag==0.
         Makes the Ctt_ attribute available.
     remove_data_mean : bool, optional, default=False
         subtract the sample mean from the time series (mean-free correlations).
     reversible : bool, optional, default=False
         symmetrize correlations.
     bessel : bool, optional, default=True
         use Bessel's correction for correlations in order to use an unbiased estimator
     sparse_mode : str, optional, default='auto'
         one of:
             * 'dense' : always use dense mode
             * 'auto' : automatic
             * 'sparse' : always use sparse mode if possible
     lagtime: int, optional, default=0
         lag time. Does not work with c0t=True or ctt=True.
     column_selection: ndarray(k, dtype=int) or None
         Indices of those columns that are to be computed. If None, all columns are computed.
     diag_only: bool
         If True, the computation is restricted to the diagonal entries (autocorrelations) only.

     """

    def __init__(self, compute_c00=True, compute_c0t=False, compute_ctt=False, remove_data_mean=False,
                 reversible=False, bessel=True, sparse_mode='auto', lagtime=0, ncov=5, column_selection=None,
                 diag_only=False):

        if (compute_c0t or compute_ctt) and lagtime == 0:
            raise ValueError("lag must be positive if compute_c0t=True or compute_ctt=True")

        if column_selection is not None and diag_only:
            raise ValueError('Computing only parts of the diagonal is not supported.')
        if diag_only and sparse_mode is not 'dense':
            if sparse_mode is 'sparse':
                import warnings
                warnings.warn('Computing diagonal entries only is not implemented for sparse mode. Switching to dense '
                              'mode.')
            sparse_mode = 'dense'

        self._model = OnlineCovarianceModel(
            compute_c00=compute_c00, compute_c0t=compute_c0t, compute_ctt=compute_ctt,
            remove_data_mean=remove_data_mean, reversible=reversible, sparse_mode=sparse_mode,
            lagtime=lagtime,bessel=bessel, ncov=ncov, column_selection=column_selection, diag_only=diag_only)

        self._rc = running_covar(xx=self._model.compute_c00, xy=self._model.compute_c0t, yy=self._model.compute_ctt,
                                 remove_mean=self._model.remove_data_mean, symmetrize=self._model.reversible,
                                 sparse_mode=self._model.sparse_mode, modify_data=False,
                                 column_selection=self._model.column_selection, diag_only=self._model.diag_only,
                                 nsave=ncov)

    def fit(self, data, n_splits=None, partial_fit=False):

        if n_splits is None:
            n_splits = int(len(data[0]) // 100 if len(data[0]) >= 1e4 else 1)
        if self._model.lagged:
            if not (isinstance(data, (list, tuple)) and len(data) == 2 and len(data[0]) == len(data[1])):
                raise ValueError("Expected tuple of arrays of equal length!")
            x, x_lagged = data
        else:
            x, x_lagged = data, np.array([], dtype=data.dtype)

        for X, Y in zip(np.array_split(x, n_splits), np.array_split(x_lagged, n_splits)):
            assert len(X) == len(Y) or (not self._model.lagged and len(Y) == 0)
            self.partial_fit((X, Y))

        return self

    def partial_fit(self, data):
        """ incrementally update the estimates

        Parameters
        ----------
        data: array, list of arrays, PyEMMA reader
            input data.
        """
        X, Y = data
        try:
            self._rc.add(X, Y)
        except MemoryError:
            raise MemoryError('Covariance matrix does not fit into memory. '
                              'Input is too high-dimensional ({} dimensions). '.format(X.shape[1]))
        return self

    def _update_model(self):
        self._model._cov_00 = self._rc.cov_XX(self._model.bessel)
        self._model._cov_01 = self._rc.cov_XY(self._model.bessel)
        self._model._cov_11 = self._rc.cov_YY(self._model.bessel)
        self._model._mean_0 = self._rc.mean_X()
        self._model._mean_1 = self._rc.mean_Y()

    @property
    def model(self):
        self._update_model()
        return self._model