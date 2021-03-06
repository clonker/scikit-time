import random
import warnings
from typing import Optional

import numpy as np

from . import _clustering_bindings as _bd

from sktime.base import Estimator, Transformer
from sktime.clustering.cluster_model import ClusterModel

__all__ = ['KmeansClustering', 'MiniBatchKmeansClustering', 'KMeansClusteringModel']

from ..util import handle_n_jobs


class KMeansClusteringModel(ClusterModel):
    r"""
    The K-means clustering model. Stores all important information which are result of the estimation procedure.

    See Also
    --------
    ClusterModel : Superclass.
    """

    def __init__(self, n_clusters, cluster_centers, metric, tolerance, inertia=np.inf, converged=False):
        r"""
        Initializes a new KmeansClustering model.

        Parameters
        ----------
        n_clusters : int
            The number of cluster centers.
        cluster_centers : (k, d) ndarray
            The d-dimensional cluster centers, length of the array should coincide with :attr:`n_clusters`.
        metric : _clustering_bindings.Metric
            The metric that was used
        tolerance : float
            Tolerance which was used as convergence criterium.
        inertia : float, optional, default=np.inf
            Value of the cost function at the end of the estimation.
        converged : bool, optional, default=False
            Whether the convergence criterium was met.
        """
        super().__init__(n_clusters, cluster_centers, metric, converged=converged)
        self._inertia = inertia
        self._tolerance = tolerance

    @property
    def tolerance(self):
        """
        The tolerance used as stopping criterion in the kmeans clustering loop. In particular, when
        the relative change in the inertia is smaller than the given tolerance value.

        Returns
        -------
        float
            the tolerance
        """
        return self._tolerance

    @property
    def inertia(self):
        """
        Sum of squared distances to centers.

        Returns
        -------
        float
            the intertia
        """
        return self._inertia


class KmeansClustering(Estimator, Transformer):
    r"""
    Clusters the data in a way that minimizes the cost function

    .. math:: C(S) = \sum_{i=1}^{k} \sum_{\mathbf{x}_j \in S_i} \left\| \mathbf{x}_j - \boldsymbol\mu_i \right\|^2

    where :math:`S_i` are clusters with centers of mass :math:`\mu_i` and :math:`\mathbf{x}_j` data points
    associated to their clusters.

    The outcome is very dependent on the initialization, in particular we offer "kmeans++" and "uniform". The latter
    picks initial centers random-uniformly over the provided data set. The former tries to find an initialization
    which is covering the spatial configuration of the dataset more or less uniformly. For details see [1]_.

    References
    ----------
    .. [1] Arthur, David, and Sergei Vassilvitskii. k-means++: The advantages of careful seeding. Stanford, 2006.
    """

    def __init__(self, n_clusters: int, max_iter: int = 500, metric=None,
                 tolerance=1e-5, init_strategy='kmeans++', fixed_seed=False,
                 n_jobs=None, initial_centers=None, random_state=None):
        r"""
        Initializes a new k-means cluster estimator.

        Parameters
        ----------
        n_clusters : int
            amount of cluster centers.

        max_iter : int
            maximum number of iterations before stopping.

        metric : subclass of :class:`_clustering_bindings.Metric`
            metric to use during clustering, default None evaluates to euclidean metric, otherwise instance of a
            subclass of :class:`_clustering__bindings.Metric`.

        tolerance : float
            Stop iteration when the relative change in the cost function (inertia)

            .. math:: C(S) = \sum_{i=1}^{k} \sum_{\mathbf x \in S_i} \left\| \mathbf x - \boldsymbol\mu_i \right\|^2

            is smaller than tolerance.

        init_strategy : str
            one of 'kmeans++', 'uniform'; determining how the initial cluster centers are being chosen

        fixed_seed : bool or int
            if True, the seed gets set to 42. Use time based seeding otherwise. If an integer is given, use this to
            initialize the random generator.

        n_jobs : int or None, default None
            Number of threads to use during clustering and assignment of data. If None, one core will be used.

        initial_centers: None or np.ndarray[k, dim]
            This is used to resume the kmeans iteration. Note, that if this is set, the init_strategy is ignored and
            the centers are directly passed to the kmeans iteration algorithm.
        """
        super(KmeansClustering, self).__init__()

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric
        self.tolerance = tolerance
        self.init_strategy = init_strategy
        self.fixed_seed = fixed_seed
        if random_state is None:
            random_state = np.random.RandomState(self.fixed_seed)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.initial_centers = initial_centers

    @property
    def initial_centers(self) -> Optional[np.ndarray]:
        r"""
        Yields initial centers which override the :meth:`init_strategy`. Can be used to resume k-means iterations.

        :getter: The initial centers or None.
        :setter: Sets the initial centers. If not None, the array is expected to have length :attr:`n_clusters`.
        :type: (k, n) ndarray or None
        """
        return self._initial_centers

    @initial_centers.setter
    def initial_centers(self, value: Optional[np.ndarray]):
        if value is not None and value.shape[0] != self.n_clusters:
            raise ValueError("initial centers must be None or of shape (k, d) where k is the number of cluster centers."
                             " Expected k={}, got {}.".format(self.n_clusters, value.shape[0]))
        self._initial_centers = value

    @property
    def n_jobs(self) -> int:
        r"""
        Number of threads to use during clustering and assignment of data.

        :getter: Yields the number of threads. If -1, all available threads are used.
        :setter: Sets the number of threads to use. If -1, use all, if None, use 1.
        :type: int
        """
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: Optional[int]):
        self._n_jobs = handle_n_jobs(value)

    @property
    def n_clusters(self) -> int:
        r"""
        The number of cluster centers to use.

        :getter: Yields the number of cluster centers.
        :setter: Sets the number of cluster centers.
        :type: int
        """
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, value: int):
        self._n_clusters = value

    @property
    def max_iter(self) -> int:
        r"""
        Maximum number of clustering iterations before stop.

        :getter: Yields the maximum number of clustering iterations
        :setter: Sets the max. number of clustering iterations
        :type: int
        """
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int):
        self._max_iter = value

    @property
    def tolerance(self) -> float:
        r"""
        Stopping criterion for the k-means iteration. When the relative change of the cost function between two
        iterations is less than the tolerance, the algorithm is considered to be converged.

        :getter: Yields the currently set tolerance.
        :setter: Sets a new tolerance.
        :type: float
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float):
        self._tolerance = value

    @property
    def metric(self) -> _bd.Metric:
        r"""
        The metric that is used for clustering.

        :getter: Yields a subclass of :class:`_clustering_bindings.Metric`.
        :setter: Set a subclass of :class:`_clustering_bindings.Metric` to be used in clustering. Value can be `None`,
                 in which case the metric defaults to an Euclidean metric.
        :type: _clustering_bindings.Metric.

        See Also
        --------
        _clustering_bindings.Metric : The metric class, can be subclassed
        """
        return self._metric

    @metric.setter
    def metric(self, value: Optional[_bd.Metric]):
        if value is None:
            value = _bd.EuclideanMetric()
        self._metric = value

    def fetch_model(self) -> Optional[KMeansClusteringModel]:
        """
        Fetches the current model. Can be `None` in case :meth:`fit` was not called yet.

        Returns
        -------
        model : KMeansClusteringModel or None
            the latest estimated model
        """
        return self._model

    def transform(self, data, **kw):
        """
        Transforms a trajectory to a discrete trajectory by assigning each frame to its respective cluster center.

        Parameters
        ----------
        data : (T, n) ndarray
            trajectory with `T` frames and data points in `n` dimensions.
        **kw
            ignored kwargs for scikit-learn compatibility

        Returns
        -------
        discrete_trajectory : (T, 1) ndarray, dtype=int
            discrete trajectory

        See Also
        --------
        ClusterModel.transform : transform method of cluster model, implicitly called.
        """
        if self._model is None:
            raise ValueError("This estimator contains no model yet, fit should be called first.")
        return self.fetch_model().transform(data)

    @property
    def init_strategy(self):
        r"""Strategy to get an initial guess for the centers.

        :getter: Yields the strategy, can be one of "kmeans++" or "uniform".
        :setter: Setter for the initialization strategy that is used when no initial centers are provided.
        :type: string
        """
        return self._init_strategy

    @init_strategy.setter
    def init_strategy(self, value: str):
        valid = ('kmeans++', 'uniform')
        if value not in valid:
            raise ValueError('invalid parameter "{}" for init_strategy. Should be one of {}'.format(value, valid))
        self._init_strategy = value

    @property
    def fixed_seed(self):
        """ seed for random choice of initial cluster centers.

        Fix this to get reproducible results in conjunction with n_jobs=0. The latter is needed, because parallel
        execution causes non-deterministic behaviour again.
        """
        return self._fixed_seed

    @fixed_seed.setter
    def fixed_seed(self, value: [bool, int, None]):
        """
        Sets a fixed seed for cluster estimation to get reproducible results. This only works in the case of n_jobs=0
        or n_jobs=1, parallel execution reintroduces non-deterministic behavior.

        Parameters
        ----------
        value : bool or int or None
            If the value is `True`, the seed will be fixed on `42`, if it is `False` or `None`, the seed gets drawn
            randomly. In case an `int` value is provided, that will be used as fixed seed.

        """
        if isinstance(value, bool) or value is None:
            if value:
                self._fixed_seed = 42
            else:
                self._fixed_seed = random.randint(0, 2 ** 32 - 1)
        elif isinstance(value, int):
            if value < 0 or value > 2 ** 32 - 1:
                warnings.warn("seed has to be non-negative (or smaller than 2**32)."
                              " Seed will be chosen randomly.")
                self.fixed_seed = False
            else:
                self._fixed_seed = value
        else:
            raise ValueError("fixed seed has to be None, bool or integer")

    def _pick_initial_centers(self, data, strategy, n_jobs, callback=None):
        if self.n_clusters > len(data):
            raise ValueError('Not enough data points for desired amount of clusters.')

        if strategy == 'uniform':
            return data[self.random_state.randint(0, len(data), size=self.n_clusters)]
        elif self.init_strategy == 'kmeans++':
            return _bd.kmeans.init_centers_kmpp(data, self.n_clusters, self.fixed_seed, n_jobs,
                                                callback, self.metric)
        else:
            raise ValueError(f"Unknown cluster center initialization strategy \"{strategy}\", supported are "
                             f"\"uniform\" and \"kmeans++\"")

    def fit(self, data, initial_centers=None, callback_init_centers=None, callback_loop=None, n_jobs=None):
        """ Perform the clustering.

        Parameters
        ----------
        data: np.ndarray
            data to be clustered, shape should be (N, D), where N is the number of data points, D the dimension.
            In case of one-dimensional data, a shape of (N,) also works.
        initial_centers: np.ndarray or None
            Optional cluster center initialization that supersedes the estimator's `initial_centers` attribute
        callback_init_centers: function or None
            used for kmeans++ initialization to indicate progress, called once per assigned center.
        callback_loop: function or None
            used to indicate progress on kmeans iterations, called once per iteration.
        n_jobs: None or int
            if not None, supersedes the n_jobs attribute of the estimator instance; must be non-negative

        Returns
        -------
        self : KmeansClustering
            reference to self
        """
        if data.ndim == 1:
            data = data[:, np.newaxis]
        n_jobs = self.n_jobs if n_jobs is None else n_jobs
        if initial_centers is not None:
            self.initial_centers = initial_centers
        if self.initial_centers is None:
            self.initial_centers = self._pick_initial_centers(data, self.init_strategy, n_jobs, callback_init_centers)

        # run k-means with all the data
        converged = False
        cluster_centers, code, iterations, cost = _bd.kmeans.cluster_loop(data, self.initial_centers, self.n_clusters,
                                                                          n_jobs, self.max_iter, self.tolerance,
                                                                          callback_loop, self.metric)
        if code == 0:
            converged = True
        else:
            warnings.warn("Algorithm did not reach convergence criterion"
                          " of {t} in {i} iterations. Consider increasing max_iter.".format(t=self.tolerance,
                                                                                            i=self.max_iter))
        self._model = KMeansClusteringModel(n_clusters=self.n_clusters, metric=self.metric, tolerance=self.tolerance,
                                            cluster_centers=cluster_centers, inertia=cost, converged=converged)

        return self


class MiniBatchKmeansClustering(KmeansClustering):
    r""" K-means clustering in a mini-batched fashion.

    See Also
    --------
    KmeansClustering : k-means clustering without mini-batching
    """

    def __init__(self, n_clusters, max_iter=5, metric=None, tolerance=1e-5, init_strategy='kmeans++',
                 n_jobs=None, initial_centers=None):
        """
        Constructs a Minibatch k-means estimator. For a detailed argument description, see :class:`KmeansClustering`.
        """
        super(MiniBatchKmeansClustering, self).__init__(n_clusters, max_iter, metric,
                                                        tolerance, init_strategy, False,
                                                        n_jobs=n_jobs,
                                                        initial_centers=initial_centers)

    def partial_fit(self, data, n_jobs=None):
        r"""
        Updates the current model (or creates a new one) with data. This method can be called repeatedly and thus
        be used to train a model in an on-line fashion. Note that usually multiple passes over the data is used.
        Also this method should not be mixed with calls to :meth:`fit`, as then the model is overwritten with a new
        instance based on the data passed to :meth:`fit`.

        Parameters
        ----------
        data : (T, n) ndarray
            Data with which the model is updated and/or initialized.
        n_jobs : int, optional, default=None
            number of jobs to use when updating the model, supersedes the n_jobs attribute of the estimator.

        Returns
        -------
        self : MiniBatchKmeansClustering
            reference to self
        """
        if self._model is None:
            self._model = KMeansClusteringModel(n_clusters=self.n_clusters, cluster_centers=None, metric=self.metric,
                                                tolerance=self.tolerance, inertia=float('inf'))
        if data.ndim == 1:
            data = data[:, np.newaxis]
        n_jobs = self.n_jobs if n_jobs is None else n_jobs
        if self._model.cluster_centers is None:
            if self.initial_centers is None:
                # we have no initial centers set, pick some based on the first partial fit
                self._model._cluster_centers = self._pick_initial_centers(data, self.init_strategy, n_jobs)
            else:
                self._model._cluster_centers = np.copy(self.initial_centers)

        self._model._cluster_centers = _bd.kmeans.cluster(data, self._model.cluster_centers, n_jobs, self.metric)
        cost = _bd.kmeans.cost_function(data, self._model.cluster_centers, n_jobs, self.metric)

        rel_change = np.abs(cost - self._model.inertia) / cost if cost != 0.0 else 0.0
        self._model._inertia = cost

        if rel_change <= self.tolerance:
            self._model._converged = True

        return self
