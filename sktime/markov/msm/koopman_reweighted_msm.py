import warnings

import numpy as _np
from msmtools.estimation import effective_count_matrix

from sktime.markov.msm import MarkovStateModel
from sktime.markov._base import _MSMBaseEstimator
from sktime.markov.transition_counting import TransitionCountEstimator, TransitionCountModel
from sktime.util import submatrix
from ._koopman_reweighted_msm_impl import (bootstrapping_count_matrix, bootstrapping_dtrajs, twostep_count_matrix,
                                           rank_decision, oom_components, equilibrium_transition_matrix)

__author__ = 'Feliks Nüske, Fabian Paul, marscher'


class KoopmanReweightedMSM(MarkovStateModel):
    def __init__(self, eigenvalues_OOM=None, sigma=None, omega=None, *args, **kwargs):
        super(KoopmanReweightedMSM, self).__init__(*args, **kwargs)
        self._eigenvalues_oom = eigenvalues_OOM
        self._Xi = oom_components
        self._omega = omega
        self._sigma = sigma
        if sigma is not None:
            self._oom_rank = sigma.size

    @property
    def eigenvalues_OOM(self):
        """System eigenvalues estimated by OOM."""
        return self._eigenvalues_oom

    @property
    def timescales_OOM(self):
        """System timescales estimated by OOM."""
        return -self.lagtime / _np.log(_np.abs(self._eigenvalues_oom[1:]))

    @property
    def OOM_rank(self):
        """Return OOM model rank."""
        return self._oom_rank

    @property
    def OOM_components(self):
        """Return OOM components."""
        return self._Xi

    @property
    def OOM_omega(self):
        """ Return OOM initial state vector."""
        return self._omega

    @property
    def OOM_sigma(self):
        """Return OOM evaluator vector."""
        return self._sigma


class OOMReweightedMSM(_MSMBaseEstimator):
    r"""Maximum likelihood estimator for MSMs given discrete trajectory statistics

    Parameters
    ----------
    lagtime : int
        lag time at which transitions are counted and the transition matrix is
        estimated.

    reversible : bool, optional, default = True
        If true compute reversible MSM, else non-reversible MSM

    count_mode : str, optional, default='sliding'
        mode to obtain count matrices from discrete trajectories. Should be
        one of:

        * 'sliding' : A trajectory of length T will have :math:`T-\tau` counts
          at time indexes

          .. math::

             (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)
        * 'sample' : A trajectory of length T will have :math:`T/\tau` counts
          at time indexes

          .. math::

                (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., (((T/\tau)-1) \tau \rightarrow T)

    sparse : bool, optional, default = False
        If true compute count matrix, transition matrix and all derived
        quantities using sparse matrix algebra. In this case python sparse
        matrices will be returned by the corresponding functions instead of
        numpy arrays. This behavior is suggested for very large numbers of
        states (e.g. > 4000) because it is likely to be much more efficient.

    time_unit : str, optional, default='1 step'
        Description of the physical time of the input trajectories. May be used
        by analysis algorithms such as plotting tools to pretty-print the axes.
        By default '1 step', i.e. there is no physical time unit. Specify by a
        number, whitespace and unit. Permitted units are (* is an arbitrary
        string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

    nbs : int, optional, default=10000
        number of re-samplings for rank decision in OOM estimation.

    rank_Ct : str, optional
        Re-sampling method for model rank selection. Can be
        * 'bootstrap_counts': Directly re-sample transitions based on effective count matrix.

        * 'bootstrap_trajs': Re-draw complete trajectories with replacement.

    tol_rank: float, optional, default = 10.0
        signal-to-noise threshold for rank decision.

    connectivity_threshold : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/n_states.

    References
    ----------
    .. [1] H. Wu and F. Noe: Variational approach for learning Markov processes from time series data
        (in preparation)

    """

    def __init__(self, lagtime, reversible=True, count_mode='sliding', sparse=False,
                 time_unit='1 step', nbs=10000, rank_Ct='bootstrap_counts', tol_rank=10.0,
                 connectivity_threshold='1/n'):

        # Check count mode:
        self.count_mode = str(count_mode).lower()
        if self.count_mode not in ('sliding', 'sample'):
            raise ValueError(
                'count mode {} is unknown. Only \'sliding\' and \'sample\' are allowed.'.format(count_mode))
        if rank_Ct not in ('bootstrap_counts', 'bootstrap_trajs'):
            raise ValueError('rank_Ct must be either \'bootstrap_counts\' or \'bootstrap_trajs\'')

        super(OOMReweightedMSM, self).__init__(lagtime=lagtime, reversible=reversible, count_mode=count_mode,
                                               sparse=sparse,
                                               time_unit=time_unit, connectivity_threshold=connectivity_threshold)
        self.nbs = nbs
        self.tol_rank = tol_rank
        self.rank_Ct = rank_Ct

    def fit(self, dtrajs, **kw):
        # remove last lag steps from dtrajs:
        dtrajs_lag = [traj[:-self.lagtime] for traj in dtrajs]
        count_model = TransitionCountEstimator(lagtime=self.lagtime, mincount_connectivity=self.connectivity_threshold,
                                               count_mode=self.count_mode).fit(dtrajs).fetch_model()

        # Estimate transition matrix using re-sampling:
        if self.rank_Ct == 'bootstrap_counts':
            Ceff_full = effective_count_matrix(dtrajs_lag, self.lagtime)
            Ceff = submatrix(Ceff_full, count_model.active_set)
            smean, sdev = bootstrapping_count_matrix(Ceff, nbs=self.nbs)
        else:
            smean, sdev = bootstrapping_dtrajs(dtrajs_lag, self.lagtime, count_model.n_states, nbs=self.nbs,
                                               active_set=count_model.active_set)
        # Estimate two step count matrices:
        C2t = twostep_count_matrix(dtrajs, self.lagtime, count_model.n_states)
        # Rank decision:
        rank_ind = rank_decision(smean, sdev, tol=self.tol_rank)
        # Estimate OOM components:
        Xi, omega, sigma, eigenvalues = oom_components(count_model.count_matrix.toarray(), C2t, rank_ind=rank_ind,
                                             lcc=count_model.active_set)
        # Compute transition matrix:
        P, lcc_new = equilibrium_transition_matrix(Xi, omega, sigma, reversible=self.reversible)

        # Update active set and derived quantities:
        # todo: dont re-initialize, this is only due to active set (see bhmm impl)
        if lcc_new.size < count_model.n_states:
            assert isinstance(count_model, TransitionCountModel)
            count_model.__init__(self.lagtime, active_set=count_model.active_set[lcc_new],
                                 physical_time=count_model.physical_time, connected_sets=count_model.connected_sets,
                                 count_matrix=count_model.count_matrix)
            warnings.warn("Caution: Re-estimation of count matrix resulted in reduction of the active set.")

        # update models
        count_model.C2t = C2t

        self._model = KoopmanReweightedMSM(
            transition_matrix=P,
            eigenvalues_OOM=eigenvalues,
            sigma=sigma,
            omega=omega,
            count_model=count_model,
            oom_components=Xi
        )

        return self
