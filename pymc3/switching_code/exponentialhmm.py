from scipy.stats import expon
import numpy as np
from hmmlearn.base import BaseHMM
from sklearn.utils import check_random_state

# If you want to implement other emission probability (e.g. Poisson), 
# you have to implement a new HMM class by inheriting the _BaseHMM 
# and overriding the methods 
# __init__, 
# _compute_log_likelihood, 
# _set and 
# _get for additional parameters, 
# _initialize_sufficient_statistics, 
# _accumulate_sufficient_statistics and 
# _do_mstep

class BaseExponentialHMM(BaseHMM):

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "l": nc * nf,
        }

    def _compute_likelihood(self, X):
        probs = np.empty((len(X), self.n_components))
        for c in range(self.n_components):
            probs[:, c] = expon.pdf(X, loc=0, scale = 1/self.lambdas_[c]).prod(axis=1)
        return probs

    def _compute_log_likelihood(self, X):
        logprobs = np.empty((len(X), self.n_components))
        for c in range(self.n_components):
            logprobs[:, c] = expon.logpdf(X, loc=0, scale = 1/self.lambdas_[c]).sum(axis=1)
        return logprobs

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, lattice,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, obs, lattice, posteriors, fwdlattice, bwdlattice)
        if 'l' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += posteriors.T @ obs

    def _generate_sample_from_state(self, state, random_state):
        return random_state.exponential(1/self.lambdas_[state])


class ExponentialHMM(BaseExponentialHMM):
    """
    Hidden Markov Model with Exponential emissions.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    lambdas_ : array, shape (n_components, n_features)
        The rate parameters for each
        feature in a given state.
    """

    def __init__(self, n_components=1, startprob_prior=1.0,
                 transmat_prior=1.0, lambdas_prior=0.0,
                 lambdas_weight=0.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stl", init_params="stl",
                 implementation="log"):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        lambdas_prior, lambdas_weight : array, shape (n_components,), optional
            The gamma prior on the lambda values using alpha-beta notation,
            respectivley. If None, will be set based on the method of
            moments.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any
            combination of 's' for startprob, 't' for transmat, and 'l' for
            lambdas.  Defaults to all parameters.

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.
        """
        BaseHMM.__init__(self, n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior,
                         algorithm=algorithm,
                         random_state=random_state,
                         n_iter=n_iter, tol=tol, verbose=verbose,
                         params=params, init_params=init_params,
                         implementation=implementation)
        self.lambdas_prior = lambdas_prior
        self.lambdas_weight = lambdas_weight

    def _init(self, X, lengths=None):
        super()._init(X, lengths)
        self.random_state = check_random_state(self.random_state)

        mean_X = X.mean()
        var_X = X.var()

        if self._needs_init("l", "lambdas_"):
            # initialize with method of moments based on X
            self.lambdas_ = self.random_state.gamma(
                shape=mean_X**2 / var_X,
                scale=var_X / mean_X,  # numpy uses theta = 1 / beta
                size=(self.n_components, self.n_features))

    def _check(self):
        super()._check()

        self.lambdas_ = np.atleast_2d(self.lambdas_)
        n_features = getattr(self, "n_features", self.lambdas_.shape[1])
        if self.lambdas_.shape != (self.n_components, n_features):
            raise ValueError(
                "lambdas_ must have shape (n_components, n_features)")
        self.n_features = n_features

    def _do_mstep(self, stats):

        super()._do_mstep(stats)

        if 'l' in self.params:
            # Based on: Hyvönen & Tolonen, "Bayesian Inference 2019"
            # section 3.2
            # https://vioshyvo.github.io/Bayesian_inference
            alphas, betas = self.lambdas_prior, self.lambdas_weight
            n = stats['post'].sum()
            y_bar = stats['obs'] / stats['post'][:, None]
            # the same as kappa notation (more intuitive) but avoids divide by
            # 0, where:
            # kappas = betas / (betas + n)
            # self.lambdas_ = kappas * (alphas / betas) + (1 - kappas) * y_bar
            #self.lambdas_ = (alphas + n * y_bar) / (betas + n)
            self.lambdas_ = (alphas + n) / (betas + n*y_bar)

from sklearn.utils.validation import check_array
import logging
_log = logging.getLogger(__name__)

class FixedExponentialHMM(ExponentialHMM):
    def __init__(self, *args, fixed_transitions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_transitions = fixed_transitions

    # def fit(self, X, lengths=None):
    #     super().fit(X, lengths)
    #     print(self.transmat_)
    #     if self.fixed_transitions is not None:
    #         self.transmat_ = self.fix_transmat(self.transmat_, self.fixed_transitions)
    #     print(self.transmat_)

    #     return self


    def fit(self, X, lengths=None):
        """
        Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)

        if lengths is None:
            lengths = np.asarray([X.shape[0]])

        self._init(X, lengths)
        self._check()
        self.monitor_._reset()

        for iter in range(self.n_iter):
            # print(iter)
            stats, curr_logprob = self._do_estep(X, lengths)

            # Compute lower bound before updating model parameters
            lower_bound = self._compute_lower_bound(curr_logprob)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.

            self._do_mstep(stats)
            # print(self.transmat_)

            if self.fixed_transitions is not None:
                self.transmat_ = self.fix_transmat(self.transmat_, self.fixed_transitions)
            # print(self.transmat_)

            self.monitor_.report(lower_bound)

            # if self.monitor_.converged:
            #     break

            if (self.transmat_.sum(axis=1) == 0).any():
                _log.warning("Some rows of transmat_ have zero sum because no "
                             "transition from the state was ever observed.")
                
        return self


    def fix_transmat(self, transmat, fixed_transitions):
        fixed_transmat = transmat.copy()
        for (i, j, value) in fixed_transitions:
            fixed_transmat[i, j] = value
        # Normalize the rows to sum to 1
        fixed_transmat /= fixed_transmat.sum(axis=1, keepdims=True)
        return fixed_transmat

