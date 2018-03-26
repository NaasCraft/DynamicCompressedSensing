"""The core algorithm definition for DCS-AMP.

TODO: remove article implementation details and add specific presentation of
      this Python implem

From the original article, the main sources are:
    - classdefs of Model parameters and Run options
    - `sp_multi_frame_fxn.m` : the main algorithm
    - `sp_frame_fxn.m`: (within) computations
    - `sp_timestep_fxn.m`: (across) messages
    - `parameter_update_fxn.m`: EM parameter updates

From `sp_multi_frame_fxn.m`, the algorithm needs:
        - Model parameters:
            - lambda,
            - p01,
            - eta,
            - kappa,
            - alpha,
            - sig2e
        - Execution options:
            - max and min smoothing iterations
              (-1 if performing in filtering mode)
            - inner AMP iterations per pass
            - whether to update (with EM) the model parameters or not
            - (if update) which parameters to update (if we want to fix any)
            - (if update) allow to control learning of parameters for certain
              groups of coefficients (by default, same values for all)
            - epsilon (<< 1) for approximation of (out) messages
            - tau (~ 1 but < 1), used to decide which Gaussian component to use
              in (out) messages

    And it should output:
        - MMSE estimates of X and its variances, as well as the probabilities
          for support activation (s(t))
        - Internal state of the algorithm, so one can warm-start again
"""
import logging
from typing import Union, Tuple, Optional, Sequence

import numpy

from dcsamp.amp import AMP
from dcsamp.exceptions import InitError
from dcsamp.parameters import ModelParams
from dcsamp.messages import Messages
from dcsamp.updater import Updater
from dcsamp.utils import Timer


LOG = logging.getLogger(__name__)


class DCS_AMP:
    """Main algorithm for the Dynamic Compressed Sensing via Approximate
    Message Passing framework."""

    def __init__(
        self,
        mode: str='smoothing',
        smoothing_iterations: Union[int, Tuple[int, int]]=5,
        inner_iterations: int=25,
        epsilon: float=1e-4,
        tau: float=(1 - 1e-4),
        updater: Optional[Updater]=None
    ):
        """Instantiate the DCS-AMP algorithm with its execution options.

        This constructor handles the validation of these options.

        Parameters:
            - `mode`: either 'filtering' or 'smoothing'
            - `smoothing_iterations`: either the maximum number of smoothing
              iterations to run, or a tuple of both the minimum and maximum
            - `inner_iterations`: maximum number of AMP (or ABP) iterations to
              run, at each forward/backward pass
            - `epsilon`: a small float for approximating outgoing messages
            - `tau`: a float slightly smaller than 1 used to decide which
              Gaussian component to use in outgoing messages
            - `updater`: an instance of an `Updater` child class (e.g.
              EMUpdater), used to adjust the model's parameters during the run
        """
        # Operating mode
        if mode not in ('smoothing', 'filtering'):
            raise InitError('Bad mode provided: {}'.format(mode))

        self.mode = mode

        # Number of smoothing iterations (one iteration = forward + backward)
        if self.mode == 'filtering':
            # Will not consider backward pass during filtering
            self.max_smooth_iter = 1
            self.min_smooth_iter = 0
        elif isinstance(smoothing_iterations, int):
            self.max_smooth_iter = smoothing_iterations
            self.min_smooth_iter = 0
        else:
            # Accept input of both min and max
            try:
                self.max_smooth_iter, self.min_smooth_iter = \
                    smoothing_iterations
            except ValueError:
                raise InitError('Bad smoothing_iterations provided: {}.'
                                .format(smoothing_iterations))
            else:
                if self.min_smooth_iter > self.max_smooth_iter:
                    self.min_smooth_iter = self.max_smooth_iter
                    LOG.warn(
                        'Min iterations (%(min)d) exceeded max (%(max)d),'
                        'thus setting both at %(max)d.',
                        min=self.min_smooth_iter,
                        max=self.max_smooth_iter
                    )

        # Inner AMP/ABP iterations
        self.inner_iter = inner_iterations

        # Check the validity of iteration numbers provided (positive integers)
        self._check_iterations()

        # Epsilon and Tau
        if not (epsilon > 0 and epsilon <= 1e-3):
            raise InitError("Invalid epsilon provided (0 < eps <= 1e-3).")
        self.epsilon = epsilon

        if not (tau >= 0 and tau <= 1):
            raise InitError("Invalid tau provided (0 <= tau <= 1).")
        self.tau = tau

        # Model parameters "Updater"
        self.updater = updater

        # Allow to store multiple timers for later comparison
        self.timers = {}

    @property
    def estimates(self):
        return {
            'x_estimates': self.x_estimates,
            'v_estimates': self.v_estimates,
            'lambda_estimates': self.lambda_estimates
        }

    def run(self, y: Sequence[numpy.ndarray], A: Sequence[numpy.ndarray],
            run_id: str='', **kwargs):
        """Recover estimates of a sparse signal from a set of measurements.

        The algorithm can be decomposed into the following steps:

            - 1 - Variables initialization

            - 2 - Forward pass: for each t in 0..T, compute

                - a - (into) messages
                    1. From h(t) and h(t+1) to s(t)
                    2. From d(t) and d(t+1) to theta(t)
                    3. From theta(t) and s(t) to f(t)

                - b - (within) AMP ("equalization")
                    1. From f(t) to x(t)
                    2. From x(t) to g(t) and reverse (approximation)
                    3. From x(t) back to f(t)

                - c - (out) messages
                    1. From f(t) to theta(t) and s(t)

                - d - (across) messages
                    1. From s(t) to h(t) and h(t+1)
                    2. From d(t) and d(t+1) to theta(t)

                - e - EM parameter tuning (only if filtering)
                    #NOTE probably not efficient

            - 3 - Backward pass (opt. if filtering)
                Same principles as the forward pass, t goes from T-1 to 0.

            - 4 - EM parameter tuning update (if smoothing)

            - 5 -> ... - Loop back to 2, until the smoothing iterations has
                reached its maximum allowed, or the average residual energy
                cannot be distinguished from the configured AWGN.

        The algorithm assumes that, for each t:
            y(t) = A(t)*x(t) + e(t)

        where:
            - x(t) is a N-length vector
            - e(t) is a AWGN, of variance `sig2e`
            - y(t) is a M(t)-length vector
            - A(t) is a (M(t) x N)-matrix

        Parameters:
            - `y`: sequence of measurements vectors y(t)
            - `A`: sequence of measurement matrices A(t)
        """
        # Store internal variables
        self._y = y
        self._A = A
        self._T = len(y) - 1
        self._N = A[0].shape[1]
        self._M = [y_t.shape[0] for y_t in y]
        self.last_iter = False

        # Initialize a timer
        _timer = Timer()
        if not run_id:
            run_id = 'timer-{}'.format(len(self.timers))
        self.timers[run_id] = _timer
        self._timer = _timer
        # Start global run countdown
        self._timer.tic('global')

        # Check compatibilty between y(t) and A(t)
        assert all(
            y_t.shape[0] == A_t.shape[0]
            for y_t, A_t in zip(y, A)
        ), "Provided measurements `y` and `A` are shape-incoherent."

        with self._timer.time('init'):
            self.initialize_variables(**kwargs)

        for k in range(self.max_smooth_iter):
            with self._timer.time('forward-{}'.format(k)):
                self.run_forward_pass(k)

            if self.mode == 'filtering':
                self._timer.toc('global')
                return self.estimates

            with self._timer.time('backward-{}'.format(k)):
                self.run_backward_pass(k)

            if k >= 2:
                # NOTE: from the article's implementation, it's suggested to
                #       wait before running EM update to avoid poor
                #       initialization.
                with self._timer.time('update-{}'.format(k)):
                    self.update_parameters(k)

            if self.should_complete(k):
                break

        self._timer.toc('global')
        return self.estimates

    def initialize_variables(self, **kwargs):
        """Create placeholders and initialize variables for the algorithm."""
        self.parameters = ModelParams(N=self._N, **kwargs)
        if 'siggen' in kwargs:
            self.parameters.from_signal_gen(kwargs['siggen'], kwargs['sig2e'])

        self.messages = Messages(
            y=self._y, N=self._N, T=self._T, params=self.parameters
        )
        self.amp = AMP(
            y=self._y, A=self._A, N=self._N, T=self._T, params=self.parameters,
            n_iter=self.inner_iter, tau=self.tau, eps=self.epsilon
        )

        self.x_estimates = [None for _ in range(self._T + 1)]
        self.v_estimates = [None for _ in range(self._T + 1)]
        self.lambda_estimates = [None for _ in range(self._T + 1)]

    def run_forward_pass(self, k: int):
        """Perform a forward pass on the factor graph."""
        self._timer.tic('forward-{}'.format(k))
        for t in range(self._T + 1):
            self.messages.compute_into_step(t)

            # (within) messages computation
            self.amp.run_loop(t,
                              pi=self.messages.pi_in[:, t],
                              psi=self.messages.psi_in[:, t],
                              xi=self.messages.xi_in[:, t])

            # Store `x` estimate from AMP state
            self.x_estimates[t] = self.amp.x_estimate

            # Extract (out) messages from AMP state
            self.messages.pi_out[:, t] = self.amp.pi_out
            self.messages.psi_out[:, t] = self.amp.psi_out
            self.messages.xi_out[:, t] = self.amp.xi_out

            self.messages.compute_across_step(t, t+1)

            if self.mode == 'filtering':
                self.update_parameters(k, t)

            if self.mode == 'filtering' or t == self._T:
                # Only save variances and support posterior probabilities
                # in forward pass if filtering or at last timestep.
                self.v_estimates[t] = self.amp.v_estimate
                self.lambda_estimates[t] = self.messages.get_lambda_est(t)

        self._timer.toc('forward-{}'.format(k))
        self.log_state('forward', k)

    def run_backward_pass(self, k: int):
        """Perform a backward pass on the factor graph."""
        self._timer.tic('backward-{}'.format(k))
        for t in range(self._T - 1, -1, -1):
            self.messages.compute_into_step(t)

            # (within) frame
            self.amp.run_loop(t,
                              pi=self.messages.pi_in[:, t],
                              psi=self.messages.psi_in[:, t],
                              xi=self.messages.xi_in[:, t])

            # Store `x` estimate from AMP state
            self.x_estimates[t] = self.amp.x_estimate

            # Extract (out) messages from AMP state
            # TODO: careful, maybe need to index in `t`
            self.messages.pi_out[:, t] = self.amp.pi_out
            self.messages.psi_out[:, t] = self.amp.psi_out
            self.messages.xi_out[:, t] = self.amp.xi_out

            self.messages.compute_across_step(t, t)

            if k == self.max_smooth_iter or self.last_iter:
                self.v_estimates[t] = self.amp.v_estimate
                self.lambda_estimates[t] = self.messages.get_lambda_est(t)

        self._timer.toc('backward-{}'.format(k))
        self.log_state('backward', k)

    def update_parameters(self, k: int, t: Optional[int]=None):
        """Calls the self.updater update method, if available.

        If we decide to test the algorithm, we can generate signals from some
        set of parameters, and provide them to the model while never updating
        them.
        """
        self.updater.update(
            parameters=self.parameters,
            amp=self.amp,
            messages=self.messages,
            iteration=k, filter_step=t
        )

    def should_complete(self, k: int) -> bool:
        """Check if the algorithm should stop.

        Uses a flag to actually set the termination at the step following the
        first to statisfy condition on residual energy.
        """
        if self.last_iter:
            return True

        if k <= self.min_smooth_iter:
            # Return early to avoid useless computation
            return False

        avg_residual_energy = self.evaluate()
        self.debug(avg_residual_energy, 'avg_res_en')
        noise_energy = sum(self._M) * (
            self.parameters.sig2e.mean() / (self._T + 1))
        self.debug(self.parameters.sig2e, 'sig2e')
        self.debug(noise_energy, 'noise_energy')
        if avg_residual_energy < noise_energy:
            # Set the flag up
            self.last_iter = True

        return False

    def evaluate(self) -> float:
        """Compute the average residual energy for the current estimates."""
        return numpy.sum([
            numpy.linalg.norm(y_t - A_t.dot(x_est_t)) ** 2
            for y_t, A_t, x_est_t in zip(self._y, self._A, self.x_estimates)
        ]) / (self._T + 1)

    def log_state(self, direction: str, k: int):
        LOG.info(
            "DCS-AMP: {direction} pass - iteration {k}/{max_iter}\n"
            "\tTotal elapsed time: {total_time:.2f}s\n"
            "\tPass duration: {pass_time:.2f}s\n"
            "\tAvg. residual energy: {avg_res_en:.5f}\n"
            "\n\n".format(
                direction=direction, k=k, max_iter=self.max_smooth_iter,
                total_time=self._timer.times.get('global', [-4.2])[-1],
                pass_time=self._timer.times.get(
                    '{}-{}'.format(direction, k), [-4.2])[-1],
                avg_res_en=self.evaluate()
            )
        )

    def debug(self, value, name: str=''):
        if isinstance(value, numpy.ndarray):
            LOG.debug(
                "Array %sof shape %r (min: %r, max: %r)",
                name + ' ', value.shape, value.min(), value.max()
            )
        else:
            LOG.debug("Debug %s = %r", name, value)

    def _check_iterations(self):
        to_check = ('max_smooth_iter', 'min_smooth_iter', 'inner_iter')
        for key in to_check:
            attr = getattr(self, key)
            if not isinstance(attr, int) and attr >= 0:
                raise InitError("Invalid number of iterations: {}={}"
                                .format(key, attr))
