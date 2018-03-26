import logging
from typing import Sequence

import numpy

from dcsamp.utils import trim_array


class AMP:
    """Class used to manipulate approximate messages in the AMP subgraph."""

    MAX_PROB = 1 - 1e-15
    MIN_PROB = 1 - MAX_PROB
    TOLERANCE = 1e-5

    def __init__(
        self,
        y: Sequence[numpy.ndarray],
        A: Sequence[numpy.ndarray],
        N: int,
        T: int,
        params: 'ModelParams',
        **options
    ):
        """Initialize variables with defaults."""
        self._y = y
        self._A = A
        self._N = N
        self._T = T
        self.sig2e = params.sig2e

        self.mu = numpy.zeros((N, T+1), dtype=numpy.complex_)
        self.v = numpy.zeros((N, T+1), dtype=numpy.complex_)
        self.c = 100 * params.kappa.mean() * (
            numpy.ones((T+1), dtype=numpy.complex_))
        # TODO: is this c init useful ?
        self.z = y

        self.eps = options.get('eps', 1e-5)
        self.tau = options.get('tau', None)
        self.n_iter = options.get('n_iter', 25)

        self.logger = logging.getLogger('AMP')

    def run_loop(self, t: int,
                 pi: numpy.ndarray, psi: numpy.ndarray, xi: numpy.ndarray):
        self.logger.debug('Running AMP at t={}'.format(t))
        # Declare runtime variables and constants
        c = 100 * psi.mean()
        z = self.z[t]
        y = self._y[t]
        M = y.size
        mu = self.mu[:, t]
        A = self._A[t]
        gam_A = (1 - pi) / pi  # Factor for computing gamma

        should_stop = False

        for i in range(self.n_iter):
            # Compute phi
            phi = A.T.dot(z) + mu

            # Compute gamma
            gam_B = psi + c
            gam_C = c / psi
            gam_exp = numpy.exp(
                -((
                    numpy.abs(phi + gam_C * xi) ** 2
                    - gam_C * (1 + gam_C) * numpy.abs(xi) ** 2
                  ) / (
                    gam_C * gam_B
                  ))
            )
            trim_array(gam_exp)
            self.debug(gam_exp, 'gamma_exp')
            gamma = gam_A * (gam_B / c) * gam_exp

            # Compute mu
            gamma_redux = 1 / (1 + gamma)
            old_mu = mu.copy()
            mu[:] = gamma_redux * (phi * psi + c * xi) / gam_B

            # Compute v
            v = gamma_redux * c * psi / gam_B + gamma * numpy.abs(mu) ** 2

            if should_stop or i == self.n_iter - 1:
                # Algorithm should stop now, compute outgoing messages
                self._pi_out = 1 / (1 + (gam_B / c) * gam_exp)
                trim_array(self._pi_out, self.MIN_PROB, self.MAX_PROB)

                self._xi_out = numpy.where(pi < self.tau, phi / self.eps, phi)
                self._psi_out = numpy.where(pi < self.tau, c / self.eps ** 2, c)

                # also compute estimates for x and its variance
                self._x_est = mu
                self._v_est = v

                # then exit loop
                break

            # Compute f_prime, c, and z
            f_prime = v / c
            c = self.sig2e + (1 / M) * v.sum()
            z[:] = y - A.dot(mu) + (1 / M) * f_prime.sum() * z

            # Check progression
            progress = numpy.linalg.norm(old_mu - mu) ** 2 / self._N
            should_stop = progress < self.TOLERANCE and i > 1

    @property
    def x_estimate(self):
        return self._x_est

    @property
    def v_estimate(self):
        return self._v_est

    @property
    def pi_out(self):
        return self._pi_out

    @property
    def psi_out(self):
        return self._psi_out

    @property
    def xi_out(self):
        return self._xi_out

    def debug(self, value, name: str=''):
        if isinstance(value, numpy.ndarray):
            self.logger.debug(
                "Array %sof shape %r (min: %r, max: %r)",
                name + ' ', value.shape, value.min(), value.max()
            )
        else:
            self.logger.debug("Debug %s = %r", name, value)
