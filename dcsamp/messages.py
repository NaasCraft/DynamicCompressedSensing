import logging

import numpy

from dcsamp.utils import trim_array


class Messages:
    """Class used to store and compute messages in the DCS framework.

    Note that we will not handle message approximation in this class, but use
    instead a dedicated object.
    """
    REAL_MSGS = (
        'lambda_fwd', 'lambda_bwd',  # from h(t), h(t+1) to s(t)
        'kappa_fwd', 'kappa_bwd',  # (vars) from d(t), d(t+1) to theta(t)
        'pi_in', 'psi_in',  # from s(t), theta(t) to f(t)
        'pi_out', 'psi_out',  # from f(t) to s(t), theta(t)
    )
    COMPLEX_MSGS = (
        'eta_fwd', 'eta_bwd',  # (means) from d(t), d(t+1) to theta(t)
        'xi_in', 'xi_out',  # from f(t) to s(t), theta(t)
    )

    def __init__(self,
                 y: numpy.ndarray,
                 N: int,
                 T: int,
                 params: 'ModelParams'):
        self._y = y
        self._N = N
        self._T = T
        self.params = params
        self.logger = logging.getLogger('MSG')

        for message in self.COMPLEX_MSGS:
            setattr(self, message, numpy.zeros((N, T+1), dtype=numpy.complex_))
        for message in self.REAL_MSGS:
            setattr(self, message, numpy.zeros((N, T+1), dtype=numpy.float_))

        # Initialize first timestep of forward messages using model parameters
        self.lambda_fwd[:, 0] = params.lambda_
        self.eta_fwd[:, 0] = params.zeta
        self.kappa_fwd[:, 0] = params.kappa

        # Initialize all backward messages to meaningless values
        self.lambda_bwd = 0.5 * numpy.ones(self.lambda_bwd.shape)
        # self.eta_bwd is already at zero
        self.kappa_bwd = numpy.inf * numpy.ones(self.kappa_bwd.shape)

    def get_lambda_est(self, t: int) -> numpy.ndarray:
        pi = self.pi_out[:, t]
        lam_f, lam_b = self.lambda_fwd[:, t], self.lambda_bwd[:, t]

        return (
            pi * lam_f * lam_b
        ) / (
            (1 - pi) * (1 - lam_f) * (1 - lam_b) + pi * lam_f * lam_b
        )

    def compute_into_step(self, t: int):
        """Message passing : (into) step, from s(t) and theta(t) to f(t)."""
        lam_f, lam_b = self.lambda_fwd[:, t], self.lambda_bwd[:, t]
        self.debug(lam_f, '[into t={}] lambda_fwd'.format(t))
        self.debug(lam_b, '[into t={}] lambda_bwd'.format(t))
        self.pi_in[:, t] = lam_f if t == self._T else (
            (lam_f * lam_b) / (1 - lam_f - lam_b + 2 * lam_f * lam_b)
        )
        trim_array(self.pi_in[:, t], 0, 1)
        self.debug(self.pi_in[:, t], '[into t={}] pi_in'.format(t))

        kap_f, kap_b = self.kappa_fwd[:, t], self.kappa_bwd[:, t]
        self.debug(kap_f, '[into t={}] kappa_fwd'.format(t))
        self.debug(kap_b, '[into t={}] kappa_bwd'.format(t))
        self.psi_in[:, t] = kap_f if t == self._T else (
            1 / (1 / kap_f + 1 / kap_b)
        )
        self.debug(self.psi_in[:, t], '[into t={}] psi_in'.format(t))

        eta_f, eta_b = self.eta_fwd[:, t], self.eta_bwd[:, t]
        self.debug(eta_f, '[into t={}] eta_fwd'.format(t))
        self.debug(eta_b, '[into t={}] eta_bwd'.format(t))
        self.xi_in[:, t] = self.eta_fwd[:, t] if t == self._T else (
            self.psi_in[:, t] * (eta_f / kap_f + eta_b / kap_b)
        )
        self.debug(self.xi_in[:, t], '[into t={}] xi_in'.format(t))

    def compute_across_step(self, t_from: int, t_to: int):
        self.logger.debug('Running (across) from %d to %d', t_from, t_to)
        if t_to == self._T + 1:
            # No forward messages can go to frame T+1
            # Instead, we use this step to perform the backward update,
            # from h(T) to s(T), and from d(T) to theta(T)
            t_to = self._T

        if t_to == 0:
            # Prior values do not receive messages
            return

        if t_to - t_from == 1:
            direction = 'fwd'
        elif t_to - t_from == 0:
            direction = 'bwd'
        else:
            raise RuntimeError("Invalid timesteps provide for (across) step.")

        # store variables
        lam_from, lam_to = (
            getattr(self, 'lambda_{}'.format(direction))[:, t]
            for t in (t_from, t_to)
        )
        eta_from, eta_to = (
            getattr(self, 'eta_{}'.format(direction))[:, t]
            for t in (t_from, t_to)
        )
        kappa_from, kappa_to = (
            getattr(self, 'kappa_{}'.format(direction))[:, t]
            for t in (t_from, t_to)
        )
        pi = self.pi_out[:, t_from]
        self.debug(pi, '[across t={}] pi'.format(t_from))
        psi = self.psi_out[:, t_from]
        self.debug(psi, '[across t={}] psi'.format(t_from))
        xi = self.xi_out[:, t_from]
        self.debug(xi, '[across t={}] xi'.format(t_from))

        # early computation of variance term
        theta_var = 1 / (1 / kappa_from + 1 / psi)

        # store useful model parameters
        p10 = self.params.p10
        p01 = self.params.p01
        alpha = self.params.alpha
        rho = self.params.rho
        zeta = self.params.zeta

        if direction == 'fwd':
            # Update lambda
            lam_to[:] = (
                p10 * (1 - lam_from) * (1 - pi) + (1 - p01) * lam_from * pi
            ) / (
                (1 - lam_from) * (1 - pi) + lam_from * pi
            )

            # Update eta
            eta_to[:] = (
                (1 - alpha) * theta_var * (
                    (eta_from / kappa_from) + (xi / psi)
                ) + alpha * zeta
            )

            # Update kappa
            kappa_to[:] = (1 - alpha) ** 2 * theta_var + alpha ** 2 * rho

        elif t_to - t_from == 0:  # backward direction
            if t_to == self._T:  # terminal node with special updates
                # Update lambda
                lam_to[:] = (p01 * (1 - pi) + (1 - p01) * pi) / \
                    ((1 - p10 + p01) * (1 - pi) + (1 - p01 + p10) * pi)

                # Update eta
                eta_to[:] = (1 / (1 - alpha)) * (xi - alpha * zeta)

                # Update kappa
                kappa_to[:] = (1 / (1 - alpha) ** 2) * (alpha ** 2 * rho + psi)

            else:  # non-terminal node, 0 < t < T
                # Update lambda
                lam_to[:] = (
                    p01 * (1 - lam_from) * (1 - pi) + (1 - p01) * lam_from * pi
                ) / (
                    (1 - pi) + (1 - p01 + p10) * lam_from * pi
                )

                # Update eta
                eta_to[:] = (1 / (1 - alpha)) * (
                    theta_var * (xi / psi + eta_from / kappa_from) -
                    alpha * zeta)

                # Update kappa
                kappa_to[:] = (1 / (1 - alpha) ** 2) * (
                    theta_var + alpha ** 2 * rho)

    def debug(self, value, name: str=''):
        if isinstance(value, numpy.ndarray):
            self.logger.debug(
                "Array %sof shape %r (min: %r, max: %r)",
                name + ' ', value.shape, value.min(), value.max()
            )
        else:
            self.logger.debug("Debug %s = %r", name, value)
