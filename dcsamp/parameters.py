from typing import Union, Any, Callable

import numpy

from dcsamp.signal_gen import SignalGen


class ModelParams:
    """Flexible class for handling the DCS-AMP model parameters."""
    def __init__(self, N: int, **kwargs):
        """Initialize parameters with placeholder values."""
        self._N = N

        for param in (
                '_lambda', '_p01', '_zeta', '_kappa', '_alpha', '_sig2e'):
            setattr(self, param, kwargs.get(param[1:], None))

    @property
    def lambda_(self):
        """Steady-state active support probability.

        P{s_n(t) = 1}
        """
        return self._lambda

    @lambda_.setter
    def lambda_(self, value: Union[float, numpy.ndarray]):
        self._set_param('_lambda', value, lambda v: v >= 0 and v < 1)

    @property
    def p01(self):
        """Active-to-inactive support probability

        P{s_n(t) = 0 | s_n(t-1) = 1}
        """
        return self._p01

    @p01.setter
    def p01(self, value: Union[float, numpy.ndarray]):
        self._set_param('_p01', value, lambda v: v >= 0 and v <= 1)

    @property
    def zeta(self):
        """Active amplitude mean.

        E{theta_n(t)|theta_n(t-1)}.

        Remember that:
        theta(t) = (1 - alpha)(theta(t-1) - zeta) + alpha*omega(t) + zeta
        """
        return self._zeta

    @zeta.setter
    def zeta(self, value: Union[float, numpy.ndarray]):
        self._set_param('_zeta', value)

    @property
    def kappa(self):
        """Active amplitude variance.

        V{theta_n(t)|theta_n(t-1)}

        See reminder in `eta` definition.
        Also see `rho` definition, since there may be confusion... TODO
        """
        return self._kappa

    @kappa.setter
    def kappa(self, value: Union[float, numpy.ndarray]):
        self._set_param('_kappa', value, lambda v: v > 0)

    @property
    def alpha(self):
        """Innovation rate.

        See reminder in `eta` definition.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: Union[float, numpy.ndarray]):
        self._set_param('_alpha', value, lambda v: v >= 0 and v < 1)

    @property
    def sig2e(self):
        """Noise (AWGN) variance.

        In y(t) = A(t)x(t) + e(t), we have e(t) ~ CN(0, sig2e).
        """
        return self._sig2e

    @sig2e.setter
    def sig2e(self, value: Union[float, numpy.ndarray]):
        self._set_param('_sig2e', value, lambda v: v >= 0)

    @property
    def p10(self):
        """Inactive-to-active support probability. (Computed)

        p01 = lambda * p01 / (1 - lambda)
        """
        return self.p01 * self.lambda_ / (1 - self.lambda_)

    @property
    def rho(self):
        """Variance of the amplitude driving noise. (Computed)

        From definition of kappa:
            rho = kappa / alpha**2

        From definition of sig2 (V{theta(t)}):
            rho = (2 - alpha) * sig2 / alpha
        """
        return (2 - self.alpha) * self.kappa / self.alpha

    def from_signal_gen(self, siggen: SignalGen, sig2e: float):
        self.lambda_ = siggen.lambda_
        self.p01 = siggen.p01
        self.zeta = siggen.zeta
        self.kappa = siggen.sigma2
        self.alpha = siggen.alpha
        self.sig2e = sig2e

    def _set_param(
        self,
        attr: str,
        value: Union[Any, numpy.ndarray],
        check: Callable=lambda _: True
    ):
        """Setter for model parameters.

        Parameters can be set either using a single value or a N-length vector.
        """
        if isinstance(value, numpy.ndarray):
            assert value.shape == (self._N,)
            assert all(check(v) for v in value)
            setattr(self, attr, value)
        else:
            assert check(value)
            setattr(self, attr, numpy.repeat(value, self._N))
