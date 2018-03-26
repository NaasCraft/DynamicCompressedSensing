"""Tool used to generate a signal according to the model chosen.

Also extracts measurements and parameters for the DCS-AMP framework to run
with.
"""
from typing import List, Tuple

import numpy


class SignalGen:
    def __init__(self, **params):
        """Initialize a set of signal parameters for later generation."""
        # Dimension of the true signal x
        self.N = params.get('N', 1024)

        # Dimension of the measurement vector y
        self.M = params.get('M', 256)

        # Number of timesteps
        self.T = params.get('T', 4)

        # Type of the random measurement matrix to generate
        #   (1) : normalized Gaussian matrix
        self.A_type = params.get('A_type', 1)

        # Active support probability
        self.lambda_ = params.get('lambda_', 0.08)  # high sparsity default

        # Amplitude mean
        self.zeta = params.get('zeta', 0)

        # Amplitude variance
        self.sigma2 = params.get('sigma2', 1)

        # Amplitude innovation rate
        self.alpha = params.get('alpha', 0.10)

        # Active-to-inactive transition probability
        self.p01 = params.get('p01', 0.10)

        # Desired signal-to-noise ratio, in dB
        self.desired_SNR = params.get('desired_SNR', 25)

    @property
    def rho(self):
        return (2 - self.alpha) * self.sigma2 / self.alpha

    @property
    def p10(self):
        return self.lambda_ * self.p01 / (1 - self.lambda_)

    def generate_signal(self) -> Tuple[numpy.ndarray]:
        s = numpy.zeros((self.N, self.T))
        support = [None for _ in range(self.T)]
        n_active = [None for _ in range(self.T)]

        # Generate initial support
        n_active[0] = numpy.random.binomial(self.N, self.lambda_)
        support[0] = numpy.random.choice(range(self.N), size=n_active[0])
        s[support[0], 0] = 1

        # Evolve support over time
        for t in range(1, self.T):
            draws = numpy.random.random(self.N)
            active_mask, inactive_mask = s[:, t-1] == 0, s[:, t-1] == 1

            deactivated = draws[active_mask] > self.p01
            activated = draws[inactive_mask] < self.p10

            s[active_mask, t] = 1 - deactivated.astype(int)
            s[inactive_mask, t] = activated.astype(int)

            support[t] = s[t] == 1
            n_active[t] = len(support[t])

        # Generate amplitude process (complex-valued)
        theta = numpy.zeros((self.N, self.T), dtype=numpy.complex)

        theta[:, 0] = self.zeta * numpy.ones(self.N)
        theta[:, 0] += numpy.sqrt(self.sigma2 / 2) * (
            numpy.random.randn(self.N, 2).dot([1, 1j])
        )
        for t in range(1, self.T):
            noise = numpy.random.randn(self.N, 2).dot([1, 1j])
            theta[:, t] = (
                (1 - self.alpha) * (theta[:, t - 1] - self.zeta) +
                self.alpha * numpy.sqrt(self.rho / 2) * noise + self.zeta
            )

        # True signal
        x = theta * s

        return x, theta, s

    def generate_measurements(
        self, x: numpy.ndarray
    ) -> Tuple[List[numpy.ndarray]]:
        """Generate measurement matrices and vectors from a given signal."""
        # Generate A matrices
        signal_power = 0
        A_list = []
        for t in range(self.T):
            if self.A_type == 1:
                # IID Gausian with unit-norm colums
                A = (
                    numpy.random.randn(self.M, self.N) +
                    1j * numpy.random.randn(self.M, self.N)
                ) / numpy.sqrt(2 * self.M)
                for n in range(self.N):
                    A[:, n] /= numpy.linalg.norm(A[:, n])
            else:
                raise ValueError("Invalid A_type: {}".format(self.A_type))

            A_list.append(A)
            signal_power += numpy.linalg.norm(A.dot(x[:, t])) ** 2

        # Extract noise variance for desired SNR
        sig2e = signal_power / (self.M * self.T) * 10 ** (-self.desired_SNR/10)

        # Generate noisy measurements
        y_list = []
        for t in range(self.T):
            e = numpy.sqrt(sig2e/2) * (
                numpy.random.randn(self.M, 2).dot([1, 1j]))
            y_list.append(
                A[t].dot(x[:, t]) + e
            )

        return y_list, A_list, sig2e
