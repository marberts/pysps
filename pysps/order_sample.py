"""
Order Sampling
"""

from typing import Callable

import numpy as np
import numpy.typing as npt

from pysps.inclusion_prob import InclusionProb


def _igpd(shape: float) -> Callable[[npt.ArrayLike], np.ndarray]:
    """
    Inverse of the generalized Pareto distribution.
    """

    if shape == 0.0:
        return lambda x: -np.log(1 - x)
    else:
        return lambda x: (1 - (1 - x)**shape) / shape
       

class OrderSample:
    """
    Order sampling scheme with fixed distribution shape.

    Parameters
    ----------
    pi : InclusionProb
        Inclusion probabilities for units in the population.
    prn : ArrayLike, optional
        Permanent random numbers. Should be a flat array of values, the
        same length as x, distributed uniform between 0 and 1. The 
        default draws a sample without permanent random numbers.
    shape : float, optional
        Shape parameter for the generalized Pareto distribution that is
        used as the fixed order distribution shape.

        shape=1  => Sequential Poisson sampling (the default)
        shape=0  => Successive sampling
        shape=-1 => Pareto order sampling

    Returns
    -------
    OrderSample
        Indices for units in the sample.

    References
    ----------
    Matei, A., and Tillé, Y. (2007). Computational aspects of order πps
        sampling schemes. Computational Statistics & Data Analysis, 51:
        3703-3717.

    Ohlsson, E. (1998). Sequential Poisson Sampling. Journal of 
        Official Statistics, 14(2): 149-162.
    
    Rosén, B. (1997). On sampling with probability proportional to 
        size. Journal of Statistical Planning and Inference, 62(2): 
        159-191.

    Examples
    --------
    >>> x = np.arange(10)
    >>> pi = InclusionProb(x, 6)

    # Draw a sequential Poisson sample using permanent random numbers
    >>> prn = np.random.default_rng(54321).uniform(size=10)
    >>> sample = OrderSample(pi, prn)
    >>> sample.units
    array([3, 4, 5, 7, 8, 9], dtype=int64)

    # Get the design weights
    >>> sample.weights
    array([2.33333333, 1.75, 1.4, 1.0, 1.0, 1.0])

    # Units 0 to 2 are take-some units...
    >>> sample.take_some
    array([0, 1, 2], dtype=int64)

    # ... and units 3 to 5 are take-all units
    >>> sample.take_all
    array([3, 4, 5], dtype=int64)
    
    # Draw a Pareto order sample using the same permanent random numbers
    >>> OrderSample(pi, prn, shape=-1).units
    array([3, 5, 6, 7, 8, 9], dtype=int64)
    """
    def __init__(self,
                 pi: InclusionProb,
                 prn: npt.ArrayLike | None = None, *,
                 shape: float = 1.0) -> np.ndarray:
        if prn is None:
            u = np.random.default_rng().uniform(size=len(pi))
        else:
            u = np.asfarray(prn).ravel()
            if len(u) != len(pi):
                raise ValueError("pi and prn must be the same length")
            if np.any(u <= 0.0) or np.any(u >= 1.0):
                raise ValueError(
                    "all elements of prn must be in (0, 1)"
                )
            
        shape = float(shape)
        n_ts = pi._n - len(pi._ta)
        if n_ts == 0:
            self._units = pi.take_all
        else:
            dist = _igpd(shape)
            xi = dist(u[pi._ts]) / dist(pi._values[pi._ts])
            # Sorting should be stable so that ties resolve in order.
            keep = np.argsort(xi, kind="stable")[:n_ts]
            res = np.concatenate((pi._ta, pi._ts[keep]))
            res.sort()
            self._units = res

        ta = np.isin(self._units, pi._ta, assume_unique=True)
        self._ta = np.flatnonzero(ta)
        self._ts = np.flatnonzero(~ta)
        self._pi = pi
        self._prn = u
        self._shape = shape

    @property
    def units(self) -> np.ndarray:
        """
        Indices for units in the sample.
        """
        return self._units.copy()
    
    @property
    def weights(self) -> np.ndarray:
        """
        Design weights for units in the sample.
        """
        return 1 / self._pi._values[self._units]
    
    @property
    def take_all(self) -> np.ndarray:
        """
        Take all units in the sample.
        """
        return self._ta.copy()
    
    @property
    def take_some(self) -> np.ndarray:
        """
        Take some units in the sample.
        """
        return self._ts.copy()
    
    @property
    def prn(self) -> np.ndarray:
        """
        Random numbers used for drawing the sample.
        """
        return self._prn.copy()
    
    def __len__(self) -> int:
        return len(self._units)
    
    def __repr__(self) -> str:
        pi = repr(self._pi)
        prn = repr(self._prn)
        return f"OrderSample({pi}, {prn}, shape={self._shape})"
    
    def __str__(self) -> str:
        return str(self._units)
