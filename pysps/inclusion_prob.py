"""
Inclusion Probabilities
"""

import numpy as np
import numpy.typing as npt


def _pi(x: np.ndarray, n: int) -> np.ndarray:
    """Unbounded first-order inclusion probabilities."""
    if n == 0:
        return np.repeat(0.0, len(x))
    else:
        return x * (n / np.sum(x))


def _which_ta(x: np.ndarray,
              n: int,
              alpha: float) -> np.ndarray:
    """Indices for take-all units."""
    # Sorting should be stable. Sorting in reverse order then flipping
    # means ties resolve according to the order of x.
    ord = np.argsort(-x, kind="stable")
    possible_ta = np.flip(ord[:n])
    x_ta = x[possible_ta]
    p = x_ta * np.arange(1, n + 1) / (np.sum(x[ord[n:]]) + np.cumsum(x_ta))
    # p[k] < 1           => p[k + 1] >= p[k]
    # p[k] >= 1          => p[k + 1] >= 1
    # Therefore...
    # p[k] >= 1 - alpha  =>  p[k + 1] >= 1 - alpha
    return possible_ta[p >= 1.0 - alpha]


class InclusionProb:
    """
    Calculate first-order inclusion probabilities.

    Parameters
    ----------
    x : ArrayLike 
        Sizes for units in the population. Should be a flat array of 
        positive numbers.
    n : int
        Sample size.
    alpha : float, optional
        A number between 0 and 1 such that units with inclusion 
        probabilities greater than or equal to 1 - alpha are set to 
        1. The default is slightly larger than 0. See Ohlsson (1998)
        for details.
    cutoff : float, optional
        A number such that all units with size greater than or equal to
        cutoff get an inclusion probability of 1. The default does not
        apply a cutoff.

    Returns
    -------
    InclusionProb
        Inclusion probabilities for units in the population.

    References
    ----------
    Ohlsson, E. (1998). Sequential Poisson Sampling. Journal of 
        Official Statistics, 14(2): 149-162.

    Tillé, Y. (2006). Sampling Algorithms. Springer.

    Examples
    --------
    >>> x = np.arange(6)
    >>> pi = InclusionProb(x, 3)
    >>> pi
    InclusionProb(array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), 3)

    # Units 1-4 belong to the take-some stratum, and units 5 belongs to
    # the take-all stratum
    >>> pi.take_some
    array([1, 2, 3, 4], dtype=int64)
    >>> pi.take_all
    array([5], dtype=int64)

    # Calculate design weights for a PPS sampling scheme
    >>> 1 / pi.values
    array([inf, 5.0, 2.5, 1.66666667, 1.25, 1.0])
    """
    def __init__(self,
                 x: npt.ArrayLike,
                 n: int, *,
                 alpha: float = 0.001,
                 cutoff: float = np.Inf) -> None:     
        x = np.asfarray(x).flatten() # copy
        if np.any(x < 0.0):
            raise ValueError(
                "elements of x must be greater than or equal to 0"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("all elements of x must be finite")

        n = int(n)
        if n < 0:
            raise ValueError("n must be greater than or equal to 0")
        if n > np.sum(x > 0.0):
            raise ValueError(
                "n cannot be greater than the number of units with "
                "non-zero sizes in the population"
            )

        alpha = float(alpha)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be between 0 and 1")
        alpha = alpha + np.finfo("float").eps**0.5
        
        cutoff = float(cutoff)
        if cutoff <= 0.0:
            raise ValueError("cutoff must greater than 0")
        
        ta = np.flatnonzero(x >= cutoff)
        if len(ta) > n:
            raise ValueError(
                "n is too small to include all units above cutoff"
            )
        
        x[ta] = 0.0

        pi = _pi(x, n - len(ta))
        if np.any(pi >= 1 - alpha):
            ta2 = _which_ta(x, n - len(ta), alpha)
            x[ta2] = 0.0
            ta = np.concatenate([ta, ta2])
            pi = _pi(x, n - len(ta))

        ta.sort()
        pi[ta] = 1.0

        self._values = pi
        self._n = n
        self._ta = ta
        self._ts = np.flatnonzero(x)

    @property
    def values(self) -> np.ndarray:
        """Vector of inclusion probabilties."""
        return self._values.copy()
    
    @property
    def n(self) -> int:
        """Sample size."""
        return self._n
    
    @property
    def take_all(self) -> np.ndarray:
        """Take all units."""
        return self._ta.copy()
    
    @property
    def take_some(self) -> np.ndarray:
        """Take some units."""
        return self._ts.copy()
    
    # Inclusion probabilities are a fixed point for the inclusion
    # probability function.
    def __repr__(self) -> str:
        return f"InclusionProb({repr(self._values)}, {self.n})"
    
    def __str__(self) -> str:
        return str(self._values)
    
    def __len__(self) -> int:
        return len(self._values)
    