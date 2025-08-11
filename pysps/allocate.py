import math
from typing import Callable


def divisor_method(name: str) -> Callable[[int], float]:
    """Divisor function for highest averages apportionment method.

    Parameters
    ----------
    name : str
        Name of the divisor method, one of:
        "Jefferson/D'Hondt",
        "Webster/Sainte-Lague",
        "Imperiali",
        "Huntington-Hill",
        "Danish",
        "Adams",
        "Dean"

    Returns
    -------
    Callable[[int], float]
        The divisor function.
    """
    name = str(name).lower()
    if name == "jefferson/d'hondt":
        return lambda x: x + 1.0
    elif name == "webster/sainte-laguë":
        return lambda x: x + 0.5
    elif name == "imperiali":
        return lambda x: x + 2.0
    elif name == "huntington-hill":
        return lambda x: math.sqrt(x * (x + 1.0))
    elif name == "danish":
        return lambda x: x + 1 / 3
    elif name == "adams":
        return lambda x: x
    elif name == "Dean":
        return lambda x: x * (x + 1.0) / (x + 0.5)
    else:
        return ValueError("'name' is not implemented")


def prop_allocation(
    x: dict,
    n: int,
    *,
    initial: dict = {},
    available: dict = {},
    divisor: Callable[[int], float] = divisor_method("Jefferson/D'Hondt"),
    ties: str = "largest",
) -> dict:
    """Proportional to size allocation for stratified sampling.

    Parameters
    ----------
    x : dict
        The size of each stratum.
    n : int
        Sample size.
    initial : dict, optional
        Initial allocation for strata in 'x'.
    available : dict, optional
        Number of available units for each strata in `x`.
    divisor : Callable[[int], float], optional
        A function for the divisor (highest-averages)
        apportionment method. The default uses the Jefferson/D'Hondt method.
    ties : {'largest', 'first'}, optional
        Either 'largest' to break ties in favor of the stratum with the
        largest size (the default), or 'first' to break ties in favor of the
        ordering of 'x'.

    Returns
    -------
    dict
        Allocation for each stratum in 'x'.

    References
    ----------
    Balinksi, M. L. and Young, H. P. (1982).
    *Fair Representation: Meeting the Ideal of One Man, One Vote*.
    Yale University Press.

    Examples
    --------
    ```{python}
    import pysps

    pysps.prop_allocation({"a": 1, "b": 2, "c": 3}, 5)
    ```

    ```{python}
    pysps.prop_allocation(
        {"a": 1, "b": 2, "c": 3},
        5,
        divisor=pysps.divisor_method("Danish")
    )
    ```
    """
    if not set(x.keys()).issuperset(initial.keys()):
        raise ValueError("all keys in 'initial' must also be in 'x'")
    if not set(x.keys()).issuperset(available.keys()):
        raise ValueError("all keys in 'available' must also be in 'x'")
    if any(x < 0 for x in x.values()):
        raise ValueError("all elements of 'x' must be positive")
    if any(x < 0 for x in initial.values()):
        raise ValueError("all elements of 'initial' must be positive")

    res = dict.fromkeys(x.keys(), 0)
    res.update(initial)

    n = int(n)
    if n < 0:
        raise ValueError("'n' must be greater than or equal to 0")

    upper = dict.fromkeys(x.keys(), n)
    upper.update(available | {k: 0 for k, v in x.items() if v == 0})

    if n < sum(res.values()):
        raise ValueError("'n' is smaller than initial allocation")
    if n > sum(upper.values()):
        raise ValueError("'n' is larger than number of available units")
    for k in res:
        if res[k] > upper[k]:
            raise ValueError(
                "initial allocation must be smaller that number of available units"
            )

    ord = list(x.keys())
    if ties == "largest":
        x = {k: x[k] for k in sorted(x, key=x.get, reverse=True)}
    elif ties == "first":
        pass
    else:
        raise ValueError("'ties' must be either 'largest' or 'first'")

    s = {k: v / divisor(res[k]) for k, v in x.items() if res[k] < upper[k]}
    n -= sum(res.values())
    while n > 0:
        k = max(s, key=s.get)
        res[k] += 1
        n -= 1
        if res[k] < upper[k]:
            s[k] = x[k] / divisor(res[k])
        else:
            del s[k]

    return {k: res[k] for k in ord}
