from typing import Callable


def prop_allocation(
    x: dict,
    n: int,
    *,
    initial: dict = {},
    available: dict = {},
    divisor: Callable[[int], float] = lambda a: a + 1.0,
    ties: str = "largest",
) -> dict:
    if not set(x.keys()).issuperset(initial.keys()):
        raise ValueError("all keys in 'initial' must also be in 'x'")
    if not set(x.keys()).issuperset(available.keys()):
        raise ValueError("all keys in 'available' must also be in 'x'")
    if any(x < 0 for x in initial.values()):
        raise ValueError("all elements of 'initial' must be positive")

    res = dict.fromkeys(x.keys(), 0)
    res.update(initial)

    n = int(n)
    if n < 0:
        raise ValueError("'n' must be greater than or equal to 0")

    upper = dict.fromkeys(x.keys(), n)
    upper.update(available)

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
