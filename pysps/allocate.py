from typing import Callable


def prop_allocation(x: dict,
                    n: int,
                    *,
                    initial: dict = {},
                    available: dict = {},
                    divisor: Callable[[int], float] = lambda a: a + 1.0,
                    ties: str = "largest") -> dict:
    if set(x.keys()).isdisjoint(initial.keys()):
        raise ValueError("all keys in 'initial' must also be in 'x'")
    if set(x.keys()).isdisjoint(available.keys()):
        raise ValueError("all keys in 'available' must also be in 'x'")

    res = dict.fromkeys(x.keys(), 0)
    res.update(initial)
    n -= sum(res.values()
    )
    upper = dict.fromkeys(x.keys(), n)
    upper.update(available)

    if ties == "largest":
        pass
    
    s = {k: v / divisor(res[k]) for k, v in x.items() if res[k] < upper[k]}
    while n > 0:
        k = max(s, key=s.get)
        res[k] += 1
        n -= 1
        if res[k] < upper[k]:
            s[k] = x[k] / divisor(res[k])
        else:
            del s[k]
    return res