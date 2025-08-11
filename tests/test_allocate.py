import numpy as np
from pysps import prop_allocation
import pytest


def test_no_alabama_paradox():
    assert prop_allocation({1: 6, 2: 6, 3: 2}, 10) == {1: 5, 2: 4, 3: 1}
    assert prop_allocation({1: 6, 2: 6, 3: 2}, 11) == {1: 5, 2: 5, 3: 1}


def test_corner_cases():
    assert prop_allocation({}, 0) == {}
    assert prop_allocation({1: 0}, 0) == {1: 0}
    assert prop_allocation({1: 1}, 0) == {1: 0}
    assert prop_allocation({1: 1}, 5) == {1: 5}
    assert prop_allocation({1: 0, 2: 1}, 5) == {1: 0, 2: 5}


def test_simple_allocations():
    assert prop_allocation({1: 1, 2: 100}, 5) == {1: 0, 2: 5}
    assert prop_allocation({1: 1, 2: 100}, 5, initial={1: 1}) == {1: 1, 2: 4}
    assert prop_allocation({1: 1, 2: 100}, 5, initial={1: 1}, available={2: 3}) == {
        1: 2,
        2: 3,
    }
    assert prop_allocation({1: 1, 2: 100}, 5, available={2: 0}) == {1: 5, 2: 0}


def test_voting_examples():
    # example from https://en.wikipedia.org/wiki/Highest_averages_method
    x = {1: 47000, 2: 16000, 3: 15900, 4: 12000, 5: 6000, 6: 0, 7: 3100}
    assert prop_allocation(x, 10) == {1: 5, 2: 2, 3: 2, 4: 1, 5: 0, 6: 0, 7: 0}
    assert prop_allocation(x, 10, divisor=lambda a: a + 0.5) == {
        1: 4,
        2: 2,
        3: 2,
        4: 1,
        5: 1,
        6: 0,
        7: 0,
    }
    assert prop_allocation(
        x,
        10,
        divisor=lambda a: np.sqrt(a * (a + 1)),
        initial={1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0, 7: 1},
        available={6: 0},
    ) == {1: 4, 2: 2, 3: 1, 4: 1, 5: 1, 6: 0, 7: 1}
    assert prop_allocation(
        x,
        10,
        divisor=lambda a: a,
        initial={1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0, 7: 1},
        available={6: 0},
    ) == {1: 3, 2: 2, 3: 2, 4: 1, 5: 1, 6: 0, 7: 1}


def test_ties():
    assert prop_allocation({1: 4, 2: 5}, 8) == {1: 3, 2: 5}
    assert prop_allocation({1: 4, 2: 5}, 8, ties="first") == {1: 4, 2: 4}
    assert prop_allocation({1: 1, 2: 1, 3: 1}, 2) == {1: 1, 2: 1, 3: 0}
    assert prop_allocation({1: 1, 2: 1, 3: 1}, 2, ties="first") == {1: 1, 2: 1, 3: 0}


def test_exceptions():
    with pytest.raises(ValueError) as execinfo:
        prop_allocation({1: 0}, 1)
    assert execinfo.type is ValueError

    with pytest.raises(ValueError) as execinfo:
        prop_allocation({1: 1}, 3, available={1: 2})
    assert execinfo.type is ValueError

    with pytest.raises(ValueError) as execinfo:
        prop_allocation({1: 1}, 1, initial={1: 2})
    assert execinfo.type is ValueError

    with pytest.raises(ValueError) as execinfo:
        prop_allocation({1: 1}, 3, initial={1: 2}, available={1: 1})
    assert execinfo.type is ValueError

    with pytest.raises(ValueError) as execinfo:
        prop_allocation({1: 1}, 3, initial={2: 2})
    assert execinfo.type is ValueError

    with pytest.raises(ValueError) as execinfo:
        prop_allocation({1: 1}, 3, available={2: 2})
    assert execinfo.type is ValueError
