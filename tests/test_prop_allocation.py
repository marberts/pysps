from pysps import prop_allocation


def no_alabama_paradox():
    assert prop_allocation({1: 6, 2: 6, 3: 2}, 10) == {1: 5, 2: 4, 3: 1}
    assert prop_allocation({1: 6, 2: 6, 3: 2}, 11) == {1: 5, 2: 5, 3: 1}

def corner_cases():
    assert prop_allocation({1: 0}, 0) == {1: 0}

def abc():
    assert 2 == 1