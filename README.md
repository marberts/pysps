# A Python module for sequential Poisson sampling

A quick Python implementation of some functions from the `sps` R package.
Requires Python >= 3.10 and numpy.

```python
>>> import pysps

>>> pi = pysps.InclusionProb([1, 2, 3, 4], 3)
>>> samp = pysps.OrderSample(pi)

>>> samp.units
array([1, 2, 3])

>>> samp.weights
array([1.5, 1.0, 1.0])
```