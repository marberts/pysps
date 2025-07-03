# Sequential Poisson sampling with Python

Sequential Poisson sampling is a variation of Poisson sampling for
drawing probability-proportional-to-size samples with a given number of
units, and is commonly used for price-index surveys. This package in a
Python implementation of the {sps} R package.

## Installation

## Usage

```python
>>> import pysps

>>> pi = pysps.InclusionProb([1, 2, 3, 4], 3)
>>> samp = pysps.OrderSample(pi)

>>> samp.units
array([1, 2, 3])

>>> samp.weights
array([1.5, 1.0, 1.0])
```