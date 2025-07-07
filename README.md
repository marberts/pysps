# Sequential Poisson sampling with Python <a href="https://marberts.github.io/pysps/"><img src="docs/logo.png" align="right" height="139" alt="pysps website" /></a>

[![CI Build](https://github.com/marberts/pysps/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/marberts/pysps/actions/workflows/ci-tests.yml)

Sequential Poisson sampling is a variation of Poisson sampling for
drawing probability-proportional-to-size samples with a given number of
units, and is commonly used for price-index surveys. This package is a
Python implementation of the [{sps}](https://cran.r-project.org/package=sps) R package.

## Installation

```
python -m pip install git+https://github.com/marberts/pysps.git
```

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