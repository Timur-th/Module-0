"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Return 1.0 if x is less than y, else 0.0."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return 1.0 if x is equal to y, else 0.0."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Return whether two numbers are within 1e-2 of each other."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    r"""Compute the sigmoid function in a numerically stable way.

    $f(x) = \frac{1.0}{1.0 + e^{-x}}$ if x >= 0 else $\frac{e^x}{1.0 + e^{x}}$
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return x if x is positive, else 0.0."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Compute the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the derivative of log at x, times d."""
    return d / x


def inv(x: float) -> float:
    """Compute the reciprocal 1/x."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of 1/x at x, times d."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of relu at x, times d."""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    Args:
    ----
        fn: Function from one value to one value.

    Returns:
    -------
        A function that takes a list, applies `fn` to each element, and returns a new list.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith (or map2).

    Args:
    ----
        fn: Function combining two values.

    Returns:
    -------
        A function that takes two equally sized lists and produces a new list
        by applying `fn` to each pair of elements.

    """

    def _zip_with(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return _zip_with


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.

    Args:
    ----
        fn: Function combining two values.
        start: Starting value $x_0$.

    Returns:
    -------
        A function that takes a list `ls` of elements $x_1 \ldots x_n$ and computes
        $fn(x_n, fn(x_{n-1}, \ldots fn(x_1, x_0)))$.

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for x in ls:
            val = fn(x, val)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate each element of a list using `map`."""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of two lists pairwise using `zipWith`."""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum a list using `reduce`."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Take the product of a list using `reduce`."""
    return reduce(mul, 1.0)(ls)
