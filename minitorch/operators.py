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


def mul(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        a: First float.
        b: Second float.

    Returns:
    -------
        Product of a and b.

    """
    return a * b


def id(a: float) -> float:
    """Identity function.

    Args:
    ----
        a: Input float.

    Returns:
    -------
        Input float.

    """
    return a


def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
    ----
        a: First float.
        b: Second float.

    Returns:
    -------
        Sum of a and b.

    """
    return a + b


def neg(a: float) -> float:
    """Negate a number.

    Args:
    ----
        a: Input float.

    Returns:
    -------
        Negated float.

    """
    return float(-a)


def lt(a: float, b: float) -> float:
    """Less than comparison.

    Args:
    ----
        a: First float.
        b: Second float.

    Returns:
    -------
        1.0 if a < b else 0.0.

    """
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Equality comparison.

    Args:
    ----
        a: First float.
        b: Second float.

    Returns:
    -------
        1.0 if a == b else 0.0.

    """
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Maximum of two numbers.

    Args:
    ----
        a: First float.
        b: Second float.

    Returns:
    -------
        Maximum of a and b.

    """
    return a if a > b else b


def is_close(a: float, b: float) -> float:
    """Check if two numbers are close within a tolerance of 1e-2.

    Args:
    ----
        a: First float.
        b: Second float.

    Returns:
    -------
        1.0 if |a - b| < 1e-2 else 0.0.

    """
    return 1.0 if abs(a - b) < 1e-2 else 0.0


def sigmoid(a: float) -> float:
    """Sigmoid activation function.

    Args:
    ----
        a: Input float.

    Returns:
    -------
        Sigmoid of a.

    """
    return 1 / (1 + math.exp(-a))


def relu(a: float) -> float:
    return max(a, 0.0)


def log(a: float) -> float:
    return math.log(a)


def exp(a: float) -> float:
    return math.exp(a)


def log_back(a: float, x: float) -> float:
    return x / a


def inv(a: float) -> float:
    return 1.0 / a


def inv_back(a: float, x: float) -> float:
    return -x / (a**2)


def relu_back(a: float, x: float) -> float:
    return 0.0 if a <= 0.0 else 1.0 * x


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
    def apply(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for i in ls:
            ret.append(fn(i))
        return ret

    return apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def apply(ls: Iterable[float], ls1: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls, ls1):
            ret.append(fn(x, y))
        return ret

    return apply


def reduce(
    fn: Callable[[float, float], float], init: float
) -> Callable[[Iterable[float]], float]:
    def apply(ls: Iterable[float]) -> float:
        curr = init
        for i in ls:
            curr = fn(curr, i)
        return curr

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)


def addLists(ls: Iterable[float], ls1: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls, ls1)


def sum(ls: Iterable[float]):
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]):
    return reduce(mul, 1)(ls)
