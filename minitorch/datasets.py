"""Simple 2D toy datasets for binary classification."""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random points uniformly distributed in the unit square.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        List of N (x_1, x_2) pairs with each coordinate in [0, 1).

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """A labeled 2D dataset.

    Attributes
    ----------
        N: Number of points.
        X: List of (x_1, x_2) points.
        y: Binary label (0 or 1) for each point.

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Points labeled 1 if they lie in the left half of the square (x_1 < 0.5).

    Args:
    ----
        N: Number of points.

    Returns:
    -------
        Graph with N points and labels.

    """
    X = make_pts(N)
    y: List[int] = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Points labeled 1 if they lie below the diagonal line x_1 + x_2 = 0.5.

    Args:
    ----
        N: Number of points.

    Returns:
    -------
        Graph with N points and labels.

    """
    X = make_pts(N)
    y: List[int] = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Points labeled 1 if they lie in the left or right edge bands (x_1 < 0.2 or x_1 > 0.8).

    Args:
    ----
        N: Number of points.

    Returns:
    -------
        Graph with N points and labels.

    """
    X = make_pts(N)
    y: List[int] = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Points labeled 1 if they lie in the top-left or bottom-right quadrant (XOR pattern).

    Args:
    ----
        N: Number of points.

    Returns:
    -------
        Graph with N points and labels.

    """
    X = make_pts(N)
    y: List[int] = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Points labeled 1 if they lie outside a circle of radius sqrt(0.1) centered at (0.5, 0.5).

    Args:
    ----
        N: Number of points.

    Returns:
    -------
        Graph with N points and labels.

    """
    X = make_pts(N)
    y: List[int] = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Two interleaved spirals centered at (0.5, 0.5), one per class.

    Args:
    ----
        N: Number of points (half are assigned to each spiral).

    Returns:
    -------
        Graph with the spiral points and labels.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
