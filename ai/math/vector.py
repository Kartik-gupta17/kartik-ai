"""
KartikAI Mathematics Engine
Vector Operations
"""

from math import sqrt
from typing import List


class Vector:
    def __init__(self, values: List[float]):
        self.values = values

    def __str__(self):
        return f"Vector({self.values})"

    def add(self, other):
        return Vector([a + b for a, b in zip(self.values, other.values)])

    def subtract(self, other):
        return Vector([a - b for a, b in zip(self.values, other.values)])

    def multiply(self, scalar: float):
        return Vector([scalar * x for x in self.values])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.values, other.values))

    def magnitude(self):
        return sqrt(sum(x * x for x in self.values))

    def normalize(self):
        mag = self.magnitude()

        if mag == 0:
            return Vector(self.values)

        return Vector([x / mag for x in self.values])

    def cosine_similarity(self, other):
        return self.dot(other) / (
            self.magnitude() * other.magnitude()
        )