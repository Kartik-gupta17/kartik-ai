from ai.math.vector import Vector

v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

print("Vector 1 :", v1)
print("Vector 2 :", v2)

print()

print("Addition :", v1.add(v2))
print("Subtraction :", v1.subtract(v2))
print("Dot Product :", v1.dot(v2))
print("Magnitude :", v1.magnitude())
print("Normalized :", v1.normalize())
print("Cosine Similarity :", v1.cosine_similarity(v2))