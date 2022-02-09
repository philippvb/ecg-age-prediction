from itertools import count
import numpy as np

x = np.array([1,2,3])
_, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
weights = 1/counts[inverse]
weights /= sum(weights)
weights *= len(x)
print(weights)