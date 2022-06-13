import numpy as np
import math
x = 100
a1 = math.pow(1, 2)
a2 = math.pow(0, 2)
l1 = math.pow(1, 2)
l2 = math.pow(2, 2)
picked_ci = np.sqrt(x + a1/l1 + a2/l2)
not_picked_ci = np.sqrt(x + a1/l2 + a2/l1)
print(picked_ci - not_picked_ci)