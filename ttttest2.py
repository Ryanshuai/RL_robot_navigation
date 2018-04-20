import numpy as np
s = np.array([1,2,3])
a = 2
r = 5
s_ = np.array([2,3,4])
print(s_)
print(type(s_))

mem = np.vstack((s, a, r, s_))
print(mem)