import torch
import numpy as np
import copy as cp

memory_size = 9
memory = [0]*memory_size
print(memory)

sars_ = [[0, 2], 'a', 0, [0, 4]]

for i in range(memory_size):
    temp = cp.deepcopy(sars_)
    temp[2] = i
    memory[i] = temp

print(memory)

sample_index = np.random.choice(memory_size, 6)
print(sample_index)
print(type(sample_index))

BS_memory = np.array(memory)[sample_index]
print(BS_memory)

BS_s = BS_memory[:, 0]
BS_a = BS_memory[:, 1]
BS_r = BS_memory[:, 2]
BS_s_ = BS_memory[:, 3]

print('----------------')
print(BS_s)
print(BS_s.size)
print(type(BS_s))

# print(BS_a)
# print(BS_r)
# print(BS_s_)