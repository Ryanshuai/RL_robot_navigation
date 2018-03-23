import torch
import numpy as np

memory = []

sars_ = [[0, 2], 'a', 0, [0, 4]]

for i in range(10):
    sars_[2] += 1
    memory.insert(1, sars_)

print(memory)

sample_index = np.random.choice(10, 6)
print(sample_index)
print(type(sample_index))

BS_memory = memory[sample_index]
print(BS_memory)

BS_s = BS_memory[1]
print(BS_s)