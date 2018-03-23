import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from neural_net import Neural_Net

class DQN():
    def __init__(self):
        self.BS = 32
        self.frozen_net, self.training_net = Neural_Net(181,10), Neural_Net(181,10)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.training_net.parameters(), lr=0.0001)

        self.init_e = 1
        self.final_e = 0.1
        self.num_e_change = 10000
        self.delta_e = (self.init_e - self.final_e)/self.num_e_change
        self.e = self.init_e

        self.memory = []
        self.memory_size = 10000
        self.memory_counter = 0

    def choose_action(self, x):
        if np.random.uniform() < self.e:
            action = np.random.randint(0, 10)
        else:
            x = Variable(torch.FloatTensor(x))
            actions_value = self.training_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0,0]

        if self.e > self.final_e:
            self.e -= self.delta_e
        else:
            self.e = self.final_e

        return action

    def store_sars_(self, s, a, r, s_, done):
        sars_ = {s, a, r ,s_, done}
        self.memory[self.memory_counter] = sars_

        self.memory_counter += 1
        if self.memory_counter == self.memory_size:
            self.memory_counter = 0

    def train_net(self):
        sample_index = np.random.choice(self.memory_size, self.BS)

    def update_parameter(self):
        self.frozen_net.load_state_dict(self.training_net.state_dict())
