import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from neural_net import Neural_Net

class DQN():
    def __init__(self, BS, n_states, n_actions):
        self.state_size = n_states
        self.action_size = n_actions

        self.BS = BS
        self.frozen_net = Neural_Net(self.state_size, self.action_size)
        self.training_net = Neural_Net(self.state_size, self.action_size)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.training_net.parameters(), lr=0.001)
        self.gamma = 0.99

        self.init_e = 1
        self.final_e = 0.1
        self.num_e_change = 5000
        self.delta_e = (self.init_e - self.final_e)/self.num_e_change
        self.e = self.init_e

        self.memory_size = 1000
        self.memory = np.zeros((self.memory_size, self.state_size * 2 + 3))
        self.memory_counter = 0

    def choose_action(self, x):
        if np.random.uniform() < self.e:
            action = np.random.randint(0, self.action_size)
        else:
            x = Variable(torch.FloatTensor(x))
            actions_value = self.training_net.forward(x)
            actions_value = actions_value.data.numpy() #Variable-->numpy
            action = np.argmax(actions_value)
            # print(actions_value)
            # print(action)

        if self.e > self.final_e:
            self.e -= self.delta_e
        else:
            self.e = self.final_e
        return action

    def store_sars_(self, s, a, r, s_, done):
        stas_d = np.hstack((s, a, r, s_, done))
        self.memory[self.memory_counter, :] = stas_d

        self.memory_counter += 1
        if self.memory_counter == self.memory_size:
            self.memory_counter = 0

    def train_net(self):
        sample_index = np.random.choice(self.memory_size, self.BS)
        BS_memory = self.memory[sample_index, :]
        BS_s = Variable(torch.FloatTensor(BS_memory[:, :self.state_size]))
        BS_a = Variable(torch.LongTensor(BS_memory[:, self.state_size:self.state_size + 1].astype(int)))
        BS_r = Variable(torch.FloatTensor(BS_memory[:, self.state_size + 1:self.state_size + 2]))
        BS_s_ = Variable(torch.FloatTensor(BS_memory[:, -self.state_size:-1]))
        BS_d = Variable(torch.FloatTensor(BS_memory[:, -1])) # shape (BS, 1)

        q = self.training_net(BS_s).gather(1, BS_a)  # shape (batch, 1) ??
        q_ = self.frozen_net(BS_s_).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = BS_r + self.gamma * q_.max(1)[0] * (1 - BS_d)  # shape (batch, 1) ??
        loss = self.loss_func(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_parameter(self):
        self.frozen_net.load_state_dict(self.training_net.state_dict())
