import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Neural_Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Neural_Net, self).__init__()
        self.fc1 =  nn.Linear(input_dim, 1024)
        init.xavier_normal(self.fc1)
        self.fc2 = nn.Linear(1024, 512)
        init.xavier_normal(self.fc2)
        self.fc3 = nn.Linear(512, 10)
        init.xavier_normal(self.fc3)

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.relu(x)
        return output