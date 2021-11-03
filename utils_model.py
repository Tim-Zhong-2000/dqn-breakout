import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__dim = action_dim
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        self.__fc1_1 = nn.Linear(64*7*7, 512)
        self.__fc1_2 = nn.Linear(64*7*7, 512)

        self.__fc2_1 = nn.Linear(512, action_dim)
        self.__fc2_2 = nn.Linear(512, 1)

        self.__relu = nn.ReLU()
        self.__device = device

    def forward(self, x):
        x = x / 255.
        x = self.__relu(self.__conv1(x))
        x = self.__relu(self.__conv2(x))
        x = self.__relu(self.__conv3(x))
        x = x.view(x.size(0), -1)
        adv = self.__relu(self.__fc1_1(x))
        val = self.__relu(self.__fc1_2(x))
        adv = self.__fc2_1(adv)
        val = self.__fc2_2(val).expand(x.size(0), self.__dim)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.__dim)
        return x

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
