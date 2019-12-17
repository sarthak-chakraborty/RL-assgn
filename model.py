import torch
import torch.nn as nn
import torch.nn.functional as F

class ANetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, h1_size=32, h2_size=16):
        
        super(ANetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.output = nn.Linear(h1_size, action_size)
         

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=1)
        
