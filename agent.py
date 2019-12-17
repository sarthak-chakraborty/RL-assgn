import numpy as np
import random
from collections import namedtuple, deque
from torch.distributions import Categorical


from model import ANetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

LR = 5e-4               # learning rate 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size, env, model_param, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.seed = random.seed(seed)

        self.anetwork = ANetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.anetwork.parameters(), lr=LR)
        
        try:
            self.anetwork.load_state_dict(torch.load(model_param))
        except:
            pass
      
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.anetwork.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)



    def learn(self, state, brain_name):
        rewards = []
        log_probs = []
        while(True):
            action, log_prob = self.act(state)
            log_probs.append(log_prob)
            
            env_info = self.env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            state = next_state

            rewards.append(reward)
            if done:
                break

        return log_probs, rewards
    
    