import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory, ReplayMemoryLSTM
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)

    """Get action using policy net using epsilon-greedy policy"""
  
    def get_action(self, state):
      if np.random.rand() <= self.epsilon:
          # Choose a random action
          a = random.randrange(self.action_size)
      else:
          # Choose the best action
          state = torch.FloatTensor(state).unsqueeze(0).to(device) #unsqueeze to convert the state of (3, 84, 84) to (1, 3, 84, 84) to pass in the nn
          with torch.no_grad():
              q_values = self.policy_net(state)
              a = q_values.argmax().item()

      # Decay epsilon
      if self.epsilon > self.epsilon_min:
          self.epsilon -= self.epsilon_decay

      return a

    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        next_states = torch.from_numpy(next_states).cuda()
        dones = mini_batch[3]  # checks if the game is over
        mask = torch.tensor(list(map(int, dones == False)), dtype=torch.uint8).cuda()

        # Compute Q(s_t, a), the Q-value of the current state
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1))

        # Compute Q function of next state
        next_q_values = self.policy_net(next_states)

        # Find maximum Q-value of action at next state from policy net
        max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)

        # Compute the expected Q values
        expected_q_values = rewards.unsqueeze(1) + self.discount_factor * max_next_q_values * mask.unsqueeze(1)

        # Compute the Huber Loss
        loss = F.smooth_l1_loss(q_values, expected_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Step the scheduler
        self.scheduler.step()






