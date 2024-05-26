# This code was modified and adapted from 
# https://github.com/XinJingHao/Prioritized-Experience-Replay-DDQN-Pytorch/tree/main

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from utils import SumTree

class PrioritizedReplayBuffer(object):
	
	def __init__(self, buffer_size, state_dim, alpha, beta_init, device):
		self.ptr = 0
		self.size = 0
		max_size = int(buffer_size)
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, 1))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.done = np.zeros((max_size, 1))
		self.max_size = max_size

		self.sum_tree = SumTree(max_size)
		self.alpha = alpha
		self.beta = beta_init
		self.device = device

	def save_to_disk(self, path):
		torch.save(self.state, f"{path}/state.pt")
		torch.save(self.action, f"{path}/action.pt")
		torch.save(self.reward, f"{path}/reward.pt")
		torch.save(self.done, f"{path}/done.pt")
		torch.save(self.ptr, f"{path}/ptr.pt")
		torch.save(self.size, f"{path}/size.pt")
		torch.save(self.beta, f"{path}/beta.pt")

	def load_from_disk(self, path):
		self.state = torch.load(f"{path}/state.pt")
		self.action = torch.load(f"{path}/action.pt")
		self.reward = torch.load(f"{path}/reward.pt")
		self.done = torch.load(f"{path}/done.pt")
		self.ptr = torch.load(f"{path}/ptr.pt")
		self.size = torch.load(f"{path}/size.pt")
		self.beta = torch.load(f"{path}/beta.pt")

	def add(self, state, action, reward, next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.done[self.ptr] = done.cpu()  #0,0,0，...，1

		# 
		priority = 1.0 if self.size == 0 else self.sum_tree.priority_max
		self.sum_tree.update_priority(data_index=self.ptr, priority=priority)  

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind, Normed_IS_weight = self.sum_tree.prioritized_sample(N=self.size, batch_size=batch_size, beta=self.beta)

		return (
			torch.tensor(self.state[ind], dtype=torch.float32).to(self.device),
			torch.tensor(self.action[ind], dtype=torch.long).to(self.device),
			torch.tensor(self.reward[ind], dtype=torch.float32).to(self.device),
			torch.tensor(self.next_state[ind], dtype=torch.float32).to(self.device),
			torch.tensor(self.done[ind], dtype=torch.float32).to(self.device),
			ind,
			Normed_IS_weight.to(self.device) # shape：(batch_size,)
		)
	
	def update_batch_priorities(self, batch_index, td_errors):  #td_errors: (batch_size,) 
		priorities = (np.abs(td_errors) + 0.01) ** self.alpha  # (batch_size,)
		for index, priority in zip(batch_index, priorities): #index: (batch_size,)  priority: (batch_size,)
			self.sum_tree.update_priority(data_index=index, priority=priority) #index: (batch_size,)  priority: (batch_size,)