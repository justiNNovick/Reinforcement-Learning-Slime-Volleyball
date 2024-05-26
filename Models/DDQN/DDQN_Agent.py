# This code was modified and adapted from 
# https://github.com/XinJingHao/Prioritized-Experience-Replay-DDQN-Pytorch/tree/main

import copy
import numpy as np
import torch
import torch.nn as nn
import os

def build_net(layer_shape, activation, output_activation):
	'''build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_layer_shape):
		super(Q_Net, self).__init__()
		layers = [state_dim] + list(hidden_layer_shape) + [action_dim]
		self.Q = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		q = self.Q(s)
		if torch.isnan(q).any():
			print("NaNs detected in output of Q")
		return q


class DDQN_Agent(object):
	def __init__(self, state_dim, action_dim, hidden_layer_shape, device, lr, gamma, batch_size, epsilon):
		
		self.q_net = Q_Net(state_dim, action_dim, (hidden_layer_shape, hidden_layer_shape)).to(device)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
		self.q_target = copy.deepcopy(self.q_net)
		
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): 
			p.requires_grad = False
		
		self.gamma = gamma
		self.tau = 0.005
		self.batch_size = batch_size
		self.epsilon = epsilon
		self.action_dim = action_dim
		self.device = device

	def select_action(self, state, deterministic): 	#only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
			if deterministic:
				a = self.q_net(state).argmax().item()
				return a
			else:
				Q = self.q_net(state)

				if np.random.rand() < self.epsilon:
					a = np.random.randint(0,self.action_dim)
					q_a = Q[0,a] # on device
				else:
					a = Q.argmax().item()
					q_a = Q[0,a] # on device
				return a, q_a


	def train(self,replay_buffer):
		s, a, r, s_prime, dw_mask, ind, Normed_IS_weight = replay_buffer.sample(self.batch_size)

		'''Compute the target Q value'''
		with torch.no_grad():
			argmax_a = self.q_net(s_prime).argmax(dim=1).unsqueeze(-1)
			max_q_prime = self.q_target(s_prime).gather(1,argmax_a)

			'''Avoid impacts caused by reaching max episode steps'''
			target_Q = r + (1 - dw_mask) * self.gamma * max_q_prime #dw: die or win; shape：(batch_size, 1)

		# Get current Q estimates
		current_q_a = self.q_net(s).gather(1,a)  # shape：(batch_size, 1)

		td_errors = (current_q_a - target_Q).squeeze(-1) # shape：(batch_size,)
		loss = (Normed_IS_weight * (td_errors ** 2)).mean()

		replay_buffer.update_batch_priorities(ind, td_errors.detach().cpu().numpy())

		self.q_net_optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
		self.q_net_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return loss.item()


	def save(self, folder_path, agent_number, n, priority_buffer):
		"""
        Save the models, their optimizers and the replay buffer to the folder_path
        """
		torch.save(self.q_net.state_dict(), f"{folder_path}/step-{n}-agent-{agent_number}-q_net.pt")
		torch.save(self.q_target.state_dict(), f"{folder_path}/step-{n}-agent-{agent_number}-q_target.pt")
		torch.save(self.q_net_optimizer.state_dict(), f"{folder_path}/step-{n}-agent-{agent_number}-q_net-optim.pt")
		# Create the folder if it does not exist
		os.makedirs(f"{folder_path}/step-{n}-priority_buffer", exist_ok=True)

		priority_buffer.save_to_disk(f"{folder_path}/step-{n}-priority_buffer")


	def load(self, folder_path, agent_number, n, priority_buffer):
		"""
        Load the models, their optimizers and the replay buffer from the folder_path
        """
		self.q_net.load_state_dict(torch.load(f"{folder_path}/step-{n}-agent-{agent_number}-q_net.pt"))
		self.q_target.load_state_dict(torch.load(f"{folder_path}/step-{n}-agent-{agent_number}-q_target.pt"))
		self.q_net_optimizer.load_state_dict(torch.load(f"{folder_path}/step-{n}-agent-{agent_number}-q_net-optim.pt"))
		priority_buffer.load_from_disk(f"{folder_path}/step-{n}-priority_buffer")

	def evaluation_mode(self):
		"""
		Set the models to evaluation mode
		"""
		self.q_net.eval()
		self.q_target.eval()

	def training_mode(self):
		"""		
		Set the models to training mode
		"""
		self.q_net.train()
		self.q_target.train()

	def disable_gradients(self):
		"""
		Disable the gradients for the target network
		"""
		for p in self.q_target.parameters():
			p.requires_grad = False

	def copy_models(self, model):
		"""
		Copy the models from another model
		"""
		self.q_net.load_state_dict(model.q_net.state_dict())
		self.q_target.load_state_dict(model.q_target.state_dict())