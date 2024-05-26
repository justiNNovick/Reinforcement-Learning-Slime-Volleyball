"""
Inspired from: https://github.com/ericyangyu/PPO-for-Beginners
Adapted for to perform self-play (From scratch) on a discrete action-space environment (Originally continuous)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.distributions import Categorical
from Models.PPO.MLP import MLP
from utils import convert_to_vector
    
class A2C_Agent:
    """
        This is the A2C class we will use as our model in main.py
    """
    def __init__(self, obs_dim, act_dim, DEVICE, \
                 timesteps_per_batch=5, lr=0.005, eps=1e-5, gamma=0.95, \
                      ent_coef=0, max_grad_norm=0.5, \
                        mlp_layers=[64, 64], render=False):
        """
            Initializes the A2C model, including hyperparameters.
        """		

        
        # Store the hyperparameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.DEVICE = DEVICE
        self.timesteps_per_batch = timesteps_per_batch # Number of timesteps to run per batch
        self.lr = lr
        self.gamma = gamma
        self.lam = 1 # GAE lambda is fixed to 1 for A2C
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.render = render

        # Initialize actor and critic networks
        self.actor = MLP(self.obs_dim, self.act_dim, is_actor=True, DEVICE=DEVICE, fc1_dims=mlp_layers[0], fc2_dims=mlp_layers[1])
        self.critic = MLP(self.obs_dim, 1, is_actor=False, DEVICE=DEVICE, fc1_dims=mlp_layers[0], fc2_dims=mlp_layers[1])

        # Initialize optimizers for actor and critic, A2C uses RMSprop for both unlike PPO that uses Adam
        self.actor_optim = RMSprop(self.actor.parameters(), lr=self.lr, eps=eps)
        self.critic_optim = RMSprop(self.critic.parameters(), lr=self.lr, eps=eps)

    def save_models(self, folder_path, agent_number, n):
        """
        Save the models
        """
        self.actor.save_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-actor.pt")
        self.critic.save_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-critic.pt")  
        torch.save(self.actor_optim.state_dict(), f"{folder_path}/step-{n}-agent-{agent_number}-actor_optim.pt")
        torch.save(self.critic_optim.state_dict(), f"{folder_path}/step-{n}-agent-{agent_number}-critic_optim.pt")

    def load_models(self, folder_path, agent_number, n):
        """
        Load the models
        """
        self.actor.load_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-actor.pt")
        self.critic.load_checkpoint(f"{folder_path}/step-{n}-agent-{agent_number}-critic.pt")
        self.actor_optim.load_state_dict(torch.load(f"{folder_path}/step-{n}-agent-{agent_number}-actor_optim.pt"))
        self.critic_optim.load_state_dict(torch.load(f"{folder_path}/step-{n}-agent-{agent_number}-critic_optim.pt"))
        
    def copy_models(self, agent):
        """
        Copy the models
        """
        self.actor.load_state_dict(agent.actor.state_dict())
        self.critic.load_state_dict(agent.critic.state_dict())

    def evaluation_mode(self):
        """
        Set the actor to evaluation mode
        """
        self.actor.eval()

    def training_mode(self):
        """
        Set the actor to training mode
        """
        self.actor.train()

    def disable_gradients(self):
        """
        Disable the gradients
        """
        for param in self.actor.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False

    def learn(self, batch_obs, batch_acts, batch_log_probs, batch_rews,\
               batch_vals, batch_dones, n_steps_so_far, total_n_steps, writer):
        """
            Train the actor and critic networks for 1 iteration given a batch of data.
        """   

        # Calculate advantage using GAE
        # Note A2C does not normalize advantages like PPO does
        A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones) 

        V = self.critic(batch_obs).squeeze()
        batch_rtgs = A_k + V.detach()   

        # Unlike PPO where we have a fixed number of epochs, in A2C we iterate over the data directly (we don't have a minibatch either)
        loss = []

        # Learning Rate Annealing
        frac = (n_steps_so_far - 1.0) / total_n_steps
        new_lr = self.lr * (1.0 - frac)
        # Make sure learning rate doesn't go below 0
        new_lr = max(new_lr, 0.0)
        self.actor_optim.param_groups[0]["lr"] = new_lr
        self.critic_optim.param_groups[0]["lr"] = new_lr

        # Log the learning rate as a function of the number of time steps
        writer.add_scalar('Learning rate value - Training step', new_lr, n_steps_so_far)

        # Calculate V_phi and pi_theta(a_t | s_t) and entropy
        V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)

        # Calculate the loss for the actor and critic networks
        actor_loss = -(A_k * curr_log_probs).mean()
        critic_loss = nn.MSELoss()(V, batch_rtgs)

        # Calulate the entropy loss
        entropy_loss = entropy.mean()

        # Discount entropy loss by given coefficient
        actor_loss = actor_loss - self.ent_coef * entropy_loss

        # Calculate gradients and perform backward propagation for actor network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        # Gradient Clipping with given threshold
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        # Calculate gradients and perform backward propagation for critic network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        loss.append(actor_loss.detach())

        # Log actor loss
        avg_loss = sum(loss) / len(loss)
        writer.add_scalar('Actor loss - Training step', avg_loss.item(), n_steps_so_far)

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []  # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []  # List to store advantages for the current episode
            last_advantage = 0  # Initialize the last computed advantage

            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]

                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  # Update the last advantage for the next timestep
                advantages.insert(0, advantage)  # Insert advantage at the beginning of the list

            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)

        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float).to(self.DEVICE)

    def select_action(self, obs, greedy=False):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        obs = torch.tensor(obs,dtype=torch.float).to(self.DEVICE)
        probs = self.actor(obs.to(self.DEVICE))
        
        # Create a categorical distribution over the list of probabilities of actions
        dist = Categorical(probs)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # If we're testing, just return the deterministic action. Sampling should only be for training
        # as our "exploration" factor.
        if greedy:
            return torch.argmax(probs).item(), 1

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().item(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
                batch_rtgs - the rewards-to-go calculated in the most recently collected
                                batch as a tensor. Shape: (number of timesteps in batch)
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        # if batch_obs.size(0) == 1:
        #     V = self.critic(batch_obs)
        # else:
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in select_action()
        probs = self.actor(batch_obs)
        dist = Categorical(probs)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, dist.entropy()

    def gather_data(self, env, otherAgent):
       
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        ep_vals = []
        ep_dones = []
        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            
            ep_rews = [] # rewards collected per episode
            ep_vals = [] # state values collected per episode
            ep_dones = [] # done flag collected per episode
            
            # Reset the environment. Note that obs is short for observation. 
            obs1 = env.reset()
            obs2 = obs1
            
            # Initially, the game is not done
            done = False
            ep_t = 0

            # Run an episode
            while not done:

                # If render is specified, render the environment
                if self.render:
                    self.env.render()

                # Track done flag of the current state
                ep_dones.append(done)

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs1)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                obs1 = torch.tensor(obs1, dtype=torch.float)
                action1, log_prob1 = self.select_action(obs1)
                action2, _ = otherAgent.select_action(obs2, greedy=True) # The opponent agent is always greedy
                val = self.critic(obs1.to(self.DEVICE))

                obs1, rew, done, info = env.step(convert_to_vector(action1), otherAction=convert_to_vector(action2))
                obs2 = info['otherObs']

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action1)
                batch_log_probs.append(log_prob1)

                # Increment the episode length
                ep_t += 1

            # Track episodic lengths, rewards, state values, and done flags
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.DEVICE)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.DEVICE)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten().to(self.DEVICE)

        # Here, we return the batch_rews instead of batch_rtgs for later calculation of GAE
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones
