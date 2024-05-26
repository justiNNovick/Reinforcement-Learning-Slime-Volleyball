# This code was modified and adapted from 
# https://github.com/XinJingHao/Prioritized-Experience-Replay-DDQN-Pytorch/tree/main

import torch
import numpy as np

class LightPriorReplayBuffer():
    """
    Implements a memory-efficient priority experience replay buffer that avoids explicitly storing the next state (s_next) 
    for each state-action pair. This design is particularly beneficial for environments with large state spaces, such as those involving images.
    
    The buffer includes mechanisms to handle terminal and truncated states correctly during training, ensuring the integrity of state transitions 
    and adjusting loss calculations appropriately.
    
    Attributes:
        device: The device on which the tensors will be allocated (e.g., CPU or CUDA).
        ptr: The current index for adding new experiences.
        size: The current size of the buffer.
        state: Tensor storing the states.
        action: Tensor storing the actions.
        reward: Tensor storing the rewards.
        done: Tensor indicating whether a state is terminal.
        priorities: Tensor storing the priorities of experiences based on their TD-error.
        buffer_size: The maximum size of the buffer.
        alpha: The exponent determining the impact of TD-error on priority.
        beta: The exponent for importance-sampling weights, adjusts how much prioritization is used.
        replacement: Boolean indicating whether sampling with replacement is allowed.

    Usage:
        - Add experiences using the add method during environment interaction.
        - Sample batches of experiences using the sample method for training.
    """
    def __init__(self, buffer_size, state_dim, alpha, beta_init, replacement, device):
        self.device = device
        self.ptr = 0
        self.size = 0

        # Preallocate memory for the buffer's components. For image states, consider using uint8 for the state tensor to save space.
        self.state = torch.zeros((buffer_size, state_dim), device=device)
        self.action = torch.zeros((buffer_size, 1), dtype=torch.int64, device=device)
        self.reward = torch.zeros((buffer_size, 1), device=device)
        self.tr = torch.zeros((buffer_size, 1), dtype=torch.bool, device=device) #only 0/1
        self.done = torch.zeros((buffer_size, 1), dtype=torch.bool, device=device)  # 0/1 indicating done states
        self.priorities = torch.zeros(buffer_size, dtype=torch.float32, device=device)  # Priorities based on TD-error
        self.buffer_size = buffer_size

        self.alpha = alpha
        self.beta = beta_init
        self.replacement = replacement

    def add(self, state, action, reward, done, tr, priority):
        """
        Adds a new experience to the buffer.
        
        Arguments:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            done: Boolean indicating if the next state is terminal.
            priority: The priority of this experience.
        """
        # Convert inputs to appropriate tensor formats and store them in the buffer.
        self.state[self.ptr] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.tr[self.ptr] = tr
        self.priorities[self.ptr] = priority

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)


    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.
        
        Arguments:
            batch_size: The number of experiences to sample.
        
        Returns:
            A tuple containing batches of states, actions, rewards, next states, done flags, indices of sampled experiences, 
            and normalized importance-sampling weights.
        """
        # Normalize priorities to create a probability distribution
        probabilities = self.priorities[:self.size] ** self.alpha  # Apply alpha scaling
        total_priority = probabilities.sum()
        if total_priority == 0:
            raise ValueError("Sum of priorities is zero, cannot sample.")
        probabilities /= total_priority

        # Setting the probability of the current pointer to zero if it is in the range
        if self.ptr < self.size:
            probabilities[self.ptr] = 0
        probabilities /= probabilities.sum()  # Re-normalize after modifying

        # Sample indices based on calculated probabilities
        ind = torch.multinomial(probabilities, num_samples=batch_size, replacement=self.replacement)

        # Compute next states indices handling edge cases
        next_indices = (ind + 1) % self.size

        # Calculate importance-sampling weights for the sampled experiences, normalizing them
        IS_weights = (1 / (probabilities[ind] * self.size)) ** self.beta
        IS_weights /= IS_weights.max()
        IS_weights = IS_weights.unsqueeze(-1)  # Adjust shape for batching

        # CHeck for NaNs:
        # if torch.isnan(Normed_IS_weight).any():
        #     print('Nans in Normed_IS_weight')
        if torch.isnan(probabilities).any():
            print('Nans in probabilities')
        if torch.isnan(IS_weights).any():
            print('Nans in IS_weight')
        if torch.isnan(ind).any():
            print('Nans in ind')      

        return self.state[ind], self.action[ind], self.reward[ind], self.state[next_indices], self.done[ind], self.tr[ind], ind, IS_weights

    def save_to_disk(self, path):
        """
        Saves the buffer to disk.
        
        Arguments:
            path: The path to save the buffer.
        """
        torch.save(self.state, f"{path}/state.pt")
        torch.save(self.action, f"{path}/action.pt")
        torch.save(self.reward, f"{path}/reward.pt")
        torch.save(self.done, f"{path}/done.pt")
        torch.save(self.priorities, f"{path}/priorities.pt")
        torch.save(self.ptr, f"{path}/ptr.pt")
        torch.save(self.size, f"{path}/size.pt")
        torch.save(self.beta, f"{path}/beta.pt")

    def load_from_disk(self, path):
        """
        Loads the buffer from disk.
        
        Arguments:
            path: The path to load the buffer.
        """
        self.state = torch.load(f"{path}/state.pt")
        self.action = torch.load(f"{path}/action.pt")
        self.reward = torch.load(f"{path}/reward.pt")
        self.done = torch.load(f"{path}/done.pt")
        self.priorities = torch.load(f"{path}/priorities.pt")
        self.ptr = torch.load(f"{path}/ptr.pt")
        self.size = torch.load(f"{path}/size.pt")
        self.beta = torch.load(f"{path}/beta.pt")