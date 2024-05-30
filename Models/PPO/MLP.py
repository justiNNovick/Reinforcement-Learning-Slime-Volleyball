import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self, in_dim, out_dim, is_actor, DEVICE, fc1_dims=64, fc2_dims=64):

        # Call the parent class constructor
        super(MLP, self).__init__()

        # Define the network
        self.network = nn.Sequential(
            nn.Linear(in_dim, fc1_dims), # *state_dim is the same as state_dim[0], state_dim[1], etc.
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, out_dim) # Output layer has action_dim neurons
        )

        # Store the type of network
        self.is_actor = is_actor

        # Move the network to the specified device
        self.to(DEVICE)

    def forward(self, state):
        """
        Forward pass through the network
        """
        if self.is_actor:
            return F.softmax(self.network(state), dim=-1)
        return self.network(state)

    def save_checkpoint(self, path):
        """
        Save the network
        """
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        """
        Load the network
        """
        self.load_state_dict(torch.load(path))