import torch
from .basic_block import *
from .encoder_gcn import Head
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()
        # Each expert has two linear layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first linear layer
        x = self.fc2(x)  # Output of the second linear layer
        return x

class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size):
        super(MixtureOfExperts, self).__init__()

        # hidden_size = 2000
        
        self.linear = nn.Linear(input_size, hidden_size)

        # Gating network to determine the weights for each expert
        self.gate = nn.Linear(hidden_size, num_experts)
        
        # Create multiple experts, each with two linear layers
        self.experts = nn.ModuleList([Expert(hidden_size, hidden_size, output_size) for _ in range(num_experts)])
        
    def forward(self, x):
        # Pass input through the linear layer
        x = F.relu(self.linear(x))

        # Get gating weights from the gate network (softmax to get probabilities)
        gate_weights = F.softmax(self.gate(x), dim=1)  # Shape: (batch_size, num_experts)
        
        # Collect the outputs from each expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # Shape: (batch_size, num_experts, output_size)
        
        # Multiply expert outputs by gate weights and sum them up
        weighted_sum = torch.einsum('be, beo -> bo', gate_weights, expert_outputs)
        
        return weighted_sum