# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

def MLP(channels, act_fn=ReLU, islast=False):
    """
    Create a Multi-Layer Perceptron.
    Args:
        channels: List of channel sizes [input_size, hidden1, hidden2, ..., output_size]
        act_fn: Activation function to use (default: ReLU)
        islast: If True, last layer will not have activation function
    Returns:
        Sequential MLP model
    """
    if not islast:
        layers = [Seq(Lin(channels[i - 1], channels[i]), act_fn())
                 for i in range(1, len(channels))]
    else:
        layers = [Seq(Lin(channels[i - 1], channels[i]), act_fn())
                 for i in range(1, len(channels)-1)]
        layers.append(Seq(Lin(channels[-2], channels[-1])))
    
    return Seq(*layers)

class SDFNetwork(nn.Module):
    """
    Neural network for learning Signed Distance Functions (SDF).
    Uses skip connections from input to each hidden layer for better geometry learning.
    """
    def __init__(self, input_size=3, output_size=1, hidden_size=128, num_layers=5, act_fn=ReLU):
        """
        Args:
            input_size: Input dimension (default: 3 for xyz coordinates)
            output_size: Output dimension (default: 1 for SDF value)
            hidden_size: Size of hidden layers
            num_layers: Number of hidden layers
            act_fn: Activation function to use
        """
        super(SDFNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Create layer structure
        channels = [hidden_size] * num_layers
        channels[0] = input_size
        channels.append(output_size)
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(MLP([input_size, hidden_size], act_fn=act_fn))
        
        # Hidden layers with skip connections
        for i in range(num_layers-1):
            self.layers.append(
                MLP([hidden_size + input_size, hidden_size], act_fn=act_fn)
            )
        
        # Output layer
        self.layers.append(
            MLP([hidden_size + input_size, output_size], act_fn=act_fn, islast=True)
        )
        
    def forward(self, x):
        """
        Forward pass with skip connections.
        Args:
            x: Input tensor of shape [batch_size, input_size]
        Returns:
            SDF values of shape [batch_size, 1]
        """
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            y = layer(torch.cat([y, x], dim=1))
        return y

class MultiSDFNetwork(nn.Module):
    """
    Combined network for multiple SDFs (one per robot link)
    """
    def __init__(self, num_links, input_size=3, hidden_size=128, num_layers=8):
        """
        Args:
            num_links: Number of robot links
            input_size: Input dimension for each SDF
            hidden_size: Hidden layer size
            num_layers: Number of hidden layers per SDF
        """
        super(MultiSDFNetwork, self).__init__()
        
        # Create an SDF network for each link
        self.sdf_networks = nn.ModuleList([
            SDFNetwork(input_size, 1, hidden_size, num_layers)
            for _ in range(num_links)
        ])
    
    def forward(self, x):
        """
        Forward pass through all SDF networks.
        Args:
            x: List of input tensors, one per link
               Each tensor shape: [batch_size, input_size]
        Returns:
            List of SDF values, one per link
        """
        return [net(x_i) for net, x_i in zip(self.sdf_networks, x)]
