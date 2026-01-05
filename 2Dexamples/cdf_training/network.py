import torch
import torch.nn as nn

class CDFNetwork(nn.Module):
    def __init__(
        self,
        input_dims=8,  # 2 (joints) * 3 (original + sin + cos) + 2 (2D point)
        output_dims=1,
        hidden_dims=[512, 256, 128, 64, 32],
        skip_in=(2, 4),
        use_skip_connections=True,
        activation='relu'
    ):
        super(CDFNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        dims = [input_dims] + hidden_dims + [output_dims]
        
        self.num_layers = len(dims)
        self.skip_in = skip_in if use_skip_connections else ()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - input_dims
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)
            setattr(self, f"lin{layer}", lin)

        # Dictionary of available activation functions
        self.activation_dict = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'mish': nn.Mish(),
            'silu': nn.SiLU(),
        }
        
        # Set activation function based on name
        if activation.lower() not in self.activation_dict:
            raise ValueError(f"Activation {activation} not supported. Choose from: {list(self.activation_dict.keys())}")
        
        self.activation = self.activation_dict[activation.lower()]
        
    def forward(self, x):
        input = x
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, f"lin{layer}")
            if layer in self.skip_in:
                x = torch.cat([x, input], -1)
            x = lin(x)
            
            if layer < self.num_layers - 2:
                x = self.activation(x)
        
        # Apply absolute value to ensure positive distances
        x = torch.abs(x)
        return x 

class CDFNetworkWithDropout(nn.Module):
    def __init__(
        self,
        input_dims=8,  # 2 (joints) * 3 (original + sin + cos) + 2 (2D point)
        output_dims=1,
        hidden_dims=[512, 256, 128, 64, 32],
        skip_in=(2, 4),
        use_skip_connections=True,
        activation='relu',
        dropout_rate=0.1
    ):
        super(CDFNetworkWithDropout, self).__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        dims = [input_dims] + hidden_dims + [output_dims]
        
        self.num_layers = len(dims)
        self.skip_in = skip_in if use_skip_connections else ()

        # Add dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - input_dims
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)
            setattr(self, f"lin{layer}", lin)

        # Dictionary of available activation functions
        self.activation_dict = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'mish': nn.Mish(),
            'silu': nn.SiLU(),
        }
        
        # Set activation function based on name
        if activation.lower() not in self.activation_dict:
            raise ValueError(f"Activation {activation} not supported. Choose from: {list(self.activation_dict.keys())}")
        
        self.activation = self.activation_dict[activation.lower()]
        
    def forward(self, x):
        input = x
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, f"lin{layer}")
            if layer in self.skip_in:
                x = torch.cat([x, input], -1)
            x = lin(x)
            
            if layer < self.num_layers - 2:
                x = self.activation(x)
                x = self.dropout(x)  # Apply dropout after activation
        
        # Apply absolute value to ensure positive distances
        x = torch.abs(x)
        return x 