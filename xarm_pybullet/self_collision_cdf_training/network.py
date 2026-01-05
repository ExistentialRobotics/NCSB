import torch
import torch.nn as nn

class SelfCollisionCDFNetwork(nn.Module):
    def __init__(
        self,
        input_dims=18,  # 6 (joints) * 3 (original + sin + cos)
        output_dims=1,
        hidden_dims=[512, 256, 128, 64],
        skip_in=(2,),
        activation='relu'
    ):
        super(SelfCollisionCDFNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        dims = [input_dims] + hidden_dims + [output_dims]
        
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - input_dims
            else:
                out_dim = dims[layer + 1]
                
            lin = nn.Linear(dims[layer], out_dim)
            setattr(self, f"lin{layer}", lin)

        self.activation_dict = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
        }
        self.activation = self.activation_dict[activation.lower()]
        
    def forward(self, x):
        input_x = x
        
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, f"lin{layer}")
            
            if layer in self.skip_in:
                x = torch.cat([x, input_x], -1)
            
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
        
        return torch.abs(x)  # Ensure positive distances 