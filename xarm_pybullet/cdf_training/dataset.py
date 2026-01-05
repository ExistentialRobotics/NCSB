import torch

class RobotCDFDataset:
    def __init__(self, configs, points, cdf_values):
        self.configs = torch.tensor(configs, dtype=torch.float32)
        self.points = torch.tensor(points, dtype=torch.float32)
        self.cdf_values = torch.tensor(cdf_values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.configs)
    
    def __getitem__(self, idx):
        config = self.configs[idx]
        point = self.points[idx]
        cdf_value = self.cdf_values[idx]
        
        encoded_config = torch.cat([config, torch.sin(config), torch.cos(config)])
        input_data = torch.cat([encoded_config, point])
        
        return input_data, cdf_value