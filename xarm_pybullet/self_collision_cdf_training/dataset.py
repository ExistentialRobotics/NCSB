import torch
import numpy as np

class SelfCollisionCDFDataset:
    def __init__(self, configs, relevant_joints):
        """
        Args:
            configs: Array of joint configurations [N, 6]
            relevant_joints: List of relevant joint indices for each config
        """
        self.configs = torch.tensor(configs, dtype=torch.float32)
        self.relevant_joints = relevant_joints
        
    def __len__(self):
        return len(self.configs)
    
    def __getitem__(self, idx):
        config = self.configs[idx]
        
        # Create joint relevance mask (1 for relevant joints, 0 for others)
        joint_mask = torch.zeros(6, dtype=torch.float32)
        joint_mask[self.relevant_joints[idx]] = 1.0
        
        # Encode configuration with sin/cos
        encoded_config = torch.cat([config, torch.sin(config), torch.cos(config)])
        
        # Combine encoded config with joint mask
        input_data = torch.cat([encoded_config, joint_mask])
        
        return input_data, config 