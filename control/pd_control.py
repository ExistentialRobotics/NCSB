import numpy as np
import torch

class PDController:
    def __init__(self, kp=10.0, kd=2.0, control_limits=1.0):
        """Simple PD Controller for joint space control.
        
        Args:
            kp (float): Proportional gain
            kd (float): Derivative gain
            max_velocity (float): Maximum joint velocity
        """
        self.kp = kp
        self.kd = kd
        self.max_velocity = control_limits
    
    def compute_control(self, current_pos, target_pos, current_vel):
        """Compute control input using PD control law.
        
        Args:
            current_pos: Current joint positions
            target_pos: Target joint positions
            current_vel: Current joint velocities
            
        Returns:
            velocity_cmd: Commanded joint velocities
        """
        # Compute position error
        pos_error = target_pos - current_pos
        
        # PD control law
        velocity_cmd = self.kp * pos_error - self.kd * current_vel
        
        # Clip velocities
        if isinstance(velocity_cmd, torch.Tensor):
            velocity_cmd = torch.clamp(velocity_cmd, -self.max_velocity, self.max_velocity)
        else:
            velocity_cmd = np.clip(velocity_cmd, -self.max_velocity, self.max_velocity)
            
        return velocity_cmd 