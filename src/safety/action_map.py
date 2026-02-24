import gymnasium as gym
import numpy as np

class ActionMapWrapper(gym.ActionWrapper):
    """
    Action Mapping: Enforces the Friction Circle Constraint.
    Restricts steering and acceleration so the combined force vector 
    never exceeds the physical limits of the tires.
    """
    def __init__(self, env, max_friction=1.0):
        super().__init__(env)
        self.max_friction = max_friction

    def action(self, action):
        # Calculate the magnitude of the requested force vector [steer, throttle]
        force_magnitude = np.linalg.norm(action)
        
        # H_AM: Project mathematically impossible actions back onto the safe boundary
        if force_magnitude > self.max_friction:
            action = (action / force_magnitude) * self.max_friction
            
        # Ensure the final output is within the environment's absolute bounds
        return np.clip(action, self.action_space.low, self.action_space.high)