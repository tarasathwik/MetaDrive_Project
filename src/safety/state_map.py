import gymnasium as gym
import numpy as np

class StateMapWrapper(gym.ObservationWrapper):
    """
    State Mapping (Continuous): Acts as a safety filter for the LiDAR array.
    Instead of binarizing the data (which breaks pre-trained BC weights),
    this mathematically bounds the continuous distances to ensure stability 
    while preserving the agent's depth perception.
    """
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # Preserve the continuous decimals, but strictly clip them to safety bounds
        if isinstance(obs, dict) and 'lidar' in obs:
            obs['lidar'] = np.clip(obs['lidar'], 0.0, 1.0)
        else:
            obs = np.clip(obs, 0.0, 1.0)
            
        return obs