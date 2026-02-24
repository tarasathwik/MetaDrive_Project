import os
import sys
import psutil
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from metadrive.envs import MetaDriveEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.safety import ActionMapWrapper, StateMapWrapper
class CompatWrapper(gym.Wrapper):
    """Fixes API mismatch by stripping 'options' and ensuring tuple returns."""
    def reset(self, *, seed=None, options=None):
        kwargs = {}
        if seed is not None:
            kwargs['seed'] = seed
            
        result = self.env.reset(**kwargs)
        
        # Gymnasium/SB3 requires an (obs, info) tuple format
        if isinstance(result, tuple) and len(result) == 2:
            return result
        else:
            return result, {}

# --- MAX PERFORMANCE CONFIGURATION ---
config = {
    "num_scenarios": 100,         
    "start_seed": 1000,
    "traffic_density": 0.07,
    "random_traffic": True,
    "use_render": False,          
    "image_observation": False,   
    "vehicle_config": {
        "lidar": {"num_lasers": 72, "distance": 40, "num_others": 4},
    }
}

def set_high_priority():
    p = psutil.Process(os.getpid())
    if sys.platform == "win32":
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    else:
        p.nice(-10)
    print(">>> High Performance Mode: CPU priority set to MAXIMUM.")

def train_rl_agent(timesteps=500000):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_load_path = os.path.join(base_dir, "models", "bc_pretrained_ppo.zip")
    log_dir = os.path.join(base_dir, "logs", "ppo_tensorboard") 
    
    set_high_priority()
    
    # 1. Initialize the raw environment
    env = MetaDriveEnv(config)
    
    # 2. Patch the environment to fix the API crash
    env = CompatWrapper(env)
    
    # 3. Apply Mathematical Safety Constraints
    print("Applying State and Action Mappers...")
    env = StateMapWrapper(env)
    env = ActionMapWrapper(env, max_friction=1.0)

    # 4. Load the model into the safe sandbox
    model = PPO.load(model_load_path, env=env, device="auto", tensorboard_log=log_dir, custom_objects={
        "learning_rate": 1e-4, 
        "clip_range": 0.2,      
        "n_steps": 2048,
        "batch_size": 128       
    })

    checkpoint_callback = CheckpointCallback(
        save_freq=25000, 
        save_path=os.path.join(base_dir, "models", "rl_checkpoints"),
        name_prefix="ppo_racing_constrained" 
    )

    print(f"\nSTARTING HIGH-SPEED PPO: {timesteps} Steps")
    print(f"Observe graphs at: {log_dir}")
    
    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback, reset_num_timesteps=False)
        model.save(os.path.join(base_dir, "models", "rl_final_constrained_agent"))
        print("\nSUCCESS: Training complete.")
    except KeyboardInterrupt:
        print("\nManually Interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    train_rl_agent()
