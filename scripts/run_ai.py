import os
import sys
import psutil
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from metadrive.envs import MetaDriveEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    log_dir = os.path.join(base_dir, "logs", "ppo_tensorboard") # TensorBoard directory
    
    set_high_priority()
    env = MetaDriveEnv(config)

    # Adding tensorboard_log for real-time graph observation
    model = PPO.load(model_load_path, env=env, device="auto", tensorboard_log=log_dir, custom_objects={
        "learning_rate": 1e-4, 
        "clip_range": 0.2,      
        "n_steps": 2048,
        "batch_size": 128       
    })

    checkpoint_callback = CheckpointCallback(
        save_freq=25000, 
        save_path=os.path.join(base_dir, "models", "rl_checkpoints"),
        name_prefix="ppo_racing_max"
    )

    print(f"\nSTARTING HIGH-SPEED PPO: {timesteps} Steps")
    print(f"Observe graphs at: {log_dir}")
    
    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback, reset_num_timesteps=False)
        model.save(os.path.join(base_dir, "models", "rl_final_optimized_agent"))
        print("\nSUCCESS: Training complete.")
    except KeyboardInterrupt:
        print("\nManually Interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    train_rl_agent()
