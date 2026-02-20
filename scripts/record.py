import sys
import os
import argparse
import pickle
import numpy as np

# Allow importing modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metadrive.envs import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy

# --- ALIGNED CONFIGURATION ---
config = {
    "num_scenarios": 50,
    "start_seed": 1000,
    
    # --- HEAVY TRAFFIC (Matches play_game.py) ---
    "traffic_density": 0.07,
    "random_traffic": True,
    "traffic_mode": "respawn",
    
    # --- EXPERT OVERRIDES ---
    "manual_control": False,   # Must be False for the PID Expert to drive
    "use_render": False,       # Headless mode for rapid data generation
    
    "window_size": (1000, 800),
    "vehicle_config": {
        "lidar": {"num_lasers": 72, "distance": 40, "num_others": 4},
        "show_lidar": False,
        "enable_reverse": True,
    }
}

def record_expert_data(target_steps=50000, save_dir="training_data", filename="user_driving_data.pkl"):
    
    # Initialize Environment
    env = MetaDriveEnv(config)
    
    # Instantiate the Built-in PID Expert
    expert = ExpertPolicy()
    
    # Data Buffer
    dataset = []
    collected_steps = 0
    
    # Ensure data directory exists based on repo structure
    full_save_path = os.path.join(os.path.dirname(__file__), save_dir)
    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)
    
    target_file = os.path.join(full_save_path, filename)

    print(f"\n AUTOMATED RECORDING STARTED: Goal = {target_steps} Steps")
    print("-" * 40)

    try:
        obs, info = env.reset()
        
        while collected_steps < target_steps:
            # 1. Ask the PID expert for the mathematically optimal action
            current_action = expert(env.vehicle)
            
            # 2. Step Environment
            next_obs, reward, terminated, truncated, info = env.step(current_action)
            done = terminated or truncated
            
            # 3. Filter & Record
            # Use speed threshold to avoid recording idle/starting noise
            speed = env.vehicle.speed_km_h
            if speed > 0.5: 
                dataset.append({
                    "state": obs,
                    "action": current_action
                })
                collected_steps += 1
                
                # Live Counter
                if collected_steps % 1000 == 0:
                    print(f"Progress: [{collected_steps}/{target_steps}] frames recorded")

            obs = next_obs
            
            # Reset if crashed or lap finished
            if done:
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n Recording stopped early by user.")
    
    finally:
        env.close()

    # Save Data
    if collected_steps > 0:
        print(f"\n Saving {collected_steps} state-action pairs...")
        with open(target_file, "wb") as f:
            pickle.dump(dataset, f)
        print(f" SUCCESS: Expert data saved to {target_file}")
    else:
        print(" No data recorded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000, help="Number of steps to record")
    args = parser.parse_args()
    
    record_expert_data(target_steps=args.steps)
