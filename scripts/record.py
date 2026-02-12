import sys
import os
import numpy as np
import argparse

# Allow importing modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metadrive.envs import MetaDriveEnv
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.constants import HELP_MESSAGE

# --- CONFIGURATION ---
config = {
    "use_render": True,
    "manual_control": True,
    "controller": "keyboard",
    "traffic_density": 0.05,   # Low traffic to make it easier to drive perfect lines
    "environment_num": 1,
    "start_seed": 1000,
    "map": "SCO",             # S=Straight, C=Curve, O=Roundabout
    "agent_policy": ManualControlPolicy,
    "window_size": (1200, 900),
}

def record_expert_data(target_steps=10000, save_dir="../data", filename="expert_data.npz"):
    
    # Initialize Environment
    env = MetaDriveEnv(config)
    
    # Data Buffers
    all_observations = []
    all_actions = []
    collected_steps = 0
    
    # Ensure data directory exists
    full_save_path = os.path.join(os.path.dirname(__file__), save_dir)
    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)
    
    target_file = os.path.join(full_save_path, filename)

    print(f"\n RECORDING STARTED: Goal = {target_steps} Steps")
    print(HELP_MESSAGE)
    print("-" * 40)

    try:
        obs, info = env.reset()
        episode_reward = 0
        
        while collected_steps < target_steps:
            env.render()
            
            # 1. Capture Action
            current_action = env.agent.controller.process_input(env.agent)
            
            # 2. Step Environment
            next_obs, reward, terminated, truncated, info = env.step(current_action)
            done = terminated or truncated
            
            # 3. Filter & Record (Only if moving)
            velocity = info.get('velocity', 0)
            
            # We use a small threshold (0.5 km/h) to avoid recording idle noise
            if velocity > 0.1: 
                all_observations.append(obs)
                all_actions.append(current_action)
                collected_steps += 1
                
                # Live Counter (Prints every 100 steps to avoid spam)
                if collected_steps % 100 == 0:
                    print(f"Example Progress: [{collected_steps}/{target_steps}] steps | Reward: {episode_reward:.1f}")

            obs = next_obs
            episode_reward += reward
            
            # Reset if crashed or lap finished
            if done:
                print(f"⚠️ Episode Reset! (Current Total: {collected_steps})")
                obs, info = env.reset()
                episode_reward = 0

    except KeyboardInterrupt:
        print("\n Recording stopped early by user.")
    
    finally:
        env.close()

    # Save Data
    if collected_steps > 0:
        print(f"\n Saving {collected_steps} data points...")
        np.savez(
            target_file, 
            obs=np.array(all_observations), 
            actions=np.array(all_actions)
        )
        print(f" SUCCESS: Data saved to {target_file}")
        print(f"Shape: {np.array(all_observations).shape}")
    else:
        print(" No data recorded.")

if __name__ == "__main__":
    # You can change the default steps here or via command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000, help="Number of steps to record")
    args = parser.parse_args()
    
    record_expert_data(target_steps=args.steps)
