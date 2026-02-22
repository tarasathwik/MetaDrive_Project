import os
import sys
from stable_baselines3 import PPO
from metadrive.envs import MetaDriveEnv

# Ensure script can find the environment modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- VISUAL EVALUATION CONFIGURATION ---
config = {
    "num_scenarios": 20,           # Test across different track layouts
    "start_seed": 5000,            # Use a new seed to test generalization
    "traffic_density": 0.1,        # Slightly more traffic to test overtaking
    "random_traffic": True,
    "use_render": True,            # ACTIVATE 3D RENDERING
    "manual_control": False,       # Let the AI take the wheel
    "window_size": (1200, 900),
    "vehicle_config": {
        "show_lidar": True,        # See what the AI "sees"
        "lidar": {"num_lasers": 72, "distance": 40, "num_others": 4},
    }
}

def watch_final_agent():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Loading the final optimized agent from your models folder
    model_path = os.path.join(base_dir, "models", "rl_final_optimized_agent.zip")

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    print("Initializing MetaDrive 3D Environment...")
    env = MetaDriveEnv(config)

    print(f"Loading Final Optimized Weights: {model_path}")
    model = PPO.load(model_path, env=env)

    try:
        obs, info = env.reset()
        print("\n" + "="*40)
        print("  AI RACING: LIVE DEMONSTRATION")
        print("  (Press 'W/S' to change camera view)")
        print("="*40 + "\n")

        for _ in range(5):  # Run for 5 full episodes
            done = False
            total_reward = 0
            while not done:
                # Use deterministic=True for the smoothest racing lines
                action, _states = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

                # Render the frame
                env.render(
                    text={
                        "Agent": "Final PPO Optimized",
                        "Reward": f"{total_reward:.2f}",
                        "Speed": f"{env.vehicle.speed_km_h:.1f} km/h"
                    }
                )
            
            print(f"Episode Finished. Reward: {total_reward:.2f}")
            obs, info = env.reset()

    except KeyboardInterrupt:
        print("\nDemonstration stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    watch_final_agent()