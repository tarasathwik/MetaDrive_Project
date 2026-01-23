import gymnasium as gym
from metadrive.envs import MetaDriveEnv
import keyboard
import numpy as np

# --- TUNING ---
STEER_SPEED = 0.08      
STEER_RETURN = 0.15     
THROTTLE_SPEED = 0.05   
BRAKE_SPEED = 0.2       

def run_arcade_traffic():
    config = {
        "num_scenarios": 50,
        "start_seed": 1000,
        
        # --- INCREASED TRAFFIC ---
        # Changed from 0.1 to 0.15
        # Be careful overtaking!
        "traffic_density": 0.15,
        "random_traffic": True,
        
        "manual_control": False,
        "use_render": True,
        "window_size": (1000, 800),
        "vehicle_config": {
            "lidar": {"num_lasers": 72, "distance": 40, "num_others": 4},
            "show_lidar": True,
            "enable_reverse": True,
        }
    }
    
    print("Initializing Arcade Traffic Mode...")
    env = MetaDriveEnv(config)
    
    current_steer = 0.0
    current_throttle = 0.0
    
    try:
        obs, info = env.reset()
        print("\n" + "="*40)
        print("  ARCADE MODE: HEAVY TRAFFIC")
        print("  ---------------------")
        print("  Density: 0.15 (Watch your mirrors!)")
        print("  Controls: Arrows + Space")
        print("="*40 + "\n")
        
        while True:
            # --- 1. SMOOTH STEERING ---
            target_steer = 0.0
            if keyboard.is_pressed('left'):  target_steer = 1.0
            if keyboard.is_pressed('right'): target_steer = -1.0
            
            if target_steer > current_steer:
                current_steer = min(target_steer, current_steer + STEER_SPEED)
            elif target_steer < current_steer:
                current_steer = max(target_steer, current_steer - STEER_SPEED)
            
            if target_steer == 0:
                if current_steer > 0:
                    current_steer = max(0, current_steer - STEER_RETURN)
                elif current_steer < 0:
                    current_steer = min(0, current_steer + STEER_RETURN)

            # --- 2. SMOOTH THROTTLE ---
            speed = env.vehicle.speed_km_h
            if speed < 0.1 and speed > -0.1: speed = 0.0
            
            # Direction Check
            heading = np.array([np.cos(env.vehicle.heading_theta), np.sin(env.vehicle.heading_theta)])
            velocity = env.vehicle.velocity
            is_reversing = np.dot(velocity, heading) < 0
            if speed > 0 and is_reversing: speed = -speed

            target_throttle = 0.0
            
            if keyboard.is_pressed('up'):
                target_throttle = 1.0
            elif keyboard.is_pressed('down'):
                target_throttle = -1.0
            
            # Brake Logic
            if keyboard.is_pressed('space'):
                if speed > 2.0: target_throttle = -1.0   
                elif speed < -2.0: target_throttle = 1.0 
                else: target_throttle = 0.0              
                current_throttle = target_throttle 
            else:
                if target_throttle > current_throttle:
                    current_throttle = min(target_throttle, current_throttle + THROTTLE_SPEED)
                elif target_throttle < current_throttle:
                    current_throttle = max(target_throttle, current_throttle - THROTTLE_SPEED)
                
                if target_throttle == 0:
                    current_throttle *= 0.9 
            
            # --- 3. APPLY ---
            action = [current_steer, current_throttle]
            env.step(action)
            
            env.render(
                text={
                    "Status": "Heavy Traffic",
                    "Speed": f"{speed:.1f} km/h",
                    "Input": f"S:{current_steer:.2f} T:{current_throttle:.2f}"
                }
            )
            
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    run_arcade_traffic()