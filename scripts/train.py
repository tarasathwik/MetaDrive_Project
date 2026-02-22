import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from metadrive.envs import MetaDriveEnv
import sys

# Allow importing modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- MATCHING CONFIGURATION ---
# We must initialize a dummy environment with the exact same config
# so SB3 builds the correct neural network sizes for the LiDAR and Action spaces.
config = {
    "num_scenarios": 50,
    "start_seed": 1000,
    "traffic_density": 0.07,
    "random_traffic": True,
    "traffic_mode": "respawn",
    "vehicle_config": {
        "lidar": {"num_lasers": 72, "distance": 40, "num_others": 4},
        "show_lidar": False,
        "enable_reverse": True,
    }
}

def train_behavioral_cloning(data_path="../data/user_driving_data.pkl", model_dir="../models", epochs=15, batch_size=256):
    
    # Resolve absolute paths based on the script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_data_path = os.path.join(base_dir, "data", "user_driving_data.pkl")
    full_model_dir = os.path.join(base_dir, "models")
    
    print(f"Loading data from {full_data_path}...")
    with open(full_data_path, "rb") as f:
        dataset_raw = pickle.load(f)
    
    states = torch.tensor(np.array([item['state'] for item in dataset_raw]), dtype=torch.float32)
    actions = torch.tensor(np.array([item['action'] for item in dataset_raw]), dtype=torch.float32)
    
    print(f"Loaded {len(states)} state-action pairs.")
    
    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("\nInitializing MetaDrive environment for architecture matching...")
    env = MetaDriveEnv(config)
    
    print("Instantiating SB3 PPO Model...")
    # Initialize PPO using "auto" device to seamlessly handle your local hardware
    model = PPO("MlpPolicy", env, verbose=0, device="auto")
    policy = model.policy
    
    # Standard supervised learning optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    print("\nStarting Behavioral Cloning Phase (Negative Log-Likelihood)...")
    policy.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            
            optimizer.zero_grad()
            
            # policy.evaluate_actions returns: state values, log probabilities, and entropy
            _, log_prob, _ = policy.evaluate_actions(batch_states, batch_actions)
            
            # Maximize the probability of the expert actions by minimizing the negative log_prob
            loss = -log_prob.mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | NLL Loss: {avg_loss:.4f}")
        
    env.close()
    
    # Ensure model directory exists
    if not os.path.exists(full_model_dir):
        os.makedirs(full_model_dir)
        
    save_path = os.path.join(full_model_dir, "bc_pretrained_ppo")
    model.save(save_path)
    print(f"\nSUCCESS: Pre-trained stochastic PPO model saved to {save_path}.zip")

if __name__ == "__main__":
    import numpy as np # Added for quick conversion
    train_behavioral_cloning()
