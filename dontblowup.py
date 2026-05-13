import vizdoom as vzd
import numpy as np
from stable_baselines3 import PPO
from brainfu_tas_refactored import DoomTASEnv # Import your class

def play_best_model():
    # 1. Setup minimal config
    config = {"wad": "doom2.wad", "frame_repeat": 4}
    
    # 2. Init env with rendering ON so people can see it
    env = DoomTASEnv(config, render=True)
    
    # 3. Load the pre-trained brain
    print("Loading TAS model...")
    model = PPO.load("best_tas_model.zip")
    
    # 4. Run the demo loop
    obs, info = env.reset()
    while True:
        # deterministic=True is key for presentations!
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    play_best_model()
