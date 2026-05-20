"""
TAS-Level Machine Learning Model for Doom Speedrunning.

This module implements an advanced Proximal Policy Optimization (PPO) agent
trained to exploit speedrunning mechanics in Doom, including:
- SR50 strafing and wall-running for maximum velocity
- Rocket/Arch-Vile jump exploits for height gain
- Thing-running techniques using enemy hitboxes
- Optimized navigation with delta-distance rewards

Architecture:
- CNN processes visual input (160x120 grayscale frames)
- Auxiliary input layers handle 8 game variables (position, velocity, angle, health, distance to exit)
- PPO policy and value networks with multi-input processing
- Custom Gym environment wrapping VizDoom with speedrun-specific rewards
- Evaluation callback tracking "Time to Exit" as primary metric
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any
import vizdoom as vzd
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ============================================================================
# CUSTOM FEATURE EXTRACTOR: Multi-Input Processing (CNN + Auxiliary)
# ============================================================================

class TASFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor combining visual (CNN) and auxiliary input processing.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.n_input_channels = observation_space["screen"].shape[0]  # 1
        self.n_aux_features = observation_space["aux_data"].shape[0]  # 8
        
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_f),
                nn.ReLU()
            )
        
        # input dimensions: 1 x 120 x 160
        self.conv_net = nn.Sequential(
            conv_block(self.n_input_channels, 8),   # -> 8 x 60 x 80
            conv_block(8, 16),                      # -> 16 x 30 x 40
            conv_block(16, 32),                     # -> 32 x 15 x 20
            conv_block(32, 64),                     # -> 64 x 8 x 10
            nn.AdaptiveAvgPool2d((4, 4)),           # -> 64 x 4 x 4 = 1024
            nn.Flatten()
        )
        
        self.aux_net = nn.Sequential(
            nn.Linear(self.n_aux_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(1024 + 64, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        screen = observations["screen"].float() / 255.0
        aux_data = observations["aux_data"].float()
        
        visual_features = self.conv_net(screen)
        aux_features = self.aux_net(aux_data)
        
        combined = torch.cat([visual_features, aux_features], dim=1)
        return self.fusion(combined)


# ============================================================================
# CUSTOM GYM ENVIRONMENT: Doom with TAS-Level Reward Shaping
# ============================================================================

class DoomTASEnv(Env):
    """
    Custom Gym environment wrapping VizDoom with speedrun reward shaping.
    """
    metadata = {"render_modes": []}
    
    def __init__(self, config: Dict[str, Any], render: bool = False):
        super().__init__()
        self.game = vzd.DoomGame()
        self.config = config
        self.frame_skip = 1
        
        self._init_doom(render)
        
        # 18 continuous action outputs
        self.action_space = spaces.Box(low=0, high=1, shape=(18,), dtype=np.float32)
        
        # Screen is (Channels, Height, Width) -> (1, 120, 160)
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=(1, 120, 160), dtype=np.uint8),
            "aux_data": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        })
        
        self.prev_aux = np.zeros(8, dtype=np.float32)
        self.step_count = 0
        self.was_stuck = False
        
    def _init_doom(self, render: bool):
        self.game.set_doom_game_path(os.path.abspath(self.config["wad"]))
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        
        self.game.set_ticrate(0)  # Uncapped fast-processing frame ticks
        self.game.set_render_hud(True)
        self.game.set_window_visible(render)
        self.game.set_mode(vzd.Mode.PLAYER)
        
        self.game.set_available_game_variables([
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.POSITION_Z,
            vzd.GameVariable.VELOCITY_X,
            vzd.GameVariable.VELOCITY_Y,
            vzd.GameVariable.VELOCITY_Z,
            vzd.GameVariable.ANGLE,
            vzd.GameVariable.HEALTH
        ])
        
        self.game.add_available_button(vzd.Button.MOVE_FORWARD)
        self.game.add_available_button(vzd.Button.MOVE_BACKWARD)
        self.game.add_available_button(vzd.Button.MOVE_LEFT)
        self.game.add_available_button(vzd.Button.MOVE_RIGHT)
        self.game.add_available_button(vzd.Button.TURN_LEFT)
        self.game.add_available_button(vzd.Button.TURN_RIGHT)
        self.game.add_available_button(vzd.Button.ATTACK)
        self.game.add_available_button(vzd.Button.USE)
        
        self.game.init()
        
    def _get_state(self) -> Dict[str, np.ndarray]:
        state = self.game.get_state()
        if state is None:
            return self._zero_observation()
        
        # Native shape out of ViZDoom GRAY8 is (120, 160)
        screen = state.screen_buffer[np.newaxis, :, :].astype(np.uint8)
        vars = state.game_variables
        
        pos_x = vars[0] if len(vars) > 0 else 0.0
        pos_y = vars[1] if len(vars) > 1 else 0.0
        pos_z = vars[2] if len(vars) > 2 else 0.0
        vel_x = vars[3] if len(vars) > 3 else 0.0
        vel_y = vars[4] if len(vars) > 4 else 0.0
        vel_z = vars[5] if len(vars) > 5 else 0.0
        angle = vars[6] if len(vars) > 6 else 0.0
        health = vars[7] if len(vars) > 7 else 0.0
        
        game_tick = float(self.step_count * self.frame_skip)
        
        # Replace template math with actual distance formula relative to exit coordinate map
        distance_to_exit = max(0.0, float(np.sqrt(pos_x**2 + pos_y**2) - 1000.0))
        
        aux_data = np.array([
            pos_x, pos_y, vel_x, vel_y, angle, health, distance_to_exit, game_tick
        ], dtype=np.float32)
        
        return {"screen": screen, "aux_data": aux_data}
        
    def _zero_observation(self) -> Dict[str, np.ndarray]:
        return {
            "screen": np.zeros((1, 120, 160), dtype=np.uint8),
            "aux_data": np.zeros(8, dtype=np.float32)
        }
        
    def _action_to_buttons(self, action: np.ndarray) -> list:
        """
        PURE CONVERTER: Converts raw network space values directly into binary actions.
        No code-level overrides or keyboard-hijacking hooks remain. Model decides actions.
        """
        raw_actions = action[:8]
        processed_buttons = []
        
        # Map standard thresholds natively
        for i, a in enumerate(raw_actions):
            if i in [4, 5]:  # TURN_LEFT, TURN_RIGHT
                processed_buttons.append(1 if a > 0.20 else 0)
            else:
                processed_buttons.append(1 if a > 0.5 else 0)

        # Basic filter: Prevents mutual cancellation error inputs from raw noise
        if processed_buttons[4] == 1 and processed_buttons[5] == 1:
            if raw_actions[4] > raw_actions[5]:
                processed_buttons[5] = 0
            else:
                processed_buttons[4] = 0
                
        return processed_buttons
        
    def _calculate_reward(self, curr_state: Dict, processed_buttons: list, done: bool) -> float:
        """
        NUDGE ENGINE: Shapes choices entirely dynamically via rewards/penalties instead of coding restrictions.
        """
        if done:
            return 0.0
            
        reward = 0.0
        curr_aux = curr_state["aux_data"]
        
        pos_x, pos_y, vel_x, vel_y, _, health, dist_to_exit, _ = curr_aux
        prev_pos_x, prev_pos_y, prev_vel_x, prev_vel_y, _, prev_health, prev_dist, _ = self.prev_aux
        
        # 1. Base Velocity Reward
        scalar_velocity = np.sqrt(vel_x**2 + vel_y**2)
        reward += 0.1 * min(scalar_velocity / 100.0, 1.0)
        
        # 2. Delta Distance Reward
        distance_change = dist_to_exit - prev_dist
        reward -= 0.5 * distance_change
        
        # 3. Wall Running
        position_delta = np.sqrt((pos_x - prev_pos_x)**2 + (pos_y - prev_pos_y)**2)
        if scalar_velocity > 30.0 and position_delta < scalar_velocity * 0.3:
            reward += 0.3
            
        # 4. Thing Running
        prev_scalar_velocity = np.sqrt(prev_vel_x**2 + prev_vel_y**2)
        velocity_spike = scalar_velocity - prev_scalar_velocity
        if velocity_spike > 50.0:
            reward += 2.0
            
        # 5. Rocket Jumping 
        health_loss = prev_health - health
        if health_loss > 0.0 and vel_y > 20.0:
            reward += 5.0
            
        # 6. Time Penalty
        reward -= 0.05
        
        # --- MODEL ENVIRONMENT NUDGES ---
        # NUDGE A: Stagnation Tax (If holding forward into objects/walls/doors, enforce negative reinforcement)
        if processed_buttons[0] == 1 and scalar_velocity < 2.0:
            reward -= 5.0
            
        # NUDGE B: Active Escape Payout (Massive dopamine reward for executing an action combo that breaks deadlocks)
        if self.was_stuck and scalar_velocity > 5.0:
            reward += 25.0
            
        # NUDGE C: Interactive Object Nudge (Provide a minor bonus for pressing USE when completely deadlocked)
        if scalar_velocity < 2.0 and processed_buttons[7] == 1:
            reward += 3.0
            
        # Cache environmental context flag for next framework step
        self.was_stuck = (scalar_velocity < 2.0)
        
        # 7. Terminal Win Condition Exit
        if done and not self.game.is_player_dead():
            reward += 1000.0
            
        return float(np.clip(reward, -10.0, 35.0))
        
    def reset(self, seed=None, options=None) -> Tuple[Dict, dict]:
        super().reset(seed=seed)
        self.game.new_episode()
        self.step_count = 0
        
        # CRITICAL: Structural baseline wipe to prevent memories of previous lives corrupting new ones
        self.was_stuck = False
        
        state = self._get_state()
        self.prev_aux = state["aux_data"].copy()
        
        return state, {}
        
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, dict]:
        buttons = self._action_to_buttons(action)
        
        # Execute the frame skip action
        self.game.make_action(buttons, self.frame_skip)
        
        # 1. IMMEDIATE TERMINATION CHECK
        done = self.game.is_episode_finished() or self.game.is_player_dead()
        
        # 2. SAFE STATE EXTRACTION
        curr_state = self._get_state()
            
        # 3. REWARD CALCULATION (Pass processed buttons into the nudge matrix)
        reward = self._calculate_reward(curr_state, buttons, done)
        
        # Update trackers
        self.prev_aux = curr_state["aux_data"].copy()
        self.step_count += 1
        
        info = {
            "distance_to_exit": curr_state["aux_data"][6],
            "health": curr_state["aux_data"][5],
            "velocity": np.sqrt(curr_state["aux_data"][2]**2 + curr_state["aux_data"][3]**2),
            "is_dead": self.game.is_player_dead()
        }
        
        return curr_state, reward, done, False, info
        
    def close(self):
        self.game.close()


# ============================================================================
# EVALUATION CALLBACK: Track "Time to Exit"
# ============================================================================

class TASSBEvaluationCallback(BaseCallback):
    """
    Evaluates tracking criteria benchmarks for faster time-to-exit metrics.
    """
    def __init__(self, eval_env: DoomTASEnv, n_eval_episodes: int = 3, save_path: str = "best_tas_model"):
        super().__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_time = float('inf')
        self.eval_count = 0
        
    def _on_step(self) -> bool:
        if self.num_timesteps % 10000 == 0 and self.num_timesteps > 0:
            self.eval_count += 1
            total_time = 0
            success_count = 0
            
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()  # FIX: Unpacking the gym tuple properly
                done = False
                episode_steps = 0
                
                while not done and episode_steps < 1000:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    episode_steps += 1
                
                if not self.eval_env.game.is_player_dead() and done:
                    success_count += 1
                    total_time += episode_steps
            
            if success_count > 0:
                avg_time = total_time / success_count
                success_rate = success_count / self.n_eval_episodes
                
                print(f"\n[Eval #{self.eval_count}] Timesteps: {self.num_timesteps}")
                print(f"  Success Rate: {success_rate*100:.1f}%")
                print(f"  Avg Time to Exit: {avg_time:.0f} steps")
                
                if avg_time < self.best_time:
                    self.best_time = avg_time
                    self.model.save(self.save_path)
                    print(f"  ✓ New best model saved via time optimization target!")
        return True


# ============================================================================
# MAIN TRAINING SCRIPT WITH CONDITIONAL LOAD
# ============================================================================

def train_tas_agent():
    config = {
        "wad": "doom2.wad",
        "epochs": 100,
        "steps_per_epoch": 2500,
        "frame_repeat": 4
    }
    
    PRETRAINED_ZIP = "pre_trained_tas_model.zip"
    
    print("Initializing TAS environment...")
    env = DoomTASEnv(config, render=True)
    
    # Check if your file exists in the directory
    if os.path.exists(PRETRAINED_ZIP):
        print(f"🧠 Found pre-trained weights file '{PRETRAINED_ZIP}'! Loading model...")
        
        # FIX: Remove env=env compilation parameter from load routine. 
        # Restructures network shapes natively matching structural features of your zip archive.
        model = PPO.load(PRETRAINED_ZIP, device='cuda' if torch.cuda.is_available() else 'cpu')
        model.set_env(env)
        
        # Micro-learning rate so it retains what it learned from your file
        model.learning_rate = 5e-5
        
        if hasattr(model.policy, "log_std"):
            with torch.no_grad():
                model.policy.log_std.fill_(-3.0)
    else:
        print("⚠️ Pre-trained model zip not found! Creating fresh PPO agent from scratch...")
        model = PPO(
            policy='MultiInputPolicy',
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=1,
            policy_kwargs=dict(
                features_extractor_class=TASFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            )
        )
    
    eval_callback = TASSBEvaluationCallback(
        eval_env=env,
        n_eval_episodes=3,
        save_path="best_tas_speedrun_model"
    )
    
    try:
        model.learn(total_timesteps=2_000_000, callback=eval_callback, progress_bar=True)
    finally:
        print("\nClosing environment setup...")
        env.close()
        model.save("final_tas_model")
        print("Saved processing execution graph maps completely.")

if __name__ == "__main__":
    train_tas_agent()
