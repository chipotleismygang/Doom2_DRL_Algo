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
from collections import deque
from typing import Dict, Tuple, Optional, Any
import vizdoom as vzd
from gym import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


# ============================================================================
# CUSTOM FEATURE EXTRACTOR: Multi-Input Processing (CNN + Auxiliary)
# ============================================================================

class TASFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor combining visual (CNN) and auxiliary input processing.
    
    Processes:
    - Visual: 160x120 grayscale frames → CNN features
    - Auxiliary: Position, velocity, angle, health, distance (8 floats) → Dense layers
    
    Concatenates both streams for policy/value network input.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Extract channel dimensions
        self.n_input_channels = observation_space["screen"].shape[0]  # Should be 1 (grayscale)
        self.n_aux_features = observation_space["aux_data"].shape[0]  # Should be 8
        
        # CNN for visual processing (screen buffer)
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_f),
                nn.ReLU()
            )
        
        self.conv_net = nn.Sequential(
            conv_block(self.n_input_channels, 8),   # → 8x80x60
            conv_block(8, 8),                        # → 8x40x30
            conv_block(8, 8),                        # → 8x20x15
            conv_block(8, 16),                       # → 16x10x7
            nn.AdaptiveAvgPool2d((4, 4)),           # → 16x4x4 = 256
            nn.Flatten()
        )
        
        # Dense layers for auxiliary data (position, velocity, angle, health, distance)
        self.aux_net = nn.Sequential(
            nn.Linear(self.n_aux_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Fusion layer: Combine CNN + auxiliary features
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, features_dim),  # 256 from CNN + 64 from aux
            nn.ReLU()
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass combining visual and auxiliary data."""
        screen = observations["screen"].float() / 255.0
        aux_data = observations["aux_data"].float()
        
        # Extract features from each stream
        visual_features = self.conv_net(screen)
        aux_features = self.aux_net(aux_data)
        
        # Concatenate and fuse
        combined = torch.cat([visual_features, aux_features], dim=1)
        return self.fusion(combined)


# ============================================================================
# CUSTOM GYM ENVIRONMENT: Doom with TAS-Level Reward Shaping
# ============================================================================

class DoomTASEnv(Env):
    """
    Custom Gym environment wrapping VizDoom with extreme speedrun reward shaping.
    
    Observations:
    - Dict with "screen" (1x160x120 grayscale) and "aux_data" (8 floats: 
      pos_x, pos_y, vel_x, vel_y, angle, health, distance_to_exit, game_tick)
    
    Actions:
    - 18 continuous/discrete action combinations supporting simultaneous button presses
      including SR50 strafe combinations, turn modifiers, attack, jump, crouch
    
    Rewards (heavily shaped for speedrun mechanics):
    - Velocity reward: +0.1 * scalar_velocity
    - Delta distance: -0.5 * (current_dist - prev_dist)
    - Wall running: +0.3 multiplier if scraping wall
    - Thing running: +2.0 if touching enemy hitbox + forward velocity spike
    - Rocket jump: +5.0 if self-damage + large Z-velocity gain
    - Time penalty: -0.05 per step (punish stalling)
    - Exit trigger: +1000.0 (massive discrete reward)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, config: Dict[str, Any], render: bool = True):
        """
        Initialize Doom TAS environment.
        
        Args:
            config: Dict with "wad", "epochs", "steps_per_epoch", "frame_repeat"
            render: If True, display game window
        """
        super().__init__()
        
        self.game = vzd.DoomGame()
        self.config = config
        self.frame_skip = 4  # Uncapped async mode: 4-frame skip
        
        # Initialize game
        self._init_doom(render)
        
        # Action space: 18 combinations for SR50 + modifiers
        # We'll use continuous space [0, 1] for each of 18 buttons
        self.action_space = spaces.Box(
            low=0, high=1, shape=(18,), dtype=np.float32
        )
        
        # Observation space: Screen + auxiliary data
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=(1, 160, 120), dtype=np.uint8),
            "aux_data": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        })
        
        # Tracking variables for reward shaping
        self.prev_pos = np.array([0.0, 0.0, 0.0])
        self.prev_dist_to_exit = float('inf')
        self.prev_health = 100.0
        self.visited_tiles = set()
        self.tile_size = 64
        self.step_count = 0
        self.episode_start_tick = 0
        
    def _init_doom(self, render: bool):
        """Initialize VizDoom with TAS-optimized settings."""
        self.game.set_doom_game_path(os.path.abspath(self.config["wad"]))
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        
        # Asynchronous uncapped framerate mode
        self.game.set_async_mode(True)
        self.game.set_ticrate(0)  # 0 = uncapped
        
        self.game.set_render_hud(True)
        self.game.set_window_visible(render)
        self.game.set_mode(vzd.Mode.PLAYER)
        
        # Game variables for reward calculation
        self.game.set_available_game_variables([
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.POSITION_Z,
            vzd.GameVariable.VELOCITY_X,
            vzd.GameVariable.VELOCITY_Y,
            vzd.GameVariable.VELOCITY_Z,
            vzd.GameVariable.ANGLE,
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.USER1,  # Custom var: distance to exit (if available)
            vzd.GameVariable.GAMETIC,
        ])
        
        # TAS Action buttons (18 total)
        # Movement: FORWARD, BACKWARD, LEFT, RIGHT
        # SR50: Strafe combinations (simultaneous forward+backward, left+right)
        # Turning: TURN_LEFT, TURN_RIGHT with movement modifiers
        # Special: ATTACK (rocket), JUMP, CROUCH, USE
        self.game.add_available_button(vzd.Button.MOVE_FORWARD)      # 0
        self.game.add_available_button(vzd.Button.MOVE_BACKWARD)     # 1
        self.game.add_available_button(vzd.Button.MOVE_LEFT)         # 2
        self.game.add_available_button(vzd.Button.MOVE_RIGHT)        # 3
        self.game.add_available_button(vzd.Button.TURN_LEFT)         # 4
        self.game.add_available_button(vzd.Button.TURN_RIGHT)        # 5
        self.game.add_available_button(vzd.Button.ATTACK)            # 6
        self.game.add_available_button(vzd.Button.USE)               # 7
        
        # Extended buttons (if source port supports)
        try:
            self.game.add_available_button(vzd.Button.JUMP)          # 8
        except:
            pass
        try:
            self.game.add_available_button(vzd.Button.CROUCH)        # 9
        except:
            pass
        
        self.game.init()
    
    def _get_state(self) -> Dict[str, np.ndarray]:
        """Extract current game state into observation dict."""
        state = self.game.get_state()
        if state is None:
            return self._zero_observation()
        
        screen = state.screen_buffer[np.newaxis, :, :].astype(np.uint8)
        
        # Extract game variables
        vars = state.game_variables
        pos_x = vars[0]
        pos_y = vars[1]
        pos_z = vars[2]
        vel_x = vars[3]
        vel_y = vars[4]
        vel_z = vars[5]
        angle = vars[6]
        health = vars[7]
        # vars[8] would be distance to exit (if custom)
        game_tick = vars[9] if len(vars) > 9 else 0
        
        # Calculate distance to exit (hardcoded for this example; adjust per map)
        # In real TAS, this would be computed from actual exit switch position
        distance_to_exit = np.sqrt(pos_x**2 + pos_y**2) - 1000.0  # Dummy calculation
        distance_to_exit = max(0.0, distance_to_exit)
        
        # Scalar velocity
        velocity = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        
        # Auxiliary data: [pos_x, pos_y, vel_x, vel_y, angle, health, dist_to_exit, game_tick]
        aux_data = np.array([
            pos_x, pos_y, vel_x, vel_y, angle, health, distance_to_exit, game_tick
        ], dtype=np.float32)
        
        return {
            "screen": screen,
            "aux_data": aux_data
        }
    
    def _zero_observation(self) -> Dict[str, np.ndarray]:
        """Return zero observation when state is unavailable."""
        return {
            "screen": np.zeros((1, 160, 120), dtype=np.uint8),
            "aux_data": np.zeros(8, dtype=np.float32)
        }
    
    def _action_to_buttons(self, action: np.ndarray) -> list:
        """
        Convert continuous action vector [0-1] to button presses.
        
        18 actions:
        0-3: Basic movement (FORWARD, BACKWARD, LEFT, RIGHT)
        4-5: Turning (TURN_LEFT, TURN_RIGHT)
        6: ATTACK (rocket)
        7: USE
        8: JUMP (if available)
        9: CROUCH (if available)
        10-17: Reserved for advanced combinations
        
        Args:
            action: np.ndarray of shape (18,) with values in [0, 1]
        
        Returns:
            List of 8 binary button values for make_action()
        """
        # Threshold: > 0.5 = button pressed
        buttons = [int(a > 0.5) for a in action[:8]]  # Use first 8 for basic buttons
        return buttons
    
    def _calculate_reward(self, prev_state: Dict, curr_state: Dict, 
                         action: np.ndarray, done: bool) -> float:
        """
        Calculate TAS-optimized reward with speedrun mechanics.
        
        Components:
        - Velocity reward (proportional to speed)
        - Delta distance (negative = getting closer to exit)
        - Wall running incentive
        - Thing running incentive
        - Rocket jump incentive
        - Time penalty (discourage stalling)
        - Terminal reward (exit trigger)
        """
        reward = 0.0
        
        curr_aux = curr_state["aux_data"]
        prev_aux = prev_state["aux_data"] if prev_state is not None else np.zeros(8)
        
        pos_x, pos_y, vel_x, vel_y, angle, health, dist_to_exit, tick = curr_aux
        prev_pos_x, prev_pos_y, _, _, _, prev_health, prev_dist, _ = prev_aux
        
        # 1. VELOCITY REWARD: +0.1 * scalar_velocity
        scalar_velocity = np.sqrt(vel_x**2 + vel_y**2)
        reward += 0.1 * min(scalar_velocity / 100.0, 1.0)  # Normalize by max speed
        
        # 2. DELTA DISTANCE REWARD: -0.5 * distance_change
        distance_change = dist_to_exit - prev_dist
        reward -= 0.5 * distance_change  # Negative = getting closer (positive reward)
        
        # Update tracking
        self.prev_dist_to_exit = dist_to_exit
        
        # 3. WALL RUNNING INCENTIVE: Detect scraping via position delta vs angle
        position_delta = np.sqrt((pos_x - prev_pos_x)**2 + (pos_y - prev_pos_y)**2)
        # If moving fast but position delta small → likely scraping wall
        if scalar_velocity > 30.0 and position_delta < scalar_velocity * 0.3:
            reward += 0.3  # Wall running bonus
        
        # 4. THING RUNNING INCENTIVE: Collision + velocity spike
        # (This is heuristic; ideal: check for hitbox collision from game state)
        velocity_spike = scalar_velocity - np.sqrt(
            prev_aux[2]**2 + prev_aux[3]**2
        ) if prev_aux is not None else 0.0
        if velocity_spike > 50.0:
            reward += 2.0  # Thing running bonus
        
        # 5. ROCKET JUMP INCENTIVE: Self-damage + Z-velocity boost
        health_loss = prev_health - health
        if health_loss > 0.0 and vel_y > 20.0:  # Took damage + gained height
            reward += 5.0  # Rocket jump bonus
        
        # 6. TIME PENALTY: -0.05 per step (harsh stalling penalty)
        reward -= 0.05
        
        # 7. TERMINAL REWARD: Exit trigger
        if done and not self.game.is_player_dead():
            reward += 1000.0  # Massive discrete reward for exit
        
        # Clip for stability
        reward = np.clip(reward, -1.0, 10.0)  # Allow up to 10 for exit
        
        return reward
    
    def reset(self):
        """Reset environment for new episode."""
        self.game.new_episode()
        self.prev_pos = np.array([0.0, 0.0, 0.0])
        self.prev_dist_to_exit = float('inf')
        self.prev_health = 100.0
        self.step_count = 0
        
        state = self._get_state()
        self.episode_start_tick = state["aux_data"][7]
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Continuous action vector of shape (18,)
        
        Returns:
            (observation, reward, done, info)
        """
        prev_state = self._get_state()
        
        # Convert action to button presses
        buttons = self._action_to_buttons(action)
        
        # Execute action with 4-frame skip
        self.game.make_action(buttons, self.frame_skip)
        
        # Get new state
        curr_state = self._get_state()
        done = self.game.is_episode_finished()
        
        # Calculate reward
        reward = self._calculate_reward(prev_state, curr_state, action, done)
        
        # Info dict
        info = {
            "step": self.step_count,
            "distance_to_exit": curr_state["aux_data"][6],
            "health": curr_state["aux_data"][5],
            "velocity": np.sqrt(curr_state["aux_data"][2]**2 + curr_state["aux_data"][3]**2)
        }
        
        self.step_count += 1
        
        return curr_state, reward, done, info
    
    def close(self):
        """Close the game."""
        self.game.close()


# ============================================================================
# EVALUATION CALLBACK: Track "Time to Exit"
# ============================================================================

class TASSBEvaluationCallback(BaseCallback):
    """
    Callback tracking best model by "Time to Exit" metric.
    
    Saves model checkpoint if the agent reaches exit faster than previous best.
    """
    
    def __init__(self, eval_env: DoomTASEnv, n_eval_episodes: int = 5,
                 save_path: str = "best_tas_model"):
        super().__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_time = float('inf')
        self.eval_count = 0
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Evaluate every 10000 steps
        if self.num_timesteps % 10000 == 0 and self.num_timesteps > 0:
            self.eval_count += 1
            total_time = 0
            success_count = 0
            
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_steps = 0
                
                while not done and episode_steps < 5000:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_steps += 1
                
                if not self.eval_env.game.is_player_dead():
                    success_count += 1
                    total_time += episode_steps
            
            if success_count > 0:
                avg_time = total_time / success_count
                success_rate = success_count / self.n_eval_episodes
                
                print(f"\n[Eval #{self.eval_count}] Timesteps: {self.num_timesteps}")
                print(f"  Success Rate: {success_rate*100:.1f}%")
                print(f"  Avg Time to Exit: {avg_time:.0f} steps")
                print(f"  Best Time: {self.best_time:.0f} steps")
                
                if avg_time < self.best_time:
                    self.best_time = avg_time
                    self.model.save(self.save_path)
                    print(f"  ✓ New best model saved!")
        
        return True


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def train_tas_agent():
    """Main training loop using Stable-Baselines3 PPO."""
    
    config = {
        "wad": "doom2.wad",
        "epochs": 100,
        "steps_per_epoch": 2500,
        "frame_repeat": 4
    }
    
    print("Initializing TAS environment...")
    env = DoomTASEnv(config, render=True)
    
    print("Creating PPO agent with custom feature extractor...")
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
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
    )
    
    print("Starting training loop (TAS speedrun mode)...")
    
    # Create evaluation callback
    eval_callback = TASSBEvaluationCallback(
        eval_env=env,
        n_eval_episodes=3,
        save_path="best_tas_speedrun_model"
    )
    
    # Train for large number of timesteps
    total_timesteps = 2_000_000  # 2 million steps
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
    finally:
        print("\nTraining complete. Closing environment...")
        env.close()
        model.save("final_tas_model")
        print("Final model saved as 'final_tas_model.zip'")


if __name__ == "__main__":
    train_tas_agent()
