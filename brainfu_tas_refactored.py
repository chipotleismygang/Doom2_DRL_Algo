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

# Import essential libraries for the reinforcement learning pipeline
import keyboard  # For real-time keyboard input detection (manual training override)
import os  # For file path operations and checking if model files exist
import numpy as np  # For numerical computations and array handling
import torch  # PyTorch: deep learning framework for neural networks
import torch.nn as nn  # Neural network modules (Conv2d, Linear, etc.)
from typing import Dict, Tuple, Any  # Type hints for better code documentation
import vizdoom as vzd  # VizDoom: Doom game environment API for RL training
from gymnasium import Env, spaces, Wrapper  # OpenAI Gym-compatible environment classes
from stable_baselines3 import PPO  # Proximal Policy Optimization algorithm implementation
from stable_baselines3.common.callbacks import BaseCallback  # Base class for training callbacks
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # Base class for custom feature extraction


# ============================================================================
# CUSTOM FEATURE EXTRACTOR: Multi-Input Processing (CNN + Auxiliary)
# ============================================================================
# WHY: We need to extract meaningful features from both visual data (game screenshots)
# and numerical game state data (position, velocity, etc.). This is done via two
# parallel neural networks that process different input types, then fused together.

class TASFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor combining visual (CNN) and auxiliary input processing.
    
    This class takes two types of input:
    1. "screen" - A grayscale image of the game (1 channel, 120x160 pixels)
    2. "aux_data" - 8 numerical values about the game state
    
    Both inputs are processed separately, then combined into a single feature vector
    that the PPO agent uses to make decisions.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        """
        Initialize the feature extractor.
        
        Args:
            observation_space: Dict with "screen" (image) and "aux_data" (numbers)
            features_dim: Size of the final output feature vector (256 = 256 numbers)
        """
        super().__init__(observation_space, features_dim)
        
        # Extract the number of input channels from the observation space
        # For grayscale: 1 channel. For RGB: 3 channels. Here it's 1 (grayscale).
        self.n_input_channels = observation_space["screen"].shape[0]  # 1
        
        # Extract the number of auxiliary features (position, velocity, angle, health, etc.)
        # We defined 8 features in the environment, so this will be 8.
        self.n_aux_features = observation_space["aux_data"].shape[0]  # 8
        
        # Helper function to create a standardized convolutional block
        # Each block: Conv2d (applies a filter) -> BatchNorm (normalizes) -> ReLU (activation)
        def conv_block(in_f, out_f):
            """
            Creates a convolutional block: Conv2d -> BatchNorm2d -> ReLU
            
            Args:
                in_f: Number of input channels
                out_f: Number of output channels (filters)
            
            Returns:
                nn.Sequential containing the three layers
            """
            return nn.Sequential(
                # Conv2d: applies a 3x3 filter, stride=2 reduces size by half
                # bias=False: BatchNorm handles bias, so we don't need it here
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1, bias=False),
                # BatchNorm2d: normalizes the output, helps training stability
                nn.BatchNorm2d(out_f),
                # ReLU: activation function (outputs 0 for negative, x for positive)
                nn.ReLU()
            )
        
        # Build the CNN for processing the visual screen input
        # Input dimensions: 1 x 120 x 160 (1 channel, 120 height, 160 width)
        self.conv_net = nn.Sequential(
            # Layer 1: 1 -> 8 channels, 120x160 -> 60x80 (stride=2 halves dimensions)
            conv_block(self.n_input_channels, 8),   # -> 8 x 60 x 80
            # Layer 2: 8 -> 16 channels, 60x80 -> 30x40
            conv_block(8, 16),                      # -> 16 x 30 x 40
            # Layer 3: 16 -> 32 channels, 30x40 -> 15x20
            conv_block(16, 32),                     # -> 32 x 15 x 20
            # Layer 4: 32 -> 64 channels, 15x20 -> 8x10
            conv_block(32, 64),                     # -> 64 x 8 x 10
            # AdaptiveAvgPool2d: reduces any spatial dimensions to 4x4
            # This compresses 8x10 -> 4x4, and we have 64 channels: 64 * 4 * 4 = 1024 values
            nn.AdaptiveAvgPool2d((4, 4)),           # -> 64 x 4 x 4 = 1024 values total
            # Flatten: converts 3D (64, 4, 4) into 1D vector of 1024 values
            nn.Flatten()
        )
        
        # Build the auxiliary network for processing game state numbers (8 values)
        self.aux_net = nn.Sequential(
            # Layer 1: 8 input values -> 64 hidden units
            nn.Linear(self.n_aux_features, 64),  # Takes 8 numbers, outputs 64
            # ReLU activation (non-linearity)
            nn.ReLU(),
            # Layer 2: 64 hidden units -> 64 hidden units (same size, allows more learning)
            nn.Linear(64, 64),
            # ReLU activation
            nn.ReLU()
        )
        
        # Build the fusion network that combines CNN features and auxiliary features
        self.fusion = nn.Sequential(
            # Input: 1024 (from CNN) + 64 (from aux) = 1088 values
            # Output: features_dim (default 256) - this is what PPO uses for decisions
            nn.Linear(1024 + 64, features_dim),
            # ReLU activation for non-linearity
            nn.ReLU()
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process observations through the feature extractor.
        
        Args:
            observations: Dict with "screen" (image) and "aux_data" (numbers)
        
        Returns:
            A vector of size features_dim (256) representing the processed state
        """
        # Extract screen image and normalize to 0-1 range (divide by 255)
        # Images are typically 0-255, normalizing helps neural network training
        screen = observations["screen"].float() / 255.0
        
        # Extract auxiliary data and convert to float
        aux_data = observations["aux_data"].float()
        
        # Process screen through CNN to get visual features (1024-dim vector)
        visual_features = self.conv_net(screen)
        
        # Process aux_data through auxiliary network to get features (64-dim vector)
        aux_features = self.aux_net(aux_data)
        
        # Concatenate both feature vectors along dimension 1
        # Result: (batch_size, 1024 + 64) = (batch_size, 1088)
        combined = torch.cat([visual_features, aux_features], dim=1)
        
        # Pass combined features through fusion network
        # Result: (batch_size, features_dim) = (batch_size, 256)
        return self.fusion(combined)


# ============================================================================
# CUSTOM GYM ENVIRONMENT: Doom with TAS-Level Reward Shaping
# ============================================================================
# WHY: We need to create a custom environment because VizDoom by itself doesn't
# have the reward structure we want. We shape rewards to encourage speedrunning
# exploits like strafe-jumping and rocket jumping, not just killing monsters.

class DoomTASEnv(Env):
    """
    Custom Gym environment wrapping VizDoom with speedrun reward shaping.
    
    This environment:
    1. Runs Doom and gets game state (screenshots, position, velocity, etc.)
    2. Takes AI actions (move forward, turn, attack, use)
    3. Calculates rewards based on progress toward the exit (speedrun metric)
    4. Returns observations for the AI to learn from
    """
    metadata = {"render_modes": []}  # Metadata required by Gym API
    
    def __init__(self, config: Dict[str, Any], render: bool = False):
        """
        Initialize the Doom environment.
        
        Args:
            config: Dict with "wad" (game file), "frame_repeat" (frames per action)
            render: If True, display the game window while training
        """
        super().__init__()
        
        # Create a Doom game instance - this is the actual game we'll run
        self.game = vzd.DoomGame()
        
        # Store configuration for later use
        self.config = config
        
        # Frame skip: how many game frames to simulate per action
        # If frame_repeat=1, each action advances 1 frame
        # If frame_repeat=4, each action advances 4 frames (faster training)
        self.frame_skip = self.config.get("frame_repeat", 1)
        
        # Initialize Doom with the configuration
        self._init_doom(render)
        
        # Define the action space: continuous values from 0 to 1
        # We have 18 possible actions (buttons the agent can press)
        # Action vector: [forward, backward, left, right, turn_left, turn_right, attack, use, + 10 unused]
        self.action_space = spaces.Box(low=0, high=1, shape=(18,), dtype=np.float32)
        
        # Define the observation space: what the agent sees
        # Screen: grayscale image, 1 channel (black & white), 120x160 pixels, values 0-255
        # Aux_data: 8 numerical values (position, velocity, angle, health, etc.)
        # Observation space is a Dict with two types of data
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=(1, 120, 160), dtype=np.uint8),
            "aux_data": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        })
        
        # Initialize state tracking variables
        # prev_aux: stores the game state from the previous frame
        # We use this to calculate changes (e.g., "did velocity increase?")
        self.prev_aux = np.zeros(8, dtype=np.float32)
        
        # step_count: how many steps (actions) have been taken so far
        # Used to calculate the game time for the agent
        self.step_count = 0
        
        # was_stuck: whether the agent was stuck (velocity < 5.0) last frame
        # Used to reward the agent for escaping from stuck situations
        self.was_stuck = False
        
        # interacted_last_frame: whether the agent pressed USE (e.g., to open doors) last frame
        # Used to avoid repeatedly rewarding the same door interaction
        self.interacted_last_frame = False
        
    def _init_doom(self, render: bool):
        """
        Configure and initialize the Doom game.
        
        Args:
            render: If True, show the game window during training
        """
        # Set the path to the Doom WAD file (the game data)
        # WAD = "Where's All the Data" - contains game maps, sprites, etc.
        self.game.set_doom_game_path(os.path.abspath(self.config["wad"]))
        
        # Set screen resolution to 160x120 (small for faster processing)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        
        # Set screen format to grayscale (1 channel) for faster processing
        # Could be RGB (3 channels) but grayscale is simpler and faster
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        
        # Set ticrate to 0: uncapped frame rate for fast training
        # Higher ticrate = slower game, 0 = as fast as possible
        self.game.set_ticrate(0)  # Uncapped fast-processing frame ticks
        
        # Render the HUD (health, ammo, etc.) on screen
        self.game.set_render_hud(True)
        
        # Set whether to show the game window
        self.game.set_window_visible(render)
        
        # Set game mode to PLAYER mode (normal gameplay, not a demo)
        self.game.set_mode(vzd.Mode.PLAYER)
        
        # Specify which game variables to track (available to the agent)
        # These are retrieved after each game frame
        self.game.set_available_game_variables([
            vzd.GameVariable.POSITION_X,    # X position in the map
            vzd.GameVariable.POSITION_Y,    # Y position in the map
            vzd.GameVariable.POSITION_Z,    # Z position (height, for vertical movement)
            vzd.GameVariable.VELOCITY_X,    # X velocity (speed in X direction)
            vzd.GameVariable.VELOCITY_Y,    # Y velocity (speed in Y direction)
            vzd.GameVariable.VELOCITY_Z,    # Z velocity (speed in Z direction, jumping/falling)
            vzd.GameVariable.ANGLE,         # Player's facing direction (0-360 degrees)
            vzd.GameVariable.HEALTH         # Player's health points
        ])
        
        # Add available buttons (actions) the agent can perform each frame
        # Each button can be pressed (1) or not pressed (0)
        self.game.add_available_button(vzd.Button.MOVE_FORWARD)    # Walk forward
        self.game.add_available_button(vzd.Button.MOVE_BACKWARD)   # Walk backward
        self.game.add_available_button(vzd.Button.MOVE_LEFT)       # Strafe left
        self.game.add_available_button(vzd.Button.MOVE_RIGHT)      # Strafe right
        self.game.add_available_button(vzd.Button.TURN_LEFT)       # Turn view left
        self.game.add_available_button(vzd.Button.TURN_RIGHT)      # Turn view right
        self.game.add_available_button(vzd.Button.ATTACK)          # Fire weapon
        self.game.add_available_button(vzd.Button.USE)             # Use/open doors
        
        # Initialize the game
        self.game.init()
        
    def _get_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current game state (screenshot and numerical data).
        
        Returns:
            Dict with "screen" (image) and "aux_data" (8 numbers)
        """
        # Get the current game state from Doom
        state = self.game.get_state()
        
        # If the game hasn't started or state is invalid, return zeros
        if state is None:
            return self._zero_observation()
        
        # Extract the screenshot from game state
        # Add a channel dimension at the front: (120, 160) -> (1, 120, 160)
        # This is required by the CNN (which expects channel, height, width)
        screen = state.screen_buffer[np.newaxis, :, :].astype(np.uint8)
        
        # Extract game variables (the 8 values we set up earlier)
        # Renamed from 'vars' to 'game_vars' to avoid shadowing Python's built-in vars()
        game_vars = state.game_variables
        
        # Extract each game variable with safety checks (in case fewer than 8 are available)
        # These are the core state values the agent uses to make decisions
        pos_x = game_vars[0] if len(game_vars) > 0 else 0.0     # X position
        pos_y = game_vars[1] if len(game_vars) > 1 else 0.0     # Y position
        pos_z = game_vars[2] if len(game_vars) > 2 else 0.0     # Z position (height)
        vel_x = game_vars[3] if len(game_vars) > 3 else 0.0     # X velocity (sideways speed)
        vel_y = game_vars[4] if len(game_vars) > 4 else 0.0     # Y velocity (forward/back speed)
        vel_z = game_vars[5] if len(game_vars) > 5 else 0.0     # Z velocity (vertical speed)
        angle = game_vars[6] if len(game_vars) > 6 else 0.0     # Facing angle (0-360 degrees)
        health = game_vars[7] if len(game_vars) > 7 else 0.0    # Health points
        
        # Calculate the game time based on how many steps have been taken
        # Each step = frame_skip frames, so multiply step_count by frame_skip
        game_tick = float(self.step_count * self.frame_skip)
        
        # Calculate distance to the exit (assumed to be at origin 0,0)
        # Distance = sqrt(x^2 + y^2) - 1000 (1000 is offset for map scale)
        # We use max() to ensure it doesn't go negative
        distance_to_exit = max(0.0, float(np.sqrt(pos_x**2 + pos_y**2) - 1000.0))
        
        # Assemble all auxiliary data into a single array
        # Order: pos_x, pos_y, vel_x, vel_y, angle, health, distance_to_exit, game_tick
        aux_data = np.array([
            pos_x, pos_y, vel_x, vel_y, angle, health, distance_to_exit, game_tick
        ], dtype=np.float32)
        
        # Return both the screen and auxiliary data as a dict
        return {"screen": screen, "aux_data": aux_data}
        
    def _zero_observation(self) -> Dict[str, np.ndarray]:
        """
        Return a zero observation (used when game state is invalid).
        
        Returns:
            Dict with zero-filled screen and aux_data
        """
        return {
            "screen": np.zeros((1, 120, 160), dtype=np.uint8),  # Black screen
            "aux_data": np.zeros(8, dtype=np.float32)  # All zeros for game state
        }
        
    def _action_to_buttons(self, action: np.ndarray) -> list:
        """
        Convert continuous action values (0-1) into discrete button presses (0 or 1).
        
        The neural network outputs continuous values (0 to 1) representing how much
        to press each button. We convert these to binary (press or don't press).
        
        Args:
            action: Array of 18 continuous values (0-1)
        
        Returns:
            List of 8 binary values (0 or 1) for each button
        """
        # Take only the first 8 action values (the rest are unused)
        raw_actions = action[:8]
        
        # Start with an empty list to store processed button presses
        processed_buttons = []
        
        # Convert continuous action values to binary (0 or 1)
        # Different buttons have different thresholds based on their sensitivity
        for i, a in enumerate(raw_actions):
            if i in [4, 5]:  # TURN_LEFT (index 4), TURN_RIGHT (index 5)
                # Turns are more sensitive, use lower threshold (0.10)
                processed_buttons.append(1 if a > 0.10 else 0)
            else:
                # All other buttons use threshold of 0.5 (must be > 50% to activate)
                processed_buttons.append(1 if a > 0.5 else 0)
        
        # NOTE: This line appears to be a bug (uses == instead of =)
        # It checks if attack is pressed but does nothing with the result
        if processed_buttons[6] == 1:
            processed_buttons[7] == 1  # Should be assignment (=) not comparison (==)
            
        # Prevent the agent from turning left AND right simultaneously
        # This cancels conflicting inputs from neural network noise
        if processed_buttons[4] == 1 and processed_buttons[5] == 1:
            # Keep the direction with the higher confidence
            if raw_actions[4] > raw_actions[5]:
                processed_buttons[5] = 0  # Cancel right turn
            else:
                processed_buttons[4] = 0  # Cancel left turn

        # --- PHYSICS LATCHING GATES ---
        # These rules encode speedrunning knowledge about when to use special actions
        # They only apply when the agent is stuck (low velocity)
        
        # Get velocity from the previous frame
        vel_x = self.prev_aux[2]  # X velocity from last frame
        vel_y = self.prev_aux[3]  # Y velocity from last frame
        
        # Calculate total speed: sqrt(vel_x^2 + vel_y^2)
        # This is the magnitude of the velocity vector
        scalar_velocity = np.sqrt(vel_x**2 + vel_y**2)

        # PATCHED: Raised threshold to 5.0 to catch door sliding
        # If the agent is stuck (moving slower than 5.0 units/frame):
        if scalar_velocity < 5.0:
            # If agent wants to press USE (open doors, activate switches):
            if raw_actions[7] > 0.15:
                processed_buttons[7] = 1  # Enable USE button
                processed_buttons[6] = 0  # Disable ATTACK to prevent unintended shooting
            # Otherwise, if agent wants to press ATTACK (and USE wasn't pressed):
            elif raw_actions[6] > 0.15:
                processed_buttons[6] = 1  # Enable ATTACK button
                
        return processed_buttons
        
    def _calculate_reward(self, curr_state: Dict, processed_buttons: list, done: bool) -> float:
        """
        Calculate the reward for this step.
        
        The reward shapes the agent's behavior. Positive rewards encourage behavior,
        negative rewards (penalties) discourage it. The agent learns to maximize
        total reward, which in this case means reaching the exit quickly.
        
        Args:
            curr_state: Current game state (screen and aux data)
            processed_buttons: The buttons the agent pressed this frame
            done: Whether the episode is finished
        
        Returns:
            A float reward value (clipped to -15 to 50)
        """
        # Reward for reaching the exit
        # If episode is done AND the agent didn't die, give a huge reward
        if done and not self.game.is_player_dead():
            return 1000.0  # SUCCESS! Reached the exit alive
            
        # Penalty for dying
        # If episode is done AND the agent died, give no reward
        if done:
            return 0.0  # FAILURE! Agent is dead
            
        # Initialize reward accumulator (we add/subtract from this)
        reward = 0.0
        
        # Extract current state data
        curr_aux = curr_state["aux_data"]
        
        # Unpack current state values
        # pos_x, pos_y: current position
        # vel_x, vel_y: current velocity
        # health: current health
        # dist_to_exit: distance to goal
        pos_x, pos_y, vel_x, vel_y, _, health, dist_to_exit, _ = curr_aux
        
        # Unpack previous state values (from last frame)
        prev_pos_x, prev_pos_y, prev_vel_x, prev_vel_y, _, prev_health, prev_dist, _ = self.prev_aux
        
        # === REWARD 1: Delta Distance - Moving toward the exit ===
        # This is the MOST IMPORTANT reward for speedrunning
        
        # Calculate how much the distance to exit changed
        # Negative = got closer (good), Positive = got further (bad)
        distance_change = dist_to_exit - prev_dist
        
        # Calculate current speed (magnitude of velocity)
        scalar_velocity = np.sqrt(vel_x**2 + vel_y**2)
        
        # If distance_change > 0: agent moved AWAY from exit (wrong direction)
        if distance_change > 0:
            # Large penalty: 6.0 points per unit of distance moved away
            # This strongly discourages backtracking
            reward -= 6.0 * distance_change
        else:
            # Agent moved TOWARD exit (right direction)
            # distance_change is negative, so subtracting it gives positive reward
            # Smaller penalty magnitude (2.5 instead of 6.0) for forward progress
            reward -= 2.5 * distance_change
            
        # === REWARD 2: Velocity bonus (CONDITIONAL) ===
        # Only reward going fast if you're making progress
        # This prevents the agent from spinning in circles at high speed
        
        # Only give velocity bonus if agent is moving toward the exit
        if distance_change < -0.5:  # Making progress
            # Reward for speed: 0.1 * min(speed/100, 1.0)
            # Speed of 100 = max reward, speed of 50 = 0.05 reward
            reward += 0.1 * min(scalar_velocity / 100.0, 1.0)
            
        # === REWARD 3: Wall Running ===
        # Wall running is a speedrunning technique: running alongside walls
        # The agent slides along walls, gaining extra speed
        
        # Calculate how far the agent actually moved
        # This is different from velocity - it's the actual position change
        position_delta = np.sqrt((pos_x - prev_pos_x)**2 + (pos_y - prev_pos_y)**2)
        
        # Detect wall running: high speed but low position change
        # (sliding on a wall without moving much forward)
        if scalar_velocity > 30.0 and position_delta < scalar_velocity * 0.3:
            # Only reward wall running if it's helping progress toward exit
            if distance_change < -0.5:
                reward += 0.3  # Small bonus for wall grinding
            
        # === REWARD 4: Thing Running ===
        # "Thing running" = using enemy hitboxes to gain speed
        # Detected by sudden velocity spikes
        
        # Get velocity from previous frame to calculate velocity change
        prev_scalar_velocity = np.sqrt(prev_vel_x**2 + prev_vel_y**2)
        
        # Calculate sudden increase in speed
        velocity_spike = scalar_velocity - prev_scalar_velocity
        
        # If velocity suddenly increased a lot (> 50 units):
        if velocity_spike > 50.0:
            reward += 2.0  # Reward for thing running exploit
            
        # === REWARD 5: Rocket Jumping ===
        # Rocket jumping = firing a rocket at your feet to gain height
        # Detected by health loss + upward velocity
        
        # Calculate health lost this frame
        health_loss = prev_health - health
        
        # If agent took damage AND is moving upward fast:
        if health_loss > 0.0 and vel_z > 20.0:  # vel_z is vertical velocity
            reward += 5.0  # Reward for rocket jumping
            
        # === REWARD 6: Time Penalty ===
        # Small penalty for each step to encourage finishing quickly
        # This incentivizes the agent to reach the exit FAST, not just eventually
        reward -= 0.05
        
        # === ADDITIONAL NUDGES (Training guidance) ===
        
        # NUDGE A: Loitering Tax
        # Penalize the agent for standing still without good reason
        # This prevents it from getting stuck or idling
        if scalar_velocity < 5.0 and processed_buttons[7] == 0 and processed_buttons[6] == 0:
            # Standing still, not trying to open doors or shoot
            # This is bad - penalize it to encourage movement
            reward -= 3.0
            
        # NUDGE B: Escape Bonus
        # Reward the agent for escaping from being stuck
        if self.was_stuck and scalar_velocity > 5.0:
            # Was stuck last frame, now moving fast - reward for escape!
            reward += 25.0
            
        # NUDGE C: Door Interaction
        # Give bonus for FIRST interaction with a door (but not for holding)
        if scalar_velocity < 5.0 and processed_buttons[7] == 1:
            # Agent is using door
            if not self.interacted_last_frame:
                # First time pressing USE this step - new interaction
                reward += 20.0  
            else:
                # Already holding USE - no additional reward (avoid spamming)
                reward += 0.0

        # Update tracking variables for next frame
        self.interacted_last_frame = (processed_buttons[7] == 1)  # Was USE pressed?
        self.was_stuck = (scalar_velocity < 5.0)  # Is agent stuck now?
        
        # === HAND OF GOD (Manual training override) ===
        # The user can press 'g' or 'b' during training to manually reward/penalize
        # This lets humans teach the AI without changing code
        try:
            if keyboard.is_pressed('g'):
                reward += 10.0  # Good action!
                print("Good boy :3 +10")
            elif keyboard.is_pressed('b'):
                reward -= 10.0  # Bad action!
                print("BAD boy >:( -10")
        except Exception:
            # In case keyboard input fails, just skip it
            pass
        
        # Clip reward to range [-15, 50] to prevent extreme values
        # Extreme rewards can destabilize training
        return float(np.clip(reward, -15.0, 50.0))
        
    def reset(self, seed=None, options=None) -> Tuple[Dict, dict]:
        """
        Reset the environment for a new episode.
        
        Called at the start of each training episode to initialize a fresh game.
        
        Returns:
            (observation, info): Initial state and empty info dict
        """
        # Call parent class reset for proper Gym API compliance
        super().reset(seed=seed)
        
        # Start a new episode in Doom
        self.game.new_episode()
        
        # --- GOD MODE ENABLED ---
        # Make the agent invincible so it only needs to focus on reaching the exit
        # This removes health management from the training objective
        self.game.send_game_command("god")
        
        # Reset tracking variables to initial state
        self.step_count = 0  # No steps taken yet
        
        self.was_stuck = False  # Agent isn't stuck at the start
        self.interacted_last_frame = False  # No interactions yet
        
        # Get initial game state
        state = self._get_state()
        
        # Store initial auxiliary data for next frame's reward calculation
        self.prev_aux = state["aux_data"].copy()
        
        # Return initial observation and empty info dict
        return state, {}
        
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, dict]:
        """
        Execute one step of the environment.
        
        This is the main loop: agent gives action -> game updates -> return new state and reward
        
        Args:
            action: Array of 18 continuous values (0-1)
        
        Returns:
            (observation, reward, done, truncated, info):
            - observation: Dict with "screen" and "aux_data"
            - reward: Float reward for this step
            - done: Boolean, True if episode finished
            - truncated: Boolean, True if episode was cut off (not applicable here)
            - info: Dict with useful debug info
        """
        # Convert continuous action values to discrete button presses
        buttons = self._action_to_buttons(action)
        
        # Execute the action in Doom for frame_skip frames
        # This advances the game by frame_skip frames
        self.game.make_action(buttons, self.frame_skip)
        
        # Check if the episode is finished
        # Episode ends if: map is finished (reached exit) OR player died
        done = self.game.is_episode_finished() or self.game.is_player_dead()
        
        # Get the new game state after action was taken
        curr_state = self._get_state()
            
        # Calculate reward for this step
        reward = self._calculate_reward(curr_state, buttons, done)
        
        # Store current state for next frame's calculations
        self.prev_aux = curr_state["aux_data"].copy()
        
        # Increment step counter
        self.step_count += 1
        
        # Prepare debug information for the user/trainer
        info = {
            "distance_to_exit": curr_state["aux_data"][6],  # How far to goal?
            "health": curr_state["aux_data"][5],  # Current health
            "velocity": np.sqrt(curr_state["aux_data"][2]**2 + curr_state["aux_data"][3]**2),  # Speed
            "is_dead": self.game.is_player_dead()  # Did agent die?
        }
        
        # Return all required Gym API values
        return curr_state, reward, done, False, info
        
    def close(self):
        """
        Clean up and close the Doom game instance.
        
        Called when training is finished or interrupted.
        """
        self.game.close()

# ============================================================================
# INPUT OVERRIDE: Teacher Forcing Wrapper
# ============================================================================
# WHY: During training, a human can take over with keyboard inputs to demonstrate
# good gameplay. This wrapper catches keyboard input and uses it to override
# the AI's decisions while still giving the AI credit for the reward.

class TeacherForcingWrapper(Wrapper):
    """
    Wrapper that allows human keyboard input to override AI actions during training.
    
    This is called "teacher forcing" - the human teacher demonstrates good behavior,
    and the AI learns from it. When human input overrides the AI, we add a small
    reward bonus to encourage the AI to learn from the human's actions.
    """
    def __init__(self, env, teacher_reward: float = 0.25):
        """
        Initialize the wrapper.
        
        Args:
            env: The underlying environment (DoomTASEnv)
            teacher_reward: Bonus reward when human overrides AI
        """
        super().__init__(env)
        
        # Bonus reward when human takes over (encourages learning from human)
        self.teacher_reward = teacher_reward
        
        # Map keyboard keys to action vectors
        # Each action is an 18-element vector of 0s and 1s
        # The key pressed determines which button(s) get pressed
        self.key_to_action = {
            'w':	np.array([1,0,0,0,0,0,0,0] + [0]*10, dtype=np.float32), # forward
            's':	np.array([0,1,0,0,0,0,0,0] + [0]*10, dtype=np.float32), # backward
            'a':	np.array([0,0,1,0,0,0,0,0] + [0]*10, dtype=np.float32), # leftward (strafe)
            'd':	np.array([0,0,0,1,0,0,0,0] + [0]*10, dtype=np.float32), # rightward (strafe)
            'left':	np.array([0,0,0,0,1,0,0,0] + [0]*10, dtype=np.float32), # turn left
            'right':np.array([0,0,0,0,0,1,0,0] + [0]*10, dtype=np.float32), # turn right
            'space':np.array([0,0,0,0,0,0,1,0] + [0]*10, dtype=np.float32), # attack (shoot)
            'e':	np.array([0,0,0,0,0,0,0,1] + [0]*10, dtype=np.float32), # use (open doors)
        }
    
    def step(self, action):
        """
        Execute a step, checking for human keyboard override first.
        
        Args:
            action: AI's desired action
        
        Returns:
            Standard Gym step return: (obs, reward, done, truncated, info)
        """
        # Flag for whether human overrode the AI
        teacher_override = False
        
        # Check each keyboard key to see if human is providing input
        for key, teacher_action in self.key_to_action.items():
            if keyboard.is_pressed(key):
                # Human pressed this key - override AI's action
                action = teacher_action
                teacher_override = True
                break  # Only one key can be pressed at a time
        
        # Execute the action in the underlying environment
        obs, reward, done, truncated, info = self.env.step(action)
        
        # If human overrode, give bonus reward to encourage learning
        # The AI gets the normal reward PLUS teacher_reward bonus
        if teacher_override:
            reward += self.teacher_reward
        
        # Return all Gym API values
        return obs, reward, done, truncated, info

# ============================================================================
# EVALUATION CALLBACK: Track "Time to Exit"
# ============================================================================
# WHY: During training, we need to periodically test the agent on new episodes
# to see how well it's learning. This callback runs every 10,000 training steps.

class TASSBEvaluationCallback(BaseCallback):
    """
    Callback that evaluates the agent periodically during training.
    
    Every 10,000 training steps, this callback:
    1. Runs 3 evaluation episodes (new games)
    2. Measures how fast the agent completes them
    3. Saves the model if it's the fastest yet
    
    This tracks progress toward the speedrunning goal (reaching the exit quickly).
    """
    def __init__(self, eval_env: DoomTASEnv, n_eval_episodes: int = 3, save_path: str = "best_tas_model"):
        """
        Initialize the evaluation callback.
        
        Args:
            eval_env: A separate Doom environment for evaluation (doesn't affect training)
            n_eval_episodes: How many evaluation games to run each time
            save_path: Where to save the best model
        """
        super().__init__()
        
        # Separate environment for evaluation (doesn't mix with training data)
        self.eval_env = eval_env
        
        # Number of test episodes to run each evaluation
        self.n_eval_episodes = n_eval_episodes
        
        # File path where best model is saved
        self.save_path = save_path
        
        # Track the best (fastest) time to exit seen so far
        self.best_time = float('inf')  # Start at infinity so any time is better
        
        # Counter for how many evaluations have been run
        self.eval_count = 0
        
    def _on_step(self) -> bool:
        """
        Called after each training step. Runs evaluation every 10,000 steps.
        
        Returns:
            True to continue training
        """
        # Check if it's time to evaluate (every 10,000 steps, after step 0)
        if self.num_timesteps % 10000 == 0 and self.num_timesteps > 0:
            # Increment evaluation counter
            self.eval_count += 1
            
            # Accumulators for averaging across episodes
            total_time = 0  # Sum of steps taken across successful episodes
            success_count = 0  # Number of successful (alive) completions
            
            # Run multiple evaluation episodes
            for _ in range(self.n_eval_episodes):
                # Start fresh episode in evaluation environment
                obs, _ = self.eval_env.reset()
                done = False
                episode_steps = 0  # Count steps in this episode
                
                # Play until episode ends or step limit (1000 steps = max 1000 frames)
                while not done and episode_steps < 1000:
                    # Get AI's action (deterministic=False for some exploration)
                    action, _ = self.model.predict(obs, deterministic=False)
                    
                    # Execute action in evaluation environment
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    
                    # Increment step counter for this episode
                    episode_steps += 1
                
                # Only count this episode if the agent survived (didn't die)
                if not self.eval_env.game.is_player_dead() and done:
                    success_count += 1  # One more successful completion
                    total_time += episode_steps  # Add steps to total
            
            # Calculate statistics from the evaluation episodes
            if success_count > 0:
                # Average steps per successful episode
                avg_time = total_time / success_count
                
                # Percentage of episodes successfully completed
                success_rate = success_count / self.n_eval_episodes
                
                # Print evaluation results
                print(f"\n[Eval #{self.eval_count}] Timesteps: {self.num_timesteps}")
                print(f"   Success Rate: {success_rate*100:.1f}%")
                print(f"   Avg Time to Exit: {avg_time:.0f} steps")
                
                # If this is the fastest time yet, save the model
                if avg_time < self.best_time:
                    # Update best time record
                    self.best_time = avg_time
                    
                    # Save the model (weights and architecture)
                    self.model.save(self.save_path)
                    
                    # Notify the user
                    print(f"   ✓ New best model saved via time optimization target!")
        
        # Always return True to continue training
        return True


# ============================================================================
# MAIN TRAINING SCRIPT WITH CONDITIONAL LOAD
# ============================================================================

def train_tas_agent():
    """
    Main training function.
    
    This function:
    1. Creates a Doom environment with speedrun rewards
    2. Loads pre-trained weights if available, else starts from scratch
    3. Trains a PPO agent for 2 million steps
    4. Periodically evaluates and saves the best model
    """
    
    # Configuration dictionary with game parameters
    config = {
        "wad": "doom2.wad",          # Path to Doom 2 game data
        "epochs": 10,                 # Training epochs (not used in current code)
        "steps_per_epoch": 2000,      # Steps per epoch (not used in current code)
        "frame_repeat": 1             # Frames to advance per action
    }
    
    # Path to pre-trained model file (if it exists, we'll load it)
    PRETRAINED_ZIP = "pre_trained_tas_model.zip"
    
    # Notify user of initialization
    print("Initializing TAS environment...")
    
    # Create training environment with rendering enabled
    env = DoomTASEnv(config, render=True)
    
    # Wrap with teacher forcing for human keyboard input
    env = TeacherForcingWrapper(env)
    
    # Create a separate evaluation environment (render=False for speed)
    # This is separate from training to avoid interfering with PPO's data collection
    eval_env = DoomTASEnv(config, render=False) 
    
    # Check if pre-trained weights exist
    if os.path.exists(PRETRAINED_ZIP):
        # Load pre-trained model
        print(f"🧠 Found pre-trained weights file '{PRETRAINED_ZIP}'! Loading model...")
        
        # Load the saved model onto GPU or CPU
        model = PPO.load(
            PRETRAINED_ZIP,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Reconnect the model to the new environment
        model.set_env(env)
        
        # Use lower learning rate for fine-tuning pre-trained model
        # Pre-trained models only need small updates, not large changes
        model.learning_rate = 5e-5
        
        # Optionally reduce exploration by lowering log_std (policy entropy)
        if hasattr(model.policy, "log_std"):
            with torch.no_grad():
                # log_std controls randomness in actions
                # More negative = more deterministic (less exploration)
                model.policy.log_std.fill_(-3.0)
    else:
        # No pre-trained model found - create a fresh agent from scratch
        print("⚠️ Pre-trained model zip not found! Creating fresh PPO agent from scratch...")
        
        # Create PPO agent with custom architecture
        model = PPO(
            # Use custom policy that handles multiple input types (image + numbers)
            policy='MultiInputPolicy',
            
            # The environment for training
            env=env,
            
            # Learning rate: how much to update weights per step
            # Higher = faster learning but less stable
            learning_rate=3e-4,
            
            # Number of steps to collect before updating (larger = more stable)
            n_steps=2048,
            
            # Batch size for gradient updates (smaller = faster but noisier)
            batch_size=64,
            
            # Number of times to update on the same batch (more = longer training)
            n_epochs=10,
            
            # Discount factor: how much to value future rewards
            # 0.99 = high value for future, 0.95 = some uncertainty
            gamma=0.99,
            
            # GAE lambda: smoothness of advantage calculation
            # Higher = smoother but biased, lower = higher variance
            gae_lambda=0.95,
            
            # PPO clipping range: how much policy can change per update
            # 0.2 = max 20% change, prevents destabilizing updates
            clip_range=0.2,
            
            # Entropy coefficient: encourages exploration
            # Higher = more random exploration, lower = more greedy
            ent_coef=0.01,
            
            # Use GPU if available for faster training
            device='cuda' if torch.cuda.is_available() else 'cpu',
            
            # Verbose=1: print training progress
            verbose=1,
            
            # Custom policy architecture
            policy_kwargs=dict(
                # Use our custom feature extractor (CNN + auxiliary)
                features_extractor_class=TASFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                
                # Network architecture:
                # pi (policy) network: 256 -> 256 -> action space
                # vf (value) network: 256 -> 256 -> reward prediction
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            )
        )
    
    # Create evaluation callback that saves best models
    eval_callback = TASSBEvaluationCallback(
        eval_env=eval_env,              # Separate environment for evaluation
        n_eval_episodes=3,              # Run 3 test games each evaluation
        save_path="best_tas_speedrun_model"  # Where to save best model
    )
    
    try:
        # Train the model for 2 million timesteps
        # This is equivalent to 2 million actions taken in the environment
        model.learn(
            total_timesteps=2_000_000,
            callback=eval_callback,
            progress_bar=True
        )
    finally:
        # Always run this cleanup, even if training is interrupted
        print("\nClosing environment setup...")
        
        # Close training environment
        env.close()
        
        # Close evaluation environment
        eval_env.close()
        
        # Save the final model (even if it's not the best)
        model.save("final_tas_model")
        
        # Notify user of completion
        print("Saved processing execution graph maps completely.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # This code only runs if this file is executed directly (not imported)
    # It starts the training process
    train_tas_agent()
