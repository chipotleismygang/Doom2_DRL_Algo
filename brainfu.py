"""
Deep Reinforcement Learning Algorithm for Doom using Dueling Double DQN.

This module implements a Dueling Dueling Double Q-Network agent that learns to play
the Doom game through pixel-based visual input. The agent uses a dueling architecture
to separately estimate state value and action advantages, improving stability and performance.

Architecture:
- Convolutional layers: Extract spatial features from 160x120 grayscale frames
- Dueling streams: Split into value stream (state value) and advantage stream (action advantages)
- Experience replay: Store and sample past transitions to decorrelate training data
- Epsilon-greedy exploration: Balance exploration vs exploitation during learning
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import trange
import vizdoom as vzd


class DuelQNet(nn.Module):
    """
    Dueling Deep Q-Network architecture.
    
    The dueling architecture splits the network into two streams:
    1. State value stream: Estimates V(s) - how good the current state is
    2. Advantage stream: Estimates A(s,a) - how much better each action is vs the average
    
    Final Q-value: Q(s,a) = V(s) + [A(s,a) - mean(A(s,*))]
    This separation helps the network learn state evaluations more efficiently.
    
    Args:
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        num_actions (int): Number of possible actions the agent can take
    """
    
    def __init__(self, in_channels, num_actions):
        super(DuelQNet, self).__init__()
        
        # Helper function to create a reusable convolutional block
        # Each block applies: Conv2d -> BatchNorm -> ReLU activation
        def conv_block(in_f, out_f):
            """
            Creates a convolutional block with batch normalization and ReLU activation.
            
            Args:
                in_f (int): Number of input feature channels
                out_f (int): Number of output feature channels
                
            Returns:
                nn.Sequential: A sequential block containing Conv2d, BatchNorm2d, and ReLU
            """
            return nn.Sequential(
                # Conv2d: 3x3 kernel, stride=2 (reduces spatial dimensions by half)
                # padding=1 ensures output size = (input_size - 1) / stride + 1
                # bias=False because BatchNorm will handle the bias
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1, bias=False),
                # BatchNorm2d: Normalizes activations across the batch dimension
                # Reduces internal covariate shift and allows higher learning rates
                nn.BatchNorm2d(out_f),
                # ReLU: Element-wise max(0, x) activation for non-linearity
                nn.ReLU()
            )

        # Build convolutional feature extraction layers
        # Input: 1x160x120 (channels x height x width)
        self.conv1 = conv_block(1, 8)    # Output: 8x80x60 (160/2 x 120/2)
        self.conv2 = conv_block(8, 8)    # Output: 8x40x30 (80/2 x 60/2)
        self.conv3 = conv_block(8, 8)    # Output: 8x20x15 (40/2 x 30/2)
        self.conv4 = conv_block(8, 16)   # Output: 16x10x7 (20/2 x 15/2)
        
        # Adaptive average pooling: Downsample to fixed 4x4 spatial dimensions
        # Ensures consistent flattened size (16 * 4 * 4 = 256) regardless of input variations
        # This makes the network more robust to slight input size variations
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Value stream: Predicts single scalar representing state value V(s)
        # Takes 256 flattened features and reduces through fully connected layers
        self.state_fc = nn.Sequential(
            nn.Linear(256, 128),   # Compress features from 256 to 128 dimensions
            nn.ReLU(),             # Add non-linearity
            nn.Linear(128, 1)      # Output single value: V(s)
        )
        
        # Advantage stream: Predicts advantage for each of 8 possible actions
        # A(s, a) tells us how much better each action is compared to average action
        self.advantage_fc = nn.Sequential(
            nn.Linear(256, 128),   # Compress features from 256 to 128 dimensions
            nn.ReLU(),             # Add non-linearity
            nn.Linear(128, 8)      # Output 8 advantages (one per action): A(s, *)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 160, 120) with pixel values 0-255
            
        Returns:
            torch.Tensor: Q-values of shape (batch_size, 8) for each action
        """
        # Normalize pixel values from [0, 255] to [0, 1] for better numerical stability
        x = x.float() / 255.0
        
        # Pass through convolutional layers to extract spatial features
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Pool to fixed 4x4 spatial dimensions
        x = self.adaptive_pool(x)
        
        # Flatten from (batch, 16, 4, 4) to (batch, 256)
        x = x.reshape(x.size(0), -1)
        
        # Compute state value V(s) and advantages A(s, a)
        state_value = self.state_fc(x)  # Shape: (batch_size, 1)
        advantage = self.advantage_fc(x)  # Shape: (batch_size, 8)
        
        # Combine value and advantages: Q(s,a) = V(s) + [A(s,a) - mean(A(s,*))]
        # Subtracting mean advantage: Centers advantages around zero (optional but helps stability)
        # keepdim=True maintains proper broadcasting: (batch, 1) - (batch, 1)
        return state_value + (advantage - advantage.mean(dim=1, keepdim=True))


# Register DuelQNet for safe deserialization with weights_only=False
# This prevents pickle injection attacks when loading model checkpoints
torch.serialization.add_safe_globals([DuelQNet])


class Agent:
    """
    Deep Q-Learning agent that interacts with the Doom environment.
    
    The agent:
    - Uses epsilon-greedy exploration: Random actions with probability epsilon
    - Stores experiences in a replay buffer for training stability
    - Updates two networks: policy_net (learns) and target_net (stable targets)
    - Implements Double DQN: Uses policy_net to select actions, target_net to evaluate them
    
    Hyperparameters:
    - Learning rate (1e-4): Controls step size for gradient updates
    - Gamma (0.99): Discount factor for future rewards
    - Epsilon (1.0 → 0.01): Exploration rate decayed over time
    - Memory size (20000): Replay buffer capacity
    - Batch size (64): Number of experiences sampled per update
    """
    
    def __init__(self, n_actions, model_path="model-doom.pth"):
        """
        Initialize the RL agent with networks and hyperparameters.
        
        Args:
            n_actions (int): Number of actions available to the agent
            model_path (str): File path to save/load trained model weights
        """
        self.n_actions = n_actions
        self.model_path = model_path
        
        # Determine device: Use GPU (cuda) if available, otherwise CPU
        # GPU training is much faster for neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Active Device: {self.device}")
        
        # Create two identical networks for Double DQN stability
        # policy_net: Updated frequently, used for selecting actions
        # target_net: Updated less frequently, used for computing target Q-values
        # Having separate networks reduces correlations in targets, improving stability
        self.policy_net = DuelQNet(1, 8).to(self.device)
        self.target_net = DuelQNet(1, 8).to(self.device)
        
        # Adam optimizer: Adaptive learning rate optimizer with momentum
        # lr=1e-4: Learning rate - step size for weight updates
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        
        # Experience replay buffer: Stores (state, action, reward, next_state, done) tuples
        # maxlen=20000: Keeps only the 20000 most recent experiences (FIFO)
        # Sampling from this buffer decorrelates training data, reducing overfitting
        self.memory = deque(maxlen=20000)
        
        # Batch size: Number of experiences sampled per training update
        # Larger batches → more stable gradients, but more memory needed
        self.batch_size = 64
        
        # Discount factor: Weight of future rewards
        # gamma=0.99: Agent values future rewards almost equally to immediate rewards
        # gamma=0.0: Agent only cares about immediate reward
        self.gamma = 0.99
        
        # Epsilon-greedy exploration parameters
        # epsilon: Probability of taking random action (vs. greedy action)
        self.epsilon = 1.0  # Start with high exploration
        self.eps_decay = 0.9995  # Multiply epsilon by this after each episode
        self.eps_min = 0.01  # Minimum exploration rate (still explores 1% of the time)
        
        # Training progress tracking
        self.train_step = 0  # Total number of policy updates performed
        self.episode_rewards = []  # List of total rewards per episode for monitoring
        
        # Load previously saved model if it exists (resume training)
        self.load_model()
        
        # Initialize target_net with same weights as policy_net
        # Both start identical; target_net is updated less frequently
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def load_model(self):
        """
        Load previously saved model weights and hyperparameters.
        
        Handles two checkpoint formats:
        1. Direct model: Old format where checkpoint is the model state_dict directly
        2. Dict format: New format with 'model_state_dict' key and 'epsilon' key
        
        If no model exists, this is silently skipped (fresh training).
        """
        if os.path.exists(self.model_path):
            print(f"Loading {self.model_path}...")
            # Load checkpoint: weights_only=False allows loading custom classes (DuelQNet)
            # map_location ensures GPU models can load on CPU and vice versa
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Handle old format: checkpoint is directly a state_dict
            if isinstance(checkpoint, DuelQNet):
                self.policy_net.load_state_dict(checkpoint.state_dict())
            # Handle new format: checkpoint is a dictionary with metadata
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                # Also restore epsilon (exploration rate) from last training session
                self.epsilon = checkpoint.get('epsilon', self.epsilon)

    def save_model(self):
        """
        Save current model weights and hyperparameters to disk.
        
        Saves as a dictionary containing:
        - model_state_dict: Network weights and biases
        - epsilon: Current exploration rate (resume training with same exploration level)
        """
        checkpoint = {
            'model_state_dict': self.policy_net.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, self.model_path)

    def select_action(self, state):
        """
        Select an action using epsilon-greedy strategy.
        
        With probability epsilon: Select random action (exploration)
        Otherwise: Select action with highest estimated Q-value (exploitation)
        
        Args:
            state (vizdoom.GameState): Game state containing screen buffer and variables
            
        Returns:
            tuple: (action_vector, action_index, screen)
                - action_vector: List of 8 binary values (one-hot encoding)
                - action_index: Index of selected action (0-7)
                - screen: Processed screen buffer for memory storage
        """
        # Extract screen buffer and add batch dimension: (1, 160, 120)
        screen = state.screen_buffer[np.newaxis, :, :]
        
        # Epsilon-greedy decision
        if random.random() < self.epsilon:
            # Exploration: Pick random action (0 to n_actions-1)
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            # Exploitation: Pick action with highest Q-value
            state_tensor = torch.from_numpy(screen).to(self.device).unsqueeze(0)  # Add batch: (1, 1, 160, 120)
            
            # Evaluate Q-values without computing gradients (faster, save memory)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)  # Shape: (1, 8)
                # Get index of maximum Q-value for first (only) batch element
                action_idx = q_values[0, :self.n_actions].argmax().item()
        
        # Convert action index to one-hot vector (required by vizdoom API)
        action = [0] * self.n_actions
        action[action_idx] = 1  # Set selected action to 1
        
        return action, action_idx, screen

    def update_policy(self):
        """
        Update network weights using Double DQN with experience replay.
        
        Process:
        1. Sample batch of past experiences (state, action, reward, next_state, done)
        2. Compute predicted Q-values with policy_net
        3. Compute target Q-values with target_net and policy_net (Double DQN)
        4. Minimize MSE loss between predicted and target Q-values
        5. Update target_net every 1000 training steps
        
        Double DQN: Use policy_net to select best action, target_net to evaluate it
        This reduces overestimation bias compared to standard DQN.
        """
        # Only train if we have enough experiences collected
        if len(self.memory) < self.batch_size:
            return
        
        # Randomly sample batch_size experiences from replay buffer
        # Random sampling breaks correlations between consecutive experiences
        batch = random.sample(self.memory, self.batch_size)
        
        # Unzip batch into separate lists: (states, actions, rewards, next_states, dones)
        s, a, r, sn, d = zip(*batch)
        
        # Convert to PyTorch tensors on the appropriate device
        # Stack converts list of (160, 120) arrays to tensor of (batch_size, 160, 120)
        s = torch.from_numpy(np.stack(s)).to(self.device)  # Shape: (64, 1, 160, 120)
        a = torch.tensor(a, device=self.device)  # Shape: (64,) - action indices
        r = torch.tensor(r, dtype=torch.float32, device=self.device)  # Shape: (64,)
        sn = torch.from_numpy(np.stack(sn)).to(self.device)  # Shape: (64, 1, 160, 120)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)  # Shape: (64,) - done flags

        # Compute predicted Q-values: Forward pass through policy_net
        # Shape: (64, 8) - Q-value for each of 8 actions
        q_values = self.policy_net(s)
        
        # Use gather to extract Q-values for the actions that were actually taken
        # gather(1, a.unsqueeze(1)): Select column corresponding to each action
        # squeeze(): Remove extra dimension to get shape (64,)
        q_values = q_values.gather(1, a.unsqueeze(1)).squeeze()
        
        # Compute target Q-values using Double DQN
        with torch.no_grad():  # Don't compute gradients for target computation (faster, save memory)
            # Use policy_net to SELECT best actions in next states
            next_actions = self.policy_net(sn).argmax(dim=1)  # Shape: (64,)
            
            # Use target_net to EVALUATE those actions (reduces overestimation)
            next_q = self.target_net(sn).gather(1, next_actions.unsqueeze(1)).squeeze()  # Shape: (64,)
            
            # Bellman equation: Q(s,a) = r + γ * Q(s',a') * (1 - done)
            # (1 - d): Zero out target if episode is done (no future reward)
            expected_q = r + (self.gamma * next_q * (1 - d))

        # Compute loss: Mean Squared Error between predicted and target Q-values
        loss = nn.MSELoss()(q_values, expected_q)
        
        # Backpropagation: Compute gradients of loss w.r.t. policy_net parameters
        self.optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Compute new gradients
        
        # Gradient clipping: Prevent exploding gradients (numerical stability)
        # Clips gradient norm to max 1.0 (rescales if larger)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        # Gradient descent step: Update weights using Adam optimizer
        self.optimizer.step()
        
        # Update training counter
        self.train_step += 1
        
        # Periodically update target_net with policy_net weights
        # Every 1000 updates: Copy policy_net → target_net
        # This decouples target computation from rapidly changing network
        if self.train_step % 1000 == 0 and self.train_step > 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


class Game:
    """
    Wrapper for the Doom game environment.
    
    Handles:
    - Game initialization and configuration
    - Episode management (starting new games, getting observations)
    - Action execution and reward calculation
    - Training loop orchestration
    """
    
    def __init__(self, config):
        """
        Initialize the Doom game with specified configuration.
        
        Args:
            config (dict): Configuration dictionary containing:
                - wad (str): Path to Doom WAD file
                - epochs (int): Number of training episodes
                - steps_per_epoch (int): Steps per episode
                - frame_repeat (int): Frames to repeat each action
        """
        # Create Doom game instance
        self.game = vzd.DoomGame()
        
        # Set WAD file (contains game maps, textures, sprites)
        self.game.set_doom_game_path(os.path.abspath(config["wad"]))
        
        # Set screen resolution: 160x120 pixels (small for fast computation)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        
        # Set grayscale mode: Single channel (more efficient than RGB for vision)
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        
        # Enable HUD: Show health, ammo, etc. for reward calculation
        self.game.set_render_hud(True)
        
        # Show game window (set False for headless training)
        self.game.set_window_visible(True)
        
        # Set to player mode (normal gameplay, not spectator)
        self.game.set_mode(vzd.Mode.PLAYER)
        
        # Set game speed: 2100 tics per second (standard Doom speed)
        self.game.set_ticrate(2100)
        
        # Enable game variables to track (these get passed to agent as rewards)
        self.game.set_available_game_variables([
            vzd.GameVariable.POSITION_X,      # Player X coordinate
            vzd.GameVariable.POSITION_Y,      # Player Y coordinate
            vzd.GameVariable.HEALTH,          # Current health (0-100)
            vzd.GameVariable.ARMOR,           # Armor level
            vzd.GameVariable.SELECTED_WEAPON_AMMO,  # Current weapon ammo
            vzd.GameVariable.ITEMCOUNT,       # Items picked up
            vzd.GameVariable.SECRETCOUNT,     # Secret areas found
            vzd.GameVariable.KILLCOUNT        # Enemies killed
        ])

        # Enable actions the agent can take (one-hot encoding of these 8 actions)
        self.game.add_available_button(vzd.Button.MOVE_FORWARD)    # W
        self.game.add_available_button(vzd.Button.MOVE_BACKWARD)   # S
        self.game.add_available_button(vzd.Button.MOVE_LEFT)       # A
        self.game.add_available_button(vzd.Button.MOVE_RIGHT)      # D
        self.game.add_available_button(vzd.Button.ATTACK)          # Spacebar (shoot)
        self.game.add_available_button(vzd.Button.TURN_LEFT)       # Left arrow
        self.game.add_available_button(vzd.Button.TURN_RIGHT)      # Right arrow
        self.game.add_available_button(vzd.Button.USE)             # Use door/item
        
        self.config = config

    def train(self, agent):
        """
        Main training loop: Run multiple episodes, collect experiences, update agent.
        
        Reward structure:
        - Discovery reward: +0.05 for visiting new tiles
        - Pickup rewards: +0.1 health/armor/items, +2.0 secrets, +0.3 kills
        - Negative rewards: -0.15 health loss, -1.0 death
        - Clipped to [-1, 1] for training stability
        
        Args:
            agent (Agent): The RL agent to train
        """
        # Initialize the game engine
        self.game.init()
        
        # Track which map tiles agent has visited (for exploration reward)
        visited_tiles = set()
        tile_size = 64  # Size of each tile in game units (larger = fewer tiles)
        
        try:
            # Train for specified number of epochs (episodes)
            for epoch in range(self.config["epochs"]):
                # Start a new episode (game instance)
                self.game.new_episode()
                visited_tiles.clear()  # Reset visited areas for new episode
                
                # Initialize previous game variables for reward computation
                # We compare current vs. previous values to give differential rewards
                last_health = 100
                last_armor = 0
                last_ammo = 50
                last_items = 0
                last_secrets = 0
                last_kills = 0
                
                epoch_reward = 0  # Accumulate total reward for this episode
                
                # Run for specified number of steps per epoch
                for _ in trange(self.config["steps_per_epoch"], desc=f"Epoch {epoch+1}"):
                    # Check if episode ended (agent died or max steps reached)
                    if self.game.is_episode_finished():
                        self.game.new_episode()

                    # Get current game state (screen buffer + game variables)
                    state = self.game.get_state()
                    if state is None:  # Handle edge case where state is unavailable
                        continue

                    # Extract game variables from state
                    vars = state.game_variables
                    pos_x = vars[0]  # X position for tile tracking
                    pos_y = vars[1]  # Y position for tile tracking
                    cur_health = vars[2]
                    cur_armor = vars[3]
                    cur_ammo = vars[4]
                    cur_items = vars[5]
                    cur_secrets = vars[6]
                    cur_kills = vars[7]

                    # Calculate reward for picking up items and managing health
                    pickup_reward = 0
                    
                    # Positive rewards for gaining resources
                    if cur_health > last_health:
                        pickup_reward += 0.1  # Health pickup
                    if cur_armor > last_armor:
                        pickup_reward += 0.1  # Armor pickup
                    if cur_ammo > last_ammo:
                        pickup_reward += 0.05  # Ammo pickup
                    if cur_items > last_items:
                        pickup_reward += 0.1  # Item pickup
                    if cur_secrets > last_secrets:
                        pickup_reward += 2.0  # Secret area (high reward)
                    if cur_kills > last_kills:
                        pickup_reward += 0.3  # Enemy killed
                    
                    # Negative rewards for losing resources
                    if cur_health < last_health:
                        pickup_reward -= 0.15  # Taking damage
                    if cur_armor < last_armor:
                        pickup_reward -= 0.1  # Armor degradation
                    if cur_ammo < last_ammo:
                        pickup_reward -= 0.02  # Ammo depletion
                    if self.game.is_player_dead():
                        pickup_reward -= 1.0  # Large penalty for death
                    
                    # Update tracking variables for next iteration
                    current_tile = (int(pos_x / tile_size), int(pos_y / tile_size))
                    last_health = cur_health
                    last_armor = cur_armor
                    last_ammo = cur_ammo
                    last_items = cur_items
                    last_secrets = cur_secrets
                    last_kills = cur_kills
                    
                    # Calculate exploration reward for visiting new areas
                    discovery_reward = 0
                    if current_tile not in visited_tiles:
                        # Small reward for exploring new tile
                        discovery_reward = 0.05
                        visited_tiles.add(current_tile)

                    # Get agent's action (epsilon-greedy selection)
                    action, action_idx, current_screen = agent.select_action(state)
                    
                    # Execute action in game for frame_repeat frames
                    # Each action is repeated to increase temporal consistency
                    nreward = self.game.make_action(action, self.config["frame_repeat"])
                    
                    # Combine all reward signals
                    reward = nreward + discovery_reward + pickup_reward
                    
                    # Clip reward to [-1, 1] range for training stability
                    # Prevents extreme rewards from dominating learning
                    reward = np.clip(reward, -1.0, 1.0)
                    
                    # Accumulate reward for episode monitoring
                    epoch_reward += reward
                    
                    # Check if episode ended
                    done = self.game.is_episode_finished()

                    # Get next state (or zeros if episode ended)
                    next_screen = np.zeros_like(current_screen)
                    if not done:
                        next_s = self.game.get_state()
                        if next_s:
                            next_screen = next_s.screen_buffer[np.newaxis, :, :]

                    # Store experience in replay buffer: (state, action, reward, next_state, done)
                    agent.memory.append((current_screen, action_idx, reward, next_screen, done))
                    
                    # Train agent: Sample from replay buffer and update weights
                    agent.update_policy()
                
                # Decay exploration rate after each episode
                agent.epsilon = max(agent.eps_min, agent.epsilon * agent.eps_decay)
                
                # Record total reward for this episode
                agent.episode_rewards.append(epoch_reward)
                
                # Print progress information
                print(f"Epoch {epoch+1}: Total Reward = {epoch_reward:.2f}, Epsilon = {agent.epsilon:.4f}")
                
                # Save trained model weights
                agent.save_model()
        
        finally:
            # Ensure game closes even if training is interrupted
            self.game.close()


if __name__ == "__main__":
    # Configuration dictionary for training
    config = {
        "wad": "doom2.wad",          # Path to Doom WAD file
        "epochs": 100,                # Number of training episodes
        "steps_per_epoch": 2500,      # Steps per episode
        "frame_repeat": 4             # Repeat each action for 4 frames
    }
    
    # Initialize agent with 8 possible actions
    agent = Agent(n_actions=8)
    
    # Initialize Doom game environment
    my_game = Game(config)
    
    # Start training loop
    my_game.train(agent)
