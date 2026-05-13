'''
This is a deep RL algorithm,
'''

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
    def __init__(self, in_channels, num_actions):
        super(DuelQNet, self).__init__()
        
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_f),
                nn.ReLU()
            )

        self.conv1 = conv_block(1, 8) 
        self.conv2 = conv_block(8, 8)
        self.conv3 = conv_block(8, 8)
        self.conv4 = conv_block(8, 16)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.state_fc = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8) 
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), -1)
        state_value = self.state_fc(x)
        advantage = self.advantage_fc(x)
        return state_value + (advantage - advantage.mean(dim=1, keepdim=True))

torch.serialization.add_safe_globals([DuelQNet])

class Agent:
    def __init__(self, n_actions, model_path="model-doom.pth"):
        self.n_actions = n_actions
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Active Device: {self.device}")
        
        self.policy_net = DuelQNet(1, 8).to(self.device)
        self.target_net = DuelQNet(1, 8).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_decay = 0.9995
        self.eps_min = 0.01
        self.train_step = 0
        self.episode_rewards = []

        self.load_model()
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading {self.model_path}...") 
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, DuelQNet):
                self.policy_net.load_state_dict(checkpoint.state_dict())
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)

    def save_model(self):
        checkpoint = {'model_state_dict': self.policy_net.state_dict(), 'epsilon': self.epsilon}
        torch.save(checkpoint, self.model_path) 

    def select_action(self, state):
        screen = state.screen_buffer[np.newaxis, :, :] 
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            state_tensor = torch.from_numpy(screen).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = q_values[0, :self.n_actions].argmax().item()
        
        action = [0] * self.n_actions
        action[action_idx] = 1
        return action, action_idx, screen

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, sn, d = zip(*batch)
        s = torch.from_numpy(np.stack(s)).to(self.device)
        a = torch.tensor(a, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        sn = torch.from_numpy(np.stack(sn)).to(self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_actions = self.policy_net(sn).argmax(dim=1)
            next_q = self.target_net(sn).gather(1, next_actions.unsqueeze(1)).squeeze()
            expected_q = r + (self.gamma * next_q * (1 - d))

        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.train_step += 1
        if self.train_step % 1000 == 0 and self.train_step > 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

class Game():
    def __init__(self, config):
        self.game = vzd.DoomGame()
        self.game.set_doom_game_path(os.path.abspath(config["wad"]))
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.game.set_render_hud(True)
        
        self.game.set_window_visible(True)
        self.game.set_mode(vzd.Mode.PLAYER) 
        self.game.set_ticrate(2100)
        self.game.set_available_game_variables([
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.ARMOR,
            vzd.GameVariable.SELECTED_WEAPON_AMMO,
            vzd.GameVariable.ITEMCOUNT,
            vzd.GameVariable.SECRETCOUNT,
            vzd.GameVariable.KILLCOUNT
        ])

        self.game.add_available_button(vzd.Button.MOVE_FORWARD)
        self.game.add_available_button(vzd.Button.MOVE_BACKWARD)
        self.game.add_available_button(vzd.Button.MOVE_LEFT)
        self.game.add_available_button(vzd.Button.MOVE_RIGHT)
        self.game.add_available_button(vzd.Button.ATTACK)
        self.game.add_available_button(vzd.Button.TURN_LEFT)
        self.game.add_available_button(vzd.Button.TURN_RIGHT)
        self.game.add_available_button(vzd.Button.USE)
        
        self.config = config

    def train(self, agent):
        self.game.init()
        visited_tiles = set()
        tile_size = 64
        try:
            for epoch in range(self.config["epochs"]):
                self.game.new_episode()
                visited_tiles.clear()
                last_health, last_armor, last_ammo, last_items, last_secrets, last_kills = 100, 0, 50, 0, 0, 0
                epoch_reward = 0
                
                for _ in trange(self.config["steps_per_epoch"], desc=f"Epoch {epoch+1}"):
                    if self.game.is_episode_finished():
                        self.game.new_episode()

                    state = self.game.get_state()
                    if state is None: 
                        continue

                    vars = state.game_variables
                    pos_x, pos_y, cur_health, cur_armor, cur_ammo, cur_items, cur_secrets, cur_kills = vars[0], vars[1], vars[2], vars[3], vars[4], vars[5], vars[6], vars[7]

                    pickup_reward = 0
                    if cur_health > last_health: pickup_reward += 0.1
                    if cur_armor > last_armor: pickup_reward += 0.1
                    if cur_ammo > last_ammo: pickup_reward += 0.05
                    if cur_items > last_items: pickup_reward += 0.1
                    if cur_secrets > last_secrets: pickup_reward += 2.0
                    if cur_kills > last_kills: pickup_reward += 0.3

                    if cur_health < last_health: pickup_reward -= 0.15
                    if cur_armor < last_armor: pickup_reward -= 0.1
                    if cur_ammo < last_ammo: pickup_reward -= 0.02
                    if self.game.is_player_dead(): pickup_reward -= 1.0
                    
                    current_tile = (int(pos_x / tile_size), int(pos_y / tile_size))
                    last_health, last_armor, last_ammo = cur_health, cur_armor, cur_ammo
                    last_items, last_secrets, last_kills = cur_items, cur_secrets, cur_kills
                    
                    discovery_reward = 0
                    if current_tile not in visited_tiles:
                        discovery_reward = 0.05 
                        visited_tiles.add(current_tile)

                    action, action_idx, current_screen = agent.select_action(state)
                    nreward = self.game.make_action(action, self.config["frame_repeat"])
                    reward = nreward + discovery_reward + pickup_reward
                    reward = np.clip(reward, -1.0, 1.0)
                    epoch_reward += reward
                    done = self.game.is_episode_finished()

                    next_screen = np.zeros_like(current_screen)
                    if not done:
                        next_s = self.game.get_state()
                        if next_s: next_screen = next_s.screen_buffer[np.newaxis, :, :]

                    agent.memory.append((current_screen, action_idx, reward, next_screen, done))
                    agent.update_policy()
                
                agent.epsilon = max(agent.eps_min, agent.epsilon * agent.eps_decay)
                agent.episode_rewards.append(epoch_reward)
                print(f"Epoch {epoch+1}: Total Reward = {epoch_reward:.2f}, Epsilon = {agent.epsilon:.4f}")
                agent.save_model()
        finally:
            self.game.close()

if __name__ == "__main__":
    config = {"wad": "doom2.wad", "epochs": 100, "steps_per_epoch": 2500, "frame_repeat": 4}
    agent = Agent(n_actions=8)
    my_game = Game(config)
    my_game.train(agent)
