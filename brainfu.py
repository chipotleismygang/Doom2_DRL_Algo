'''
This is a deep RL algorithm,
'''





# --- 0. THE BULLSHIT ---
# i now know what this does but it works and im still scared to touch it
import os#Handles file paths (so the AI can find the doom2.wad file).
import random#Used for the "Exploration" phase where the AI just mashes random buttons to see what happens.
import numpy as np#Handles the math for the images (converting screen pixels into numbers).
import torch #It handles all the complex calculus of the "brain."
import torch.nn as nn#"Neural Network" tools—like building blocks for the brain.
import torch.optim as optim#The "Teacher" math that updates the brain when it makes a mistake.
from collections import deque#A list that automatically deletes old memories so your RAM doesn't fill up.
from tqdm import trange#Creates the fancy progress bar while training. that i stole from the internet
import vizdoom as vzd#The bridge that lets Python control the actual Doom game engine. and if i wanted. play a pirated version of doom

# --- 1. THE BRAIN BOX ---
class DuelQNet(nn.Module):#Tells Python this class is a neural network.
    def __init__(self, in_channels, num_actions):
        super(DuelQNet, self).__init__()
        
        def conv_block(in_f, out_f):#A "Helper" function. It builds a "scanner" that looks for shapes (monsters, walls) 
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1, bias=False),#The "Scanner." It slides a filter over the image to find edges and patterns.
                nn.BatchNorm2d(out_f),#Standardizes the data so the brain doesn't get overwhelmed by sudden flashes of light or dark. we dont want the brain to have epilepsy

                nn.ReLU()#The "Filter." It kills off useless negative numbers and keeps the useful positive signals. this was a problem with old perceptrons and old machine learning algorithms
            )

        self.conv1 = conv_block(1, 8) 
        self.conv2 = conv_block(8, 8)
        self.conv3 = conv_block(8, 8)
        self.conv4 = conv_block(8, 16)
        
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 2)) #orces any screen size into a tiny 3 by 2 grid so the brain's "Decision Center" always sees the same amount of data.

        self.state_fc = nn.Sequential(#On one part of the brain. It calculates: "How good is my current situation?"

            nn.Linear(96, 64), 
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_fc = nn.Sequential(#The other half. It calculates: "How much better is this than that?"
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 8) 
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

# Security allowlist for PyTorch 2.6+ because for some reason it just completely breaks without it??? (also idk what it means. reddit said do this and yay)
torch.serialization.add_safe_globals([DuelQNet])

# --- 2. THE AGENT ---
class Agent:
    def __init__(self, n_actions, model_path="model-doom.pth"): # you can make new memories by changing every instance of this file name with another one.
        self.n_actions = n_actions
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #if i want to train it on my home computer i can use CUDA. idk what it actually is but its good. i think.
        print(f"Active Device: {self.device}")
        
        self.policy_net = DuelQNet(1, 8).to(self.device)
        self.target_net = DuelQNet(1, 8).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4) #<-----no clue what most of that means but lr=1e-4 is the Learning Rate. 0.0001. It's how much the AI "changes its mind" every time it learns something.
        #the crap the algorithm actually gets effected by
        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_decay = 0.9995
        self.eps_min = 0.01
        self.train_step = 0

        self.load_model()
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading {self.model_path}...") 
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, DuelQNet): #apparently DuelQNet and DQN cant both be used without it breaking even though its THE SAME SHIT
                self.policy_net.load_state_dict(checkpoint.state_dict())
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)

    def save_model(self): #yay no amnesia between runs anymore
        checkpoint = {'model_state_dict': self.policy_net.state_dict(), 'epsilon': self.epsilon}
        torch.save(checkpoint, self.model_path) 

    def RANDOM_BULLSHIT_GO(self, state): #RANDOM_BULLSHIT_GO should be renamed to something proper but thats boring so ill keep it for now
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

    def update_policy(self): #was the random bullshit good?
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, sn, d = zip(*batch)
        s = torch.from_numpy(np.stack(s)).to(self.device)
        a = torch.tensor(a, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        sn = torch.from_numpy(np.stack(sn)).to(self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(sn).max(1)[0]
            expected_q = r + (self.gamma * next_q * (1 - d))

        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() #its learning oooh spooky scifi
        self.train_step += 1
        if self.train_step % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# --- 3. THE GAME CLASS ---
class Game():
    def __init__(self, config):
        self.game = vzd.DoomGame()
        self.game.set_doom_game_path(os.path.abspath(config["wad"]))
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120) # Grayscale 
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.game.set_render_hud(True)
        
        # High Speed Settings to go HIGH SPEED (unpatched working 2026 cracked FREE DOWNLOAD)
        self.game.set_window_visible(True) # HEY HEY ITS RIGHT HERE ITS SIMPLE AS SHIT MAKE IT SO I CAN TOGGLE WINDOW VISIBILITY BETWEEEEENNNNN BECAUSE I WANT TO SEE THE algorithm PLAY AND SHOOT THE WAAAAAAAAAAALLLLLLLLLLLLLLLLLLLSSSSSSSSSSSSSS
        self.game.set_mode(vzd.Mode.PLAYER) 
        self.game.set_ticrate(2100) # Uncap FPS because fast
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

        #Schmovement
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
            for epoch in range(self.config["epochs"]): #an epoch is a generation but fancy name because yes. google dictionary thing says "An epoch represents one full pass of the entire training dataset through a machine learning model"
                self.game.new_episode()
                visited_tiles.clear()
                last_health, last_armor, last_ammo, last_items, last_secrets, last_kills = 100, 0, 50, 0, 0, 0,
                
                #FIXED: Use a proper steps_per_epoch loop or explode and be bad at video games
                for _ in trange(self.config["steps_per_epoch"], desc=f"Epoch {epoch+1}"):
                    if self.game.is_episode_finished():
                        self.game.new_episode()

                    state = self.game.get_state()
                    #FIXED: Use continue instead of break to avoid stopping the epoch 
                    if state is None: 
                        continue

                    vars = state.game_variables
                    # Removing the 9th variable (vars[8])
                    pos_x, pos_y, cur_health, cur_armor, cur_ammo, cur_items, cur_secrets, cur_kills = vars[0], vars[1], vars[2], vars[3], vars[4], vars[5], vars[6], vars[7]


                    pickup_reward = 0
                    if cur_health > last_health: pickup_reward += 0.5 #Yay health!
                    if cur_armor > last_armor:   pickup_reward += 0.5 #Yay armor!
                    if cur_ammo > last_ammo:     pickup_reward += 0.3 #Yay ammo!
                    if cur_items > last_items: pickup_reward += 0.5 #Yay Item!
                    if cur_secrets > last_secrets: pickup_reward += 60 #Yay secret!
                    if cur_kills > last_kills: pickup_reward += 1.2 #Yay murder!

                    if cur_health < last_health: pickup_reward += -0.7 #Oh no my health!
                    if cur_armor < last_armor: pickup_reward += -0.5 #Oh no my armor!
                    if cur_ammo < last_ammo: pickup_reward += -0.1 #Oh no my ammo!
                    if self.game.is_player_dead(): pickup_reward += -9999 #Oh no my life!
                    
                    current_tile = (int(pos_x / tile_size), int(pos_y / tile_size))
                    last_health, last_armor, last_ammo = cur_health, cur_armor, cur_ammo
                    
                    discovery_reward = 0
                    if current_tile not in visited_tiles:
                        discovery_reward = 0.05 
                        visited_tiles.add(current_tile)

                    action, action_idx, current_screen = agent.RANDOM_BULLSHIT_GO(state)
                    nreward = self.game.make_action(action, self.config["frame_repeat"])
                    reward = nreward + discovery_reward + pickup_reward
                    print(reward)
                    done = self.game.is_episode_finished()

                    next_screen = np.zeros_like(current_screen)
                    if not done:
                        next_s = self.game.get_state()
                        if next_s: next_screen = next_s.screen_buffer[np.newaxis, :, :]

                    agent.memory.append((current_screen, action_idx, reward, next_screen, done))
                    agent.update_policy()
                    agent.epsilon = max(agent.eps_min, agent.epsilon * agent.eps_decay)
                
                agent.save_model() #saves every epock so even if it crashes it should still learn SOOOMMMMEEETHIIINNNG
        finally: #sarcastic ass code syntax
            self.game.close() #HMMM WONDER WHAT THIS DOES

if __name__ == "__main__": #this is the only stuff that isnt in definitions
    config = {"wad": "doom2.wad", "epochs": 100, "steps_per_epoch": 2500, "frame_repeat": 4} #setting right at the end so easy to find could make a popup t change it each time but that might be annoying or break it
    agent = Agent(n_actions=8)
    my_game = Game(config)
    my_game.train(agent)
