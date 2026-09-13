import numpy as np
import torch
import torch.nn.functional as F

global WORLD

# WORLD = [
#     [2, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Start (Top-Left)
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
#     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # Central Obstacles
#     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 3, 0], # Goal (Bottom-Right area)
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
# ]

# model.py

# 2 = Start, 3 = Goal, 1 = Wall, 0 = Empty
WORLD = [
    [2, 0, 1, 0, 0],  # Start (Top-Left)
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],  # The agent has to go around this
    [0, 1, 0, 1, 0],  
    [0, 0, 0, 0, 3]   # Goal (Bottom-Right)
]

def weighted_choice(p_values: np.ndarray):
    actions_idxs = range(len(p_values))
    decided_action_idx = np.random.choice(actions_idxs, p=p_values)
    return decided_action_idx


WORLD_WITH_BOUNDARY = np.pad(WORLD, pad_width=1, mode='constant', constant_values=1)


class Agent:
    def __init__(self, pos: tuple) -> None:
        self.pos = pos # (y, x) tuple, [more like (row, col)]

        self.q_table = np.zeros((5, 5, 8)) 

        self.COMPLETED = False
        self.STEPS_AVAILBLE = 20
        self.ITERATIONS = 0
        self.SUCCESS = 0
        self.iter_memory = []

    
    def reset(self):
        self.pos = (0,0)
        self.COMPLETED = False
        self.STEPS_AVAILBLE = np.clip(20 - self.SUCCESS, 8, 20)
        self.ITERATIONS += 1
        self.iter_memory = []
        
        WORLD[0][0] = 2
        
    def move(self):
        y, x = self.pos

        raw_values = torch.tensor(self.q_table[y][x], dtype=torch.float32)
        probs = F.softmax(raw_values, dim=0).numpy()
    
        action = weighted_choice(probs)
        self.STEPS_AVAILBLE -= 1

        if action == 0:
            new_pos = (y - 1, x)     # Up
        elif action == 1:
            new_pos = (y -1 , x+1)     # Up-Right
        elif action == 2:
            new_pos = (y, x + 1)     # Right
        elif action == 3:
            new_pos = (y+1, x + 1)     # Right-Down
        elif action == 4:
            new_pos = (y + 1, x)     # Down
        elif action == 5:
            new_pos = (y + 1, x - 1)     # Down-Left
        elif action == 6:
            new_pos = (y, x - 1)     # Left
        elif action == 7:
            new_pos = (y - 1, x - 1)     # Left-Up
        else:
            new_pos = (y, x)         # No Move (Shouldn't happen)

        new_y, new_x = new_pos

        check_y = new_y + 1
        check_x = new_x + 1

        # 4. FIX: Logic Chain (if-elif-else)
        # We must use 'elif' so only ONE outcome happens per turn.
        
        if self.STEPS_AVAILBLE <= 0:
             # Out of steps
            self.q_table[y][x][action] += 0.1 * (-15 - 0.9* self.q_table[y][x][action])

            # Agent Learns from the run's memory.
            
            for i in range(len(self.iter_memory)-1):
                pos = self.iter_memory[i][0]
                act = self.iter_memory[i][1]
                self.q_table[pos[0]][pos[1]][act] -= 10 * (i/len(self.iter_memory))

            self.COMPLETED = True
            
        elif WORLD_WITH_BOUNDARY[check_y][check_x] == 1:
            # HIT WALL
            self.q_table[y][x][action] = -100
            # Do NOT update self.pos. The agent stays at (y, x).
            
        elif WORLD_WITH_BOUNDARY[check_y][check_x] == 3:
            # HIT GOAL
            # Reward + Discounted Future (0 since finished)
            self.q_table[y][x][action] += 0.1 * (15 - 0.9 * self.q_table[y][x][action])

            # Agent Learns from the run's memory.
            for i in range(len(self.iter_memory)-1):
                pos = self.iter_memory[i][0]
                act = self.iter_memory[i][1]
                self.q_table[pos[0]][pos[1]][act] += 10

            self.COMPLETED = True
            # Optional: Move onto the goal visually
            self.pos = new_pos
            self.SUCCESS += 1

        else:
            # VALID MOVE (Empty Space)
            # 1. Move the agent
            self.pos = new_pos
            
            # 2. Calculate Q-Update
            max_future_q = np.max(self.q_table[new_y][new_x])
            current_q = self.q_table[y][x][action]
            
            # NewQ = OldQ + Alpha * (Reward + Gamma * MaxFuture - OldQ)
            new_q = current_q + 0.1 * (-1 + 0.6 * max_future_q - current_q)
            self.q_table[y][x][action] = new_q
        
        self.iter_memory.append((self.pos, action))
        WORLD[self.pos[0]][self.pos[1]] = 2
        