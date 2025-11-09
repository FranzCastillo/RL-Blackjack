import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def add(self, obs, action, reward, done, next_obs):
        self.buffer.append((obs, action, reward, done, next_obs))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, int(batch_size))
        obs, actions, rewards, dones, next_obs = zip(*batch)
        return list(obs), np.array(actions, dtype=np.int64), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), list(next_obs)

    def __len__(self):
        return len(self.buffer)