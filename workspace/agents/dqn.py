# python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from workspace.utils.training_config import TrainingConfig
from workspace.agents.replay_buffer import ReplayBuffer

class _MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(128, 128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, action_n, cfg: TrainingConfig):
        self.action_n = int(action_n)
        self.cfg = cfg

        # hyperparams (allow override from cfg)
        self.lr = getattr(cfg, "learning_rate", 1e-3)
        self.gamma = getattr(cfg, "discount_factor", 0.99)
        self.batch_size = getattr(cfg, "batch_size", 64)
        self.replay_size = getattr(cfg, "replay_size", 10000)
        self.target_update_freq = getattr(cfg, "target_update_freq", 1000)
        self.learn_every = getattr(cfg, "learn_every", 1)
        self.hidden = getattr(cfg, "dqn_hidden", (128, 128))

        self.epsilon = getattr(cfg, "start_epsilon", 1.0)
        self.epsilon_decay = getattr(cfg, "epsilon_decay", 1e-4)
        self.final_epsilon = getattr(cfg, "final_epsilon", 0.1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # lazy init for networks (built on first observation)
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        self.replay = ReplayBuffer(self.replay_size)
        self.steps = 0
        self.training_error = []

        # compatibility with tabular plotting utilities:
        # map discrete obs keys -> numpy array of action values
        self.q_values = defaultdict(lambda: np.zeros(self.action_n))

    def _obs_to_tensor(self, obs):
        if isinstance(obs, np.ndarray):
            arr = obs.ravel().astype(np.float32)
        else:
            arr = np.array(obs, dtype=np.float32).ravel()
        return torch.from_numpy(arr).to(self.device)

    def _obs_to_key(self, obs):
        # convert obs to a hashable tuple of ints/bools (suitable for Blackjack)
        arr = np.array(obs).ravel()
        key = []
        for v in arr:
            if isinstance(v, (np.floating, float)):
                # if represents integer value, cast to int
                if abs(v - int(v)) < 1e-6:
                    key.append(int(v))
                else:
                    key.append(float(v))
            elif isinstance(v, (np.integer, int)):
                key.append(int(v))
            elif isinstance(v, (np.bool_, bool)):
                key.append(bool(v))
            else:
                key.append(v)
        return tuple(key)

    def _maybe_build(self, obs):
        if self.policy_net is not None:
            return
        x = self._obs_to_tensor(obs)
        input_dim = x.shape[0]
        self.policy_net = _MLP(input_dim, self.action_n, hidden=self.hidden).to(self.device)
        self.target_net = _MLP(input_dim, self.action_n, hidden=self.hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def get_action(self, obs) -> int:
        self._maybe_build(obs)
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.action_n))
        with torch.no_grad():
            t = self._obs_to_tensor(obs).unsqueeze(0)
            q = self.policy_net(t)
            return int(torch.argmax(q, dim=1).item())

    def update(self, obs, action, reward, terminated, next_obs):
        self._maybe_build(obs)
        self.replay.add(obs, int(action), float(reward), bool(terminated), next_obs)
        self.steps += 1

        # store current network Q for the discrete state for plotting/grids
        try:
            obs_key = self._obs_to_key(obs)
            with torch.no_grad():
                q_vec = self.policy_net(self._obs_to_tensor(obs).unsqueeze(0)).squeeze(0).cpu().numpy()
            self.q_values[obs_key] = q_vec.copy()
        except Exception:
            # keep running even if conversion fails
            pass

        if len(self.replay) < self.batch_size:
            return

        if (self.steps % self.learn_every) != 0:
            return

        obs_batch, actions, rewards, dones, next_batch = self.replay.sample(self.batch_size)

        # build tensors
        obs_t = torch.stack([self._obs_to_tensor(o) for o in obs_batch]).to(self.device)
        next_t = torch.stack([self._obs_to_tensor(o) for o in next_batch]).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.uint8)).to(self.device)

        # current Q
        q_vals = self.policy_net(obs_t)
        q_a = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # target Q (Double-DQN style)
        with torch.no_grad():
            next_q_policy = self.policy_net(next_t)
            next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)
            next_q_target = self.target_net(next_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + (1.0 - dones_t.float()) * (self.gamma * next_q_target)

        loss = nn.functional.mse_loss(q_a, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_error.append(float(loss.item()))

        # update stored q_values for the batch states (best-effort)
        try:
            with torch.no_grad():
                q_batch = self.policy_net(obs_t).cpu().numpy()
            for o, qv in zip(obs_batch, q_batch):
                key = self._obs_to_key(o)
                self.q_values[key] = qv.copy()
        except Exception:
            pass

        # hard update target network periodically
        if (self.steps % self.target_update_freq) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)