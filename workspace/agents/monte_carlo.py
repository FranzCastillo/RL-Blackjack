from collections import defaultdict
import numpy as np
from workspace.utils.training_config_tabular import TrainingConfigTabular

class MonteCarloAgent:
    def __init__(self, action_n, cfg: TrainingConfigTabular):
        # Q estimate as average returns
        self.q_values = defaultdict(lambda: np.zeros(action_n))

        # Sum and count of returns for averaging
        self.returns_sum = defaultdict(lambda: np.zeros(action_n))
        self.returns_count = defaultdict(lambda: np.zeros(action_n))
        self.lr = None  # not used for MC but kept for similarity to the first Agent
        self.discount_factor = cfg.discount_factor

        self.epsilon = cfg.start_epsilon
        self.epsilon_decay = cfg.epsilon_decay
        self.final_epsilon = cfg.final_epsilon

        # buffer for the current episode: list of (obs, action, reward)
        self.episode = []
        self.training_error = []

    def get_action(self, obs) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, len(self.q_values[(0, 0, False)])))
        return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        # buffer the step
        self.episode.append((obs, action, reward))

        if not terminated:
            return

        # episode finished: compute first-visit returns and update Q
        G = 0.0
        seen = set()
        # traverse backwards to compute discounted return G
        for state, act, rew in reversed(self.episode):
            G = rew + self.discount_factor * G
            key = (state, act)
            if key in seen:
                continue
            seen.add(key)

            # update sums/counts and compute new average
            self.returns_sum[state][act] += G
            self.returns_count[state][act] += 1.0
            new_q = self.returns_sum[state][act] / self.returns_count[state][act]
            old_q = self.q_values[state][act]
            self.q_values[state][act] = new_q

            # track change magnitude as training error
            self.training_error.append(float(abs(new_q - old_q)))

        # clear episode buffer
        self.episode.clear()

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)