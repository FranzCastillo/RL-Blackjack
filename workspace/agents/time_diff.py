from collections import defaultdict
import numpy as np
from workspace.utils.training_config_tabular import TrainingConfigTabular

class TDAgent:
    # Tabular, model-free Q-learning agent (off-policy), epsilon-greedy
    def __init__(self, action_n, cfg: TrainingConfigTabular):
        self.q_values = defaultdict(lambda: np.zeros(action_n))
        self.lr = cfg.learning_rate
        self.discount_factor = cfg.discount_factor

        self.epsilon = cfg.start_epsilon
        self.epsilon_decay = cfg.epsilon_decay
        self.final_epsilon = cfg.final_epsilon

        self.training_error = []

    def get_action(self, obs) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, len(self.q_values[(0, 0, False)])))  # fallback shape
        return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs,
        action,
        reward,
        terminated,
        next_obs,
    ):
        future_q = (not terminated) * np.max(self.q_values[next_obs])
        td = reward + self.discount_factor * future_q - self.q_values[obs][action]
        self.q_values[obs][action] += self.lr * td
        self.training_error.append(float(td))

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)