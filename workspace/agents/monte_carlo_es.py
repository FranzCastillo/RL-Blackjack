from workspace.utils.training_config_tabular import TrainingConfigTabular
from collections import defaultdict
import numpy as np
class MonteCarloESAgent:
    def __init__(self, action_n, cfg: TrainingConfigTabular):


        self.q_values = defaultdict(lambda: np.zeros(action_n))
        self.returns_sum = defaultdict(lambda: np.zeros(action_n))
        self.returns_count = defaultdict(lambda: np.zeros(action_n))
        self.discount_factor = cfg.discount_factor

        # buffer for current episode: list of (obs, action, reward)
        self.episode = []
        self.training_error = []

    def get_action(self, obs) -> int:
        import numpy as np
        # Exploring starts: if this is the first step of the episode,
        # choose an action uniformly at random.
        if len(self.episode) == 0:
            return int(np.random.randint(0, len(self.q_values[(0, 0, False)])))
        # thereafter act greedily w.r.t. current Q
        return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        # buffer the step
        self.episode.append((obs, action, reward))

        if not terminated:
            return

        # Episode finished: compute first-visit discounted returns and update Q
        G = 0.0
        seen = set()
        for state, act, rew in reversed(self.episode):
            G = rew + self.discount_factor * G
            key = (state, act)
            if key in seen:
                continue
            seen.add(key)

            self.returns_sum[state][act] += G
            self.returns_count[state][act] += 1.0
            new_q = self.returns_sum[state][act] / self.returns_count[state][act]
            old_q = self.q_values[state][act]
            self.q_values[state][act] = new_q
            self.training_error.append(float(abs(new_q - old_q)))

        # clear episode buffer for next episode
        self.episode.clear()

    def decay_epsilon(self):
        # No epsilon schedule needed for strict exploring starts;
        # keep method to match agent interface.
        return