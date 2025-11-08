from dataclasses import dataclass

@dataclass
class TrainingConfig:
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    final_epsilon = 0.1
    discount_factor = 0.95

    @property
    def epsilon_decay(self) -> float:
        return self.start_epsilon / (self.n_episodes / 2)