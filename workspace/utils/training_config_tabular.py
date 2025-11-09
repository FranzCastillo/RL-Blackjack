from dataclasses import dataclass

@dataclass
class TrainingConfigTabular:
    learning_rate: float = 0.1
    n_episodes: int = 200_000
    start_epsilon: float = 1.0
    final_epsilon: float = 0.05
    discount_factor: float = 1.0

    @property
    def epsilon_decay(self) -> float:
        return self.start_epsilon / (self.n_episodes / 2)