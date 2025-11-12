from dataclasses import dataclass

@dataclass
class TrainingConfigDQN:
    learning_rate: float = 1e-3
    n_episodes: int = 500_000
    start_epsilon: float = 1.0
    final_epsilon: float = 0.05
    discount_factor: float = 1.0

    # DQN-specific
    batch_size: int = 64
    replay_size: int = 10_000
    target_update_freq: int = 1000
    learn_every: int = 1
    dqn_hidden: tuple = (32, 32)

    @property
    def epsilon_decay(self) -> float:
        return self.start_epsilon / (self.n_episodes / 2)