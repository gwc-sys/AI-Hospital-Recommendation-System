from dataclasses import dataclass


@dataclass
class ModelSettings:
    random_state: int = 42
    test_size: float = 0.2
    max_depth: int = 4


settings = ModelSettings()

