import numpy as np

class StepDecay():
    def __init__(self, initial_lr: float, drop_every: int, decay_factor: float) -> None:
        self.initial_lr = initial_lr
        self.drop_every = drop_every
        self.decay_factor = decay_factor

    def __call__(self, epoch: int) -> float:
        exp = np.floor((1+epoch) / self.drop_every)
        new_lr = self.initial_lr * (self.decay_factor ** exp)
        return new_lr