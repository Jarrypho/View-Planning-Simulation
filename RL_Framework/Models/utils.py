'''
Copyright (C) 2022  Jan-Philipp Kaiser

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
'''

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
