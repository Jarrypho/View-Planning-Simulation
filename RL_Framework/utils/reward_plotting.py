import matplotlib.pyplot as plt
import numpy as np


def exp_dcov(x):
    y = np.exp(x/100-1)*(x/100)
    return y

def exploration_decay_rate(x):
    exploration_value = 0.25
    decay_rate = 0.7
    num_steps = 350
    episodes = x
    y = exploration_value * (decay_rate ** (episodes/num_steps))
    return y



def plot():
    x = np.linspace(0, 100, 100000)
    y = exp_dcov(x)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x,y, 'r')

    plt.show()

if __name__ == '__main__':
    plot()
