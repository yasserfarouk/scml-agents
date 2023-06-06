import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from rl.PPO2 import PPO
from rl.model import OneshotPPO


def main(
        state_dim=100,
        action_dim=[10] * 10,
        lr_actor: float = 3e-3,
        lr_critic: float = 1e-2,
        gamma: float = 0.99,
        k_epochs: int = 10,
        eps_clip: float = 0.2,
        net_arch=[64, 64]
):
    d_path = 'test'
    ppo = OneshotPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=k_epochs,
        eps_clip=eps_clip,
        net_arch=net_arch,
    )

    writer = SummaryWriter(log_dir=d_path)
    ppo.set_writer(writer, 0)

    rewards = []
    for i in range(10000):
        state = np.array([1] * 10)
        state_ = torch.concatenate([int_to_array(s, 10) for s in state], dim=0)
        action = ppo.select_action(state_)

        reward = action.sum()
        ppo.buffer.rewards.append(reward)
        ppo.buffer.is_terminals.append(True)

        rewards.append(reward)
        pass # print(f'\rstep{i} reward: {reward}', end='')

        if i % 64 == 0 and i != 0:
            ppo.update()

    plt.figure()
    plt.plot(rewards)
    plt.plot(moving_average(rewards, 100))
    plt.show()


def int_to_array(s: int, n: int):
    a = torch.zeros(n)
    s = np.clip(s, a_min=0, a_max=n - 1)
    a[s] = 1
    return a


def moving_average(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec.tolist()


if __name__ == '__main__':
    main()
