import os
import sys
from statistics import mean

from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from typing import List
from torch.distributions import Categorical

from PPO2 import PPO, device


class OneshotActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int | List[int],
        net_arch: List[int],
    ):
        super().__init__()
        self.multi_discrete = isinstance(action_dim, list)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # actor
        actor_net = []
        if self.multi_discrete:
            n_input = state_dim
            for n_output in net_arch:
                actor_net += [
                    nn.Linear(n_input, n_output),
                    nn.Tanh(),
                ]
                n_input = n_output
            actor_net += [nn.Linear(n_input, sum(action_dim)), nn.Tanh()]
        else:
            n_input = state_dim
            for n_output in net_arch:
                actor_net += [
                    nn.Linear(n_input, n_output),
                    nn.Tanh(),
                ]
                n_input = n_output
            actor_net += [nn.Linear(n_input, action_dim), nn.Softmax(dim=1)]
        self.actor = nn.Sequential(*actor_net)

        # critic
        critic_net = []
        n_input = state_dim
        for n_output in net_arch:
            critic_net += [
                nn.Linear(n_input, n_output),
                nn.Tanh(),
            ]
            n_input = n_output
        critic_net += [nn.Linear(n_input, 1)]
        self.critic = nn.Sequential(*critic_net)

    def act(self, state):
        if self.multi_discrete:
            action_probs = self.actor(state)
            dist = [
                Categorical(logits=split)
                for split in torch.split(action_probs, tuple(self.action_dim), dim=0)
            ]
            action = torch.stack([d.sample() for d in dist], dim=0)
            action_logprob = torch.stack(
                [d.log_prob(a) for d, a in zip(dist, action)], dim=0
            ).sum(dim=0)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        if self.multi_discrete:
            action_probs = self.actor(state)
            dist = [
                Categorical(logits=split)
                for split in torch.split(action_probs, tuple(self.action_dim), dim=1)
            ]
            action_logprob = torch.stack(
                [d.log_prob(a) for d, a in zip(dist, torch.unbind(action, dim=1))],
                dim=1,
            ).sum(dim=1)
            dist_entropy = torch.stack([d.entropy() for d in dist], dim=1).sum(dim=1)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action_logprob = dist.log_prob(action)
            dist_entropy = dist.entropy()

        state_values = self.critic(state)

        return action_logprob, state_values, dist_entropy


class OneshotPPO(PPO):
    _n_update = [0, 0]

    def __init__(
        self,
        state_dim: int,
        action_dim: int | List[int],
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        K_epochs: int,
        eps_clip: float,
        net_arch: List[int],
    ):
        super().__init__(
            state_dim,
            sum(action_dim),
            lr_actor,
            lr_critic,
            gamma,
            K_epochs,
            eps_clip,
            isinstance(action_dim, list),
        )

        self.policy = OneshotActorCritic(state_dim, action_dim, net_arch).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = OneshotActorCritic(state_dim, action_dim, net_arch)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self._writer = None
        self._idx = None

    def set_writer(self, writer: SummaryWriter, idx: int):
        self._writer = writer
        self._idx = idx

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # log reward
        self._writer.add_scalar(
            "reward/rollout,", mean(rewards), self._n_update[self._idx]
        )

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            # log loss
            self._writer.add_scalar(
                "loss", loss.detach().mean(), self._n_update[self._idx]
            )

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        # increment count
        OneshotPPO._n_update[self._idx] += 1
