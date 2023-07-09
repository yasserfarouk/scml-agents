import datetime
import os
import pathlib
import sys
from typing import Dict, List

from negmas import Contract, ResponseType, SAOResponse
from negmas.outcomes import Outcome
from scml.oneshot import *
from torch.utils.tensorboard import SummaryWriter

from .agents.utils import opp_name_from_contract
from .rl.action import (
    IndActionManager,
    IndMultiDiscreteAM,
    SyncActionManager,
    SyncMultiDiscreteAM,
)
from .rl.model import OneshotPPO
from .rl.observe import (
    IndMultiDiscreteOM,
    IndObserveManager,
    SyncMultiDiscreteOM,
    SyncObserveManager,
)
from .tutorials.example_agents_oneshot import SimpleAgent, SyncAgent

# sys.path.append(os.path.dirname(__file__))


__all__ = ["RLIndAgent"]


class RLSyncAgent(SyncAgent):
    step_count = [0, 0]
    sim_count = [0, 0]
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer: List[SummaryWriter] = [None, None]
    step_rewards = []
    reward_th = [-1000, 1000]

    def __init__(
        self,
        *args,
        model_name: str = "Sync_PPO_S-MD-AM_S-MD-OM_64-32-32_BEST.pth",
        threshold: float = 0.5,
        action_manager: SyncActionManager = None,
        observe_manager: SyncObserveManager = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        k_epochs: int = 10,
        eps_clip: float = 0.2,
        action_std: float = 0.6,  # 初期の行動の分布の標準偏差
        update_freq: int = 2,
        cont_reward: int = 0,
        first_quantity: int = 5,
        update_model: bool = False,
        save_model: bool = False,
        tb_log: bool = False,
        accept: bool = True,
        episode_as_simulation: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # model parameters
        self.threshold = threshold
        self.action_manager = action_manager
        self.observe_manager = observe_manager
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.action_std = action_std
        self.net_arch = [64, 32, 32]

        # model
        if self.action_manager is None:
            self.action_manager = SyncMultiDiscreteAM()
        if self.observe_manager is None:
            self.observe_manager = SyncMultiDiscreteOM()
        self.ppo = OneshotPPO(
            state_dim=self.observe_manager.state_dim,
            action_dim=self.action_manager.action_space,
            lr_actor=self.lr_actor,
            lr_critic=self.lr_critic,
            gamma=self.gamma,
            K_epochs=self.k_epochs,
            eps_clip=self.eps_clip,
            net_arch=self.net_arch,
        )
        self.model_name = model_name
        self.checkpoint_path = None
        self.obs = [0, None, None]
        self.update_model = update_model
        self.save_model = save_model
        self.update_freq = update_freq

        # mdp params
        self.cont_reward = cont_reward

        # agent state
        self.is_selling = True
        self.idx = 0
        self.balances = []
        self.contracts: Dict[str, Contract] = {}
        self.first_quantity = first_quantity
        self.accept = accept
        self.episode_as_simulation = episode_as_simulation

        # log
        self.tb_log = tb_log
        self.step_need = 0
        self.round_rewards = []
        self.step_rewards = []
        self.round_actions = []

    def init(self):
        super().init()

        directory = pathlib.Path(__file__).parent.parent / "PPO_preTrained"
        self.is_selling = self.awi.my_suppliers[0] == "SELLER"
        self.idx = 0 if self.is_selling else 1
        env_name = "OneShot-" + ("Seller" if self.is_selling else "Buyer")
        directory = os.path.join(directory, env_name)
        if self.model_name:
            self.checkpoint_path = os.path.join(directory, "models", self.model_name)
            self.ppo.load(self.checkpoint_path)
        else:
            os.makedirs(directory, exist_ok=True)
            self.checkpoint_path = os.path.join(
                directory,
                "models",
                f'Sync_PPO_{self.action_manager}_{self.observe_manager}_{"-".join([str(i) for i in self.net_arch])}.pth',
            )
        # print("save checkpoint path : " + self.checkpoint_path)

        if RLSyncAgent.writer[self.idx] is None:
            d_path = os.path.join(directory, "tb_log", self.now)
            os.makedirs(d_path, exist_ok=True)
            RLSyncAgent.writer[self.idx] = SummaryWriter(log_dir=d_path)
        self.balances.append(self.awi.current_balance)

        self.ppo.set_writer(RLSyncAgent.writer[self.idx], self.idx)

    def before_step(self):
        super().before_step()
        self.obs = [0, None, None]
        self.round_rewards.clear()
        self.round_actions.clear()

        self.step_need = self._needed()
        self.contracts: Dict[str, Contract] = {}

    def step(self):
        # calculate reward
        self.balances.append(self.awi.current_balance)
        profit = self.ufun.from_contracts(
            list(self.contracts.values()), ignore_exogenous=False
        )
        diff = abs(self.step_need - self.secured) ** 2
        reward = profit

        # record reward
        if self.ppo.buffer.rewards:
            self.ppo.buffer.rewards[-1] = reward
        if len(self.round_rewards):
            self.round_rewards[-1] = reward
        pass  # print(f'\r reward on step{self.awi.current_step}: {reward}', end='')

        # step reward
        step_reward = sum(self.round_rewards)
        self.step_rewards.append(step_reward)
        RLSyncAgent.step_rewards.append(step_reward)

        # is terminated
        if self.ppo.buffer.is_terminals:
            if not self.episode_as_simulation:
                self.ppo.buffer.is_terminals[-1] = True
            elif self.awi.current_step == self.awi.n_steps - 1:
                self.ppo.buffer.is_terminals[-1] = True

        # tensorboard log
        if self.tb_log:
            self.writer[self.idx].add_scalar(
                "reward/step_profit", profit, self.step_count[self.idx]
            )

            self.writer[self.idx].add_scalar(
                "reward/step", step_reward, self.step_count[self.idx]
            )
            self.writer[self.idx].add_scalar(
                "stats/n_actions_per_step",
                len(self.round_actions),
                self.step_count[self.idx],
            )
            if self.step_need != 0:
                self.writer[self.idx].add_scalar(
                    "stats/needs_complete_rate",
                    float(self.secured) / self.step_need,
                    self.step_count[self.idx],
                )

        # update model using replay buffer
        if len(self.ppo.buffer.rewards) >= self.update_freq and self.update_model:
            self.ppo.update()

        RLSyncAgent.step_count[self.idx] += 1

        # simulation finish process
        if self.awi.current_step == self.awi.n_steps - 1:
            # calculate simulation reward
            sim_reward = sum(self.step_rewards)
            if self.tb_log:
                self.writer[self.idx].add_scalar(
                    "reward/simulation", sim_reward, self.sim_count[self.idx]
                )
            RLSyncAgent.sim_count[self.idx] += 1

            if self.save_model:
                # save the trained model
                self.ppo.save(self.checkpoint_path)

                if RLSyncAgent.reward_th[self.idx] < sim_reward:
                    self.ppo.save(f'{self.checkpoint_path.split(".")[0]}_BEST.pth')

    def on_negotiation_success(self, contract, mechanism):
        # todo: 交渉成功時の報酬を記述
        super().on_negotiation_success(contract, mechanism)

        opp_id = opp_name_from_contract(self.is_selling, contract)
        self.contracts[opp_id] = contract

    def first_proposals(self):
        nmi = self.get_nmi(list(self.negotiators.keys())[0])
        q = self.first_quantity
        t = self.awi.current_step
        p = (
            nmi.issues[UNIT_PRICE].min_value
            if self.is_selling
            else nmi.issues[UNIT_PRICE].max_value
        )

        offers = {k: (q, t, p) for k in self.negotiators.keys()}

        responses = self.counter_all(offers, self.states)

        proposals = {k: v.outcome for k, v in responses.items()}

        return proposals

    def counter_all(self, offers, states):
        responses = {
            k: SAOResponse(ResponseType.REJECT_OFFER, (-1, -1, -1))
            for k in sorted(self.negotiators.keys())
        }
        my_needs = self._needed()
        nmi = self.get_nmi(list(offers.keys())[0])

        # pad the offer
        for k in responses.keys():
            if k not in offers.keys():
                offers[k] = (0, 0, 0)

        # select action

        action = self.ppo.select_action(
            self.observe_manager.encode(offers, states, my_needs, nmi)
        )

        # decode action to responses
        responses = self.action_manager.decode(
            action, responses, nmi, self.awi.current_step
        )
        if not self.accept:
            # reject all offers when accept flag is False
            responses = {
                k: SAOResponse(ResponseType.REJECT_OFFER, v.outcome)
                for k, v in responses.items()
            }
        if my_needs <= 0:
            # If needs are not met or in the final round, reject all offers
            responses = {
                k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                for k, v in responses.items()
            }

        # add buffer and log
        reward = self.cont_reward
        self.ppo.buffer.rewards.append(reward)
        self.ppo.buffer.is_terminals.append(False)
        self.round_rewards.append(reward)
        self.round_actions.append(responses)

        return responses


class RLIndAgent(SimpleAgent):
    step_count = [0, 0]
    sim_count = [0, 0]
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer: List[SummaryWriter] = [None, None]
    step_rewards = []
    reward_th = [-1000, 1000]

    def __init__(
        self,
        *args,
        model_name: str = "Ind_PPO_I-MD-AM_I-MD-OM_64-32-32_BEST.pth",
        threshold: float = 0.5,
        action_manager: IndActionManager = None,
        observe_manager: IndObserveManager = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        k_epochs: int = 10,
        eps_clip: float = 0.2,
        action_std: float = 0.6,  # 初期の行動の分布の標準偏差
        update_freq: int = 2,
        cont_reward: int = 0,  # ラウンド経過による報酬
        first_quantity: int = 5,
        needs_coef: float = 1.0,
        update_model: bool = False,
        save_model: bool = False,
        tb_log: bool = False,
        accept: bool = True,
        episode_as_simulation: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # model parameters
        self.threshold = threshold
        self.action_manager = action_manager
        self.observe_manager = observe_manager
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.action_std = action_std
        self.net_arch = [64, 32, 32]

        # model
        if self.action_manager is None:
            self.action_manager = IndMultiDiscreteAM()
        if self.observe_manager is None:
            self.observe_manager = IndMultiDiscreteOM()
        self.ppo = OneshotPPO(
            state_dim=self.observe_manager.state_dim,
            action_dim=self.action_manager.action_space,
            lr_actor=self.lr_actor,
            lr_critic=self.lr_critic,
            gamma=self.gamma,
            K_epochs=self.k_epochs,
            eps_clip=self.eps_clip,
            net_arch=self.net_arch,
        )
        self.model_name = model_name
        self.checkpoint_path = None
        self.obs = [0, None, None]
        self.update_model = update_model
        self.save_model = save_model
        self.update_freq = update_freq

        # agent params
        self.cont_reward = cont_reward
        self.needs_coef = needs_coef

        # agent state
        self.is_selling = True
        self.idx = 0
        self.balances = []
        self.opp_offers: Dict[str, tuple] = {}
        self.contracts: Dict[str, Contract] = {}
        self.first_quantity = first_quantity
        self.accept = accept
        self.episode_as_simulation = episode_as_simulation

        # log
        self.tb_log = tb_log
        self.step_need = 0
        self.round_rewards = []
        self.step_rewards = []
        self.round_actions = []

    def init(self):
        super().init()

        directory = pathlib.Path(__file__).parent.parent / "PPO_preTrained"
        self.is_selling = self.awi.my_suppliers[0] == "SELLER"
        self.idx = 0 if self.is_selling else 1
        env_name = "OneShot-" + ("Seller" if self.is_selling else "Buyer")
        directory = os.path.join(directory, env_name)
        if self.model_name:
            self.checkpoint_path = os.path.join(directory, "models", self.model_name)
            self.ppo.load(self.checkpoint_path)
        else:
            os.makedirs(directory, exist_ok=True)
            self.checkpoint_path = os.path.join(
                directory,
                "models",
                f'Ind_PPO_{self.action_manager}_{self.observe_manager}_{"-".join([str(i) for i in self.net_arch])}.pth',
            )
        # print("save checkpoint path : " + self.checkpoint_path)

        if RLIndAgent.writer[self.idx] is None:
            d_path = os.path.join(directory, "tb_log", self.now)
            os.makedirs(d_path, exist_ok=True)
            RLIndAgent.writer[self.idx] = SummaryWriter(log_dir=d_path)
        self.balances.append(self.awi.current_balance)

        self.ppo.set_writer(RLIndAgent.writer[self.idx], self.idx)

    def before_step(self):
        super().before_step()
        self.obs = [0, None, None]
        self.round_rewards.clear()
        self.round_actions.clear()

        self.step_need = self._needed()
        self.opp_offers: Dict[str, tuple] = {}
        self.contracts: Dict[str, Contract] = {}

    def step(self):
        # calculate reward
        self.balances.append(self.awi.current_balance)
        profit = self.ufun.from_contracts(
            list(self.contracts.values()), ignore_exogenous=False
        )
        diff = abs(self.step_need - self.secured) ** 2
        reward = profit

        # record reward
        if self.ppo.buffer.rewards:
            self.ppo.buffer.rewards[-1] = reward
        if len(self.round_rewards):
            self.round_rewards[-1] = reward
        pass  # print(f'\r reward on step{self.awi.current_step}: {reward}', end='')

        # step reward
        step_reward = sum(self.round_rewards)
        self.step_rewards.append(step_reward)
        RLSyncAgent.step_rewards.append(step_reward)

        # is terminated
        if self.ppo.buffer.is_terminals:
            if not self.episode_as_simulation:
                self.ppo.buffer.is_terminals[-1] = True
            elif self.awi.current_step == self.awi.n_steps - 1:
                self.ppo.buffer.is_terminals[-1] = True

        # tensorboard log
        if self.tb_log:
            self.writer[self.idx].add_scalar(
                "reward/step_profit", profit, self.step_count[self.idx]
            )

            self.writer[self.idx].add_scalar(
                "reward/step", step_reward, self.step_count[self.idx]
            )
            self.writer[self.idx].add_scalar(
                "stats/n_actions_per_step",
                len(self.round_actions),
                self.step_count[self.idx],
            )
            if self.step_need != 0:
                self.writer[self.idx].add_scalar(
                    "stats/needs_complete_rate",
                    float(self.secured) / self.step_need,
                    self.step_count[self.idx],
                )

        # update model using replay buffer
        if len(self.ppo.buffer.rewards) >= self.update_freq and self.update_model:
            self.ppo.update()

        RLIndAgent.step_count[self.idx] += 1

        # simulation finish process
        if self.awi.current_step == self.awi.n_steps - 1:
            # calculate simulation reward
            sim_reward = sum(self.step_rewards)
            if self.tb_log:
                self.writer[self.idx].add_scalar(
                    "reward/simulation", sim_reward, self.sim_count[self.idx]
                )
            RLIndAgent.sim_count[self.idx] += 1

            if self.save_model:
                # save the trained model
                self.ppo.save(self.checkpoint_path)

                if RLIndAgent.reward_th[self.idx] < sim_reward:
                    self.ppo.save(f'{self.checkpoint_path.split(".")[0]}_BEST.pth')

    def on_negotiation_success(self, contract, mechanism):
        # todo: 交渉成功時の報酬を記述
        super().on_negotiation_success(contract, mechanism)

        opp_id = opp_name_from_contract(self.is_selling, contract)
        self.contracts[opp_id] = contract

    def propose(self, negotiator_id: str, state) -> "Outcome":
        nmi = self.get_nmi(negotiator_id)
        if negotiator_id not in self.opp_offers.keys():
            q = self.first_quantity
            t = self.awi.current_step
            p = (
                nmi.issues[UNIT_PRICE].min_value
                if self.is_selling
                else nmi.issues[UNIT_PRICE].max_value
            )
            self.respond(negotiator_id, state)

        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None

        return self.opp_offers.get(negotiator_id, None)

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if not offer:
            self.opp_offers[negotiator_id] = None
            return ResponseType.REJECT_OFFER
        my_needs = self._needed(negotiator_id)
        nmi = self.get_nmi(negotiator_id)

        # select action
        action = self.ppo.select_action(
            self.observe_manager.encode(offer, state, my_needs, nmi)
        )

        # decode action to responses
        response = self.action_manager.decode(action, nmi, self.awi.current_step)
        self.opp_offers[negotiator_id] = response.outcome

        # add buffer and log
        reward = self.cont_reward
        self.ppo.buffer.rewards.append(reward)
        self.ppo.buffer.is_terminals.append(False)
        self.round_rewards.append(reward)
        self.round_actions.append(response)

        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            response.response
            if offer[QUANTITY] <= my_needs * self.needs_coef
            else ResponseType.REJECT_OFFER
        )
