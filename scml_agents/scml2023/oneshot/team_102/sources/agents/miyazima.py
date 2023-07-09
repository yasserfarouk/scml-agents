import os
import random
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import markers
from matplotlib import pyplot as plt
from negmas import ResponseType
from PPO_Emb31 import PPO_Emb
from scml.oneshot import *

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_columns", 50)
import datetime

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter

N_SIMULATIONS = 30000  # 50000
N_SIMULATIONS_SHORT = 3000  # 3000
N_SIMULATIONS_SHORTS = [1000, 2000, 3000]
N_SIMULATIONS_TEST = 50  # 100


class PPOTrainAgent(OneShotAgent):
    # クラス変数
    _count = -1  # インスタンス化回数をカウント
    total_rewards = []
    simulation_total_rewards = []
    opp_simulation_total_rewards = {}
    dt_now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    writer = None

    # シミュレーションの開始時(インスタンス化直後)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        PPOTrainAgent._count += 1
        if PPOTrainAgent._count == 0:
            PPOTrainAgent.writer = SummaryWriter(log_dir=f"logs/{PPOTrainAgent.dt_now}")

    # シミュレーションの開始時(awiのセットアップ完了後)
    def init(self):
        super().init()

        self.state_dim, self.action_dim = 4, 1
        self.lr_actor, self.lr_critic = (
            3e-6,
            1e-5,
        )  # dafault:0.0003(3e-4),0.001(1e-3), 割と上手くいってた3e-5,1e-4
        # self.lr_actor,self.lr_critic=3e-5,1e-4
        # self.lr_actor,self.lr_critic=3e-4,1e-3

        self.gamma = 0.99  # default:0.99
        self.K_epochs = 200  # 200
        self.eps_clip = 0.2
        self.reward_for_neg_failure = -0.5
        self.reward_for_opp_concession = 0.01

        # 交渉相手リストの作成
        if self.awi.my_suppliers[0] == "SELLER":
            self.my_negotiators = self.awi.my_consumers
        else:
            self.my_negotiators = self.awi.my_suppliers

        self.emb_dim = 0
        # self.emb_dim=16
        self.lr_emb = 1e-6  # 1e-6
        self.ppo = PPO_Emb(
            self.state_dim,
            self.action_dim,
            self.emb_dim,
            self.lr_actor,
            self.lr_critic,
            self.lr_emb,
            self.gamma,
            self.K_epochs,
            self.eps_clip,
            True,
            [negotiator_id[2:-2] for negotiator_id in self.my_negotiators],
        )
        self.ppo_action_flag = {}  # self.ppo.select_actionを呼び出したか否かk
        self.simulation_total_reward = 0
        self.opp_step_rewards = {}
        self.embs = {}
        self.all_embs = {}
        self.n_agreements = {}
        self.n_exogenous_contracts = {}
        for i, negotiator_id in enumerate(self.my_negotiators):
            negotiator_id = negotiator_id[2:-2]

            self.ppo.buffer[negotiator_id].clear()
            self.opp_step_rewards[negotiator_id] = []
            self.embs[negotiator_id] = list(np.random.rand(self.emb_dim))
            self.all_embs[negotiator_id] = []
            self.n_agreements[negotiator_id] = 0
            self.n_exogenous_contracts[negotiator_id] = 0

        directory = "PPO_preTrained"
        # env_name = "OneShot-v2-"+(f"{'-'.join(self.awi.my_consumers)}_seller" if self.awi.my_suppliers[0]=="SELLER" else f"{'-'.join(self.awi.my_suppliers)}_buyer")
        env_name = "OneShot-v3-" + (
            f"seller" if self.awi.my_suppliers[0] == "SELLER" else f"buyer"
        )
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory = directory + "/" + env_name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        random_seed = 0  # set random seed if required (0 = no random seed)
        self.checkpoint_path = directory + "PPO_{}_{}_{}".format(
            env_name, random_seed, PPOTrainAgent.dt_now
        )
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        if PPOTrainAgent._count == 0:
            """from tkinter import filedialog
            checkpoint_directory = filedialog.askdirectory()
            if checkpoint_directory !='':
                #self.ppo.load_repfunc(checkpoint_directory)
                self.ppo.load(checkpoint_directory)"""
            pass
        else:
            if PPOTrainAgent._count == 1:
                pass  # print(PPOTrainAgent._count, "load checkpoint path : " + self.checkpoint_path)
            self.ppo.load(self.checkpoint_path)

    # １日の始まり
    def before_step(self):
        self.secured = 0
        self.obs = {}
        self.action = {}
        self.contracts = {}
        if self.awi.my_suppliers[0] == "SELLER":
            for negotiator_id in self.my_negotiators:
                self.obs[negotiator_id[2:-2]] = [
                    self.awi.current_step / (self.awi.n_steps - 1),
                    0,
                    1,
                    0,
                ]  # 日,交渉ラウンド,前回の自分のオファーの単価,直前の相手のオファーの単価
            self.contracts["EXOGENOUS"] = {
                "unit_price": self.awi.current_exogenous_input_price
                / self.awi.current_exogenous_input_quantity
                if self.awi.current_exogenous_input_quantity != 0
                else -1,
                "quantity": self.awi.current_exogenous_input_quantity,
            }
        else:
            for negotiator_id in self.my_negotiators:
                self.obs[negotiator_id[2:-2]] = [
                    self.awi.current_step / (self.awi.n_steps - 1),
                    0,
                    0,
                    1,
                ]
            self.contracts["EXOGENOUS"] = {
                "unit_price": self.awi.current_exogenous_output_price
                / self.awi.current_exogenous_output_quantity
                if self.awi.current_exogenous_output_quantity != 0
                else -1,
                "quantity": self.awi.current_exogenous_output_quantity,
            }

        self.opp_pre_offer = {}
        self.first_proposer = {}
        for negotiator_id in self.my_negotiators:
            negotiator_id = negotiator_id[2:-2]

            self.ppo_action_flag[negotiator_id] = False  # その日にselect_actionを行ったらTrueにする
            self.first_proposer[negotiator_id] = None
            if negotiator_id not in PPOTrainAgent.opp_simulation_total_rewards.keys():
                PPOTrainAgent.opp_simulation_total_rewards[negotiator_id] = []
            if (
                negotiator_id in self.ppo.episode_buffer.keys()
                and len(self.ppo.episode_buffer[negotiator_id]) > 0
            ):
                self.embs[negotiator_id] = self.ppo.get_emb(
                    self.ppo.episode_buffer[negotiator_id][-1]
                )
                self.all_embs[negotiator_id].append(self.embs[negotiator_id])

        # print(f'{self.awi.current_step} {self.awi.agent.id} {self.awi.exogenous_contract_summary} current_exogenous_input_quantity:{self.awi.current_exogenous_input_quantity} tp:{self.awi.trading_prices} penalty:{self.awi.current_shortfall_penalty} dispcost:{self.awi.current_disposal_cost}')

    def propose(self, negotiator_id: str, state) -> "Outcome":
        ami = self.get_nmi(negotiator_id)

        negotiator_id = negotiator_id[2:-2]

        if self.first_proposer[negotiator_id] == None:  # 自分が先手で提案する日の第一ラウンド
            self.first_proposer[negotiator_id] = 1
            self.action[negotiator_id] = self.ppo.select_action(
                self.obs[negotiator_id] + self.embs[negotiator_id], negotiator_id
            )
            self.ppo.buffer[negotiator_id].rewards.append(0)
            self.ppo.buffer[negotiator_id].is_terminals.append(False)
            self.ppo_action_flag[negotiator_id] = True

        offer = [-1, -1, -1]
        offer[TIME] = self.awi.current_step
        offer[QUANTITY] = 1  # 学習時は量が過剰であることでリジェクトされるのを防ぐために常に1個で提案
        offer[UNIT_PRICE] = (
            self.action[negotiator_id]
            * (ami.issues[UNIT_PRICE].max_value - ami.issues[UNIT_PRICE].min_value)
            + ami.issues[UNIT_PRICE].min_value
        )
        # 状態を記録
        self.obs[negotiator_id][2] = self.action[negotiator_id]  # 直前の自分のオファー価格

        return offer

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        ami = self.get_nmi(negotiator_id)

        negotiator_id = negotiator_id[2:-2]

        if self.first_proposer[negotiator_id] == None:  # 自分が後手で提案する日の第一ラウンド
            self.first_proposer[negotiator_id] = 0

        # 相手のofferの変化はその前の自分の行動が影響している可能性があるので，一つ前のbufferのrewardに譲歩等に関する報酬を与える
        if (
            negotiator_id in self.opp_pre_offer.keys()
            and not self.ppo.buffer[negotiator_id].is_terminals[-1]
        ):
            if offer[UNIT_PRICE] > self.opp_pre_offer[negotiator_id][UNIT_PRICE]:
                self.ppo.buffer[negotiator_id].rewards[-1] = (
                    self.reward_for_opp_concession
                    if self._is_selling(ami)
                    else -self.reward_for_opp_concession
                )
            elif offer[UNIT_PRICE] < self.opp_pre_offer[negotiator_id][UNIT_PRICE]:
                self.ppo.buffer[negotiator_id].rewards[-1] = (
                    -self.reward_for_opp_concession
                    if self._is_selling(ami)
                    else self.reward_for_opp_concession
                )

        self.obs[negotiator_id][1] = state.step / (ami.n_steps - 1)  # 交渉ラウンド
        self.action[negotiator_id] = self.ppo.select_action(
            self.obs[negotiator_id] + self.embs[negotiator_id], negotiator_id
        )
        self.ppo.buffer[negotiator_id].rewards.append(0)
        self.ppo.buffer[negotiator_id].is_terminals.append(False)
        self.ppo_action_flag[negotiator_id] = True

        # ここ変えた(行動選択より前にここに相当する処理入れた)
        """if negotiator_id in self.opp_pre_offer.keys():
            if self._is_selling(ami):
                self.ppo.buffer[negotiator_id].rewards.append(self.reward_for_opp_concession if offer[UNIT_PRICE]<self.opp_pre_offer[negotiator_id][UNIT_PRICE] else 0)
            else:
                self.ppo.buffer[negotiator_id].rewards.append(self.reward_for_opp_concession if offer[UNIT_PRICE]>self.opp_pre_offer[negotiator_id][UNIT_PRICE] else 0)
        else:
            self.ppo.buffer[negotiator_id].rewards.append(self.reward_for_opp_concession)"""

        self.opp_pre_offer[negotiator_id] = offer
        # 状態を記録
        self.obs[negotiator_id][3] = (
            offer[UNIT_PRICE] - ami.issues[UNIT_PRICE].min_value
        ) / (
            ami.issues[UNIT_PRICE].max_value - ami.issues[UNIT_PRICE].min_value
        )  # 直前の相手のオファー価格
        # if self._needed(negotiator_id) - offer[QUANTITY] >= len(self.my_negotiators) - (len(self.contracts.keys())-1) - 1:
        if True:
            # print(negotiator_id, self._needed(negotiator_id), offer[QUANTITY],len(self.my_negotiators) - (len(self.contracts.keys())-1) - 1)
            if self._is_selling(ami):
                if self.obs[negotiator_id][3] >= self.action[negotiator_id]:
                    return ResponseType.ACCEPT_OFFER
            else:
                if self.obs[negotiator_id][3] <= self.action[negotiator_id]:
                    return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    def on_negotiation_end(self, negotiator_id, state):
        negotiator_id = negotiator_id[2:-2]

        if self.ppo_action_flag[negotiator_id]:
            self.ppo.buffer[negotiator_id].is_terminals[-1] = True
        return super().on_negotiation_end(negotiator_id, state)

    def on_negotiation_success(self, contract, mechanism):
        my_id = self.awi.agent.id
        negotiator_id = list(contract.partners).copy()
        negotiator_id.remove(my_id)
        negotiator_id = negotiator_id[0]
        ami = self.get_nmi(negotiator_id)

        negotiator_id = negotiator_id[2:-2]

        # 報酬の算出
        if self._is_selling(ami):
            reward = (
                (contract.agreement["unit_price"] - ami.issues[UNIT_PRICE].min_value)
                / (ami.issues[UNIT_PRICE].max_value - ami.issues[UNIT_PRICE].min_value)
            ) ** 1.0
        else:
            reward = (
                (ami.issues[UNIT_PRICE].max_value - contract.agreement["unit_price"])
                / (ami.issues[UNIT_PRICE].max_value - ami.issues[UNIT_PRICE].min_value)
            ) ** 1.0
            # print('buyer day',self.awi.current_step,reward)
        if self.ppo_action_flag[negotiator_id]:
            self.ppo.buffer[negotiator_id].rewards[-1] = reward

        self.secured += contract.agreement["quantity"]
        self.contracts[negotiator_id] = contract.agreement
        self.n_agreements[negotiator_id] += 1

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        my_id = self.awi.agent.id
        negotiator_id = list(partners).copy()
        negotiator_id.remove(my_id)
        negotiator_id = negotiator_id[0]
        ami = self.get_nmi(negotiator_id)
        my_need = self._needed(negotiator_id)

        negotiator_id = negotiator_id[2:-2]

        if self.ppo_action_flag[negotiator_id]:
            self.ppo.buffer[negotiator_id].rewards[-1] = self.reward_for_neg_failure

    def step(self):
        for negotiator_id in self.my_negotiators:
            negotiator_id = negotiator_id[2:-2]

            if self.ppo_action_flag[negotiator_id]:
                self.opp_step_rewards[negotiator_id].append(
                    self.ppo.buffer[negotiator_id].rewards[-1]
                )

                episode = []
                for i, (state, action, is_terminal) in enumerate(
                    zip(
                        reversed(self.ppo.buffer[negotiator_id].states),
                        reversed(self.ppo.buffer[negotiator_id].actions),
                        reversed(self.ppo.buffer[negotiator_id].is_terminals),
                    )
                ):
                    if i != 0 and is_terminal:
                        break
                    episode.insert(0, list(state[: self.state_dim]) + [action])
                # print(episode)
                if negotiator_id not in self.ppo.episode_buffer.keys():
                    self.ppo.episode_buffer[negotiator_id] = [episode]
                else:
                    self.ppo.episode_buffer[negotiator_id].append(episode)

            # print(torch.FloatTensor(self.ppo.episode_buffer[negotiator_id][0]))

        for negotiator_id in self.my_negotiators:
            if negotiator_id[2:-2] in self.contracts.keys():
                ami = self.get_nmi(negotiator_id)

                negotiator_id = negotiator_id[2:-2]

                PPOTrainAgent.writer.add_scalar(
                    f"agreement_price/{negotiator_id}",
                    (
                        self.contracts[negotiator_id]["unit_price"]
                        - ami.issues[UNIT_PRICE].min_value
                    )
                    / (
                        ami.issues[UNIT_PRICE].max_value
                        - ami.issues[UNIT_PRICE].min_value
                    ),
                    self.awi.n_steps * PPOTrainAgent._count + self.awi.current_step,
                )

        # 最終日の終わりに，PPOモデルを保存, シミュレーションの総報酬をログに記録
        if self.awi.current_step == self.awi.n_steps - 1:
            # 1シミュレーション(50日)の終わりにPPO Update
            losses = {}
            if (
                PPOTrainAgent._count + 1
            ) % 10 == 0 and self.emb_dim != 0:  # and PPOTrainAgent._count<0.5*N_SIMULATIONS: #10シミュレーションに1回表現関数のUpdate
                # self.ppo.update_rep(coef_im=1.0,coef_id=0.1)

                # IICRの算出
                numerator, denominator = 0, 0
                n_negotiators = len(self.all_embs.keys())
                if n_negotiators != 0:
                    for neg1 in self.all_embs.keys():
                        tmp = 0
                        for emb1 in self.all_embs[neg1]:
                            for emb2 in self.all_embs[neg1]:
                                tmp += np.sqrt(
                                    sum((np.array(emb1) - np.array(emb2)) ** 2)
                                )
                        tmp /= len(self.all_embs[neg1]) * len(self.all_embs[neg1])
                        numerator += tmp
                        for neg2 in self.all_embs.keys():
                            if neg1 == neg2:
                                continue
                            tmp = 0
                            for emb1 in self.all_embs[neg1]:
                                for emb2 in self.all_embs[neg2]:
                                    tmp += np.sqrt(
                                        sum((np.array(emb1) - np.array(emb2)) ** 2)
                                    )
                            tmp /= len(self.all_embs[neg1]) * len(self.all_embs[neg2])
                            denominator += tmp
                iicr = (n_negotiators - 1) * numerator / denominator

                if iicr > 0.4:
                    self.ppo.update_rep(coef_im=1.0, coef_id=0.1)

            if False not in [
                len(self.ppo.buffer[negotiator_id[2:-2]].rewards) > 1
                for negotiator_id in self.my_negotiators
            ]:
                losses = self.ppo.update(coef_adv=1.0, coef_critic=0.5, coef_ent=0.01)

            for i, negotiator_id in enumerate(self.my_negotiators):
                negotiator_id = negotiator_id[2:-2]

                PPOTrainAgent.opp_simulation_total_rewards[negotiator_id].append(
                    sum(self.opp_step_rewards[negotiator_id])
                )
                PPOTrainAgent.writer.add_scalar(
                    f"simulation_reward/{negotiator_id}",
                    sum(self.opp_step_rewards[negotiator_id]),
                    PPOTrainAgent._count,
                )
                PPOTrainAgent.writer.add_scalar(
                    f"n_agreements/{negotiator_id}",
                    self.n_agreements[negotiator_id],
                    PPOTrainAgent._count,
                )
                """if PPOTrainAgent._count%20==0:
                    if negotiator_id in losses.keys():
                        plt.plot(range(len(losses[negotiator_id])),losses[negotiator_id])
                        plt.show()
                    else:
                        pass # print(f'Policy update is not done at the final step of Simulation {PPOTrainAgent._count}.')
                """
            # 100回に一回，1日のアクションと報酬と，1シミュレーションのiicrを表示
            if (PPOTrainAgent._count + 1) % 500 == 0:
                # ある1日のアクションと報酬を表示
                for negotiator_id in self.my_negotiators:
                    negotiator_id = negotiator_id[2:-2]

                    pass  # print(f'Simuration{PPOTrainAgent._count + 1} {negotiator_id}')
                    step_actions, step_rewards = [], []
                    for j, (action, reward, is_terminal) in enumerate(
                        zip(
                            reversed(self.ppo.buffer[negotiator_id].actions),
                            reversed(self.ppo.buffer[negotiator_id].rewards),
                            reversed(self.ppo.buffer[negotiator_id].is_terminals),
                        )
                    ):
                        if j != 0 and is_terminal:
                            break
                        step_actions.insert(0, action)
                        step_rewards.insert(0, reward)
                    pass  # print(f'actions:{step_actions}')
                    pass  # print(f'rewards:{step_rewards}', end='\n\n')
                # IICRの算出
                numerator, denominator = 0, 0
                n_negotiators = len(self.all_embs.keys())
                if n_negotiators != 0:
                    for neg1 in self.all_embs.keys():
                        tmp = 0
                        for emb1 in self.all_embs[neg1]:
                            for emb2 in self.all_embs[neg1]:
                                tmp += np.sqrt(
                                    sum((np.array(emb1) - np.array(emb2)) ** 2)
                                )
                        tmp /= len(self.all_embs[neg1]) * len(self.all_embs[neg1])
                        numerator += tmp
                        for neg2 in self.all_embs.keys():
                            if neg1 == neg2:
                                continue
                            tmp = 0
                            for emb1 in self.all_embs[neg1]:
                                for emb2 in self.all_embs[neg2]:
                                    tmp += np.sqrt(
                                        sum((np.array(emb1) - np.array(emb2)) ** 2)
                                    )
                            tmp /= len(self.all_embs[neg1]) * len(self.all_embs[neg2])
                            denominator += tmp
                iicr = (n_negotiators - 1) * numerator / denominator
                pass  # print(f'Simuration{PPOTrainAgent._count + 1} iicr={iicr}', end='\n\n')

            self.ppo.save(self.checkpoint_path)

            if PPOTrainAgent._count + 1 in N_SIMULATIONS_SHORTS:
                if not os.path.exists(
                    self.checkpoint_path + f"/{PPOTrainAgent._count + 1}"
                ):
                    os.makedirs(self.checkpoint_path + f"/{PPOTrainAgent._count + 1}")
                self.ppo.save(self.checkpoint_path + f"/{PPOTrainAgent._count + 1}")

            # 最後のシミュレーションの終わりに，ログファイルを閉じる
            if PPOTrainAgent._count + 1 == N_SIMULATIONS:
                PPOTrainAgent.writer.close()  # runner側でcloseする
                """for negotiator_id in self.my_negotiators:
                    pass # print(pd.DataFrame(vars(self.ppo.buffer[negotiator_id])))

                for negotiator_id in self.my_negotiators:
                    fig,ax=plt.subplots()
                    ax.set_title(f'opp_step_rewards with {negotiator_id}')
                    pass # print(negotiator_id,self.opp_step_rewards[negotiator_id])
                    ax.plot(range(len(self.opp_step_rewards[negotiator_id])),self.opp_step_rewards[negotiator_id])
                """

                """fig,ax=plt.subplots()
                for negotiator_id in self.my_negotiators:
                    negotiator_id = negotiator_id[2:-2]
                    #ax.set_title(f'opp_siuration_total_rewards with {negotiator_id}')
                    ax.set_xlabel('simulation')
                    ax.set_xlim([-0.5,N_SIMULATIONS+0.5])
                    ax.set_ylabel('total reward')
                    ax.set_ylim([-25.1,50.1])
                    ax.plot(range(len(PPOTrainAgent.opp_simulation_total_rewards[negotiator_id])),PPOTrainAgent.opp_simulation_total_rewards[negotiator_id],'-',lw=0.4,label=negotiator_id)
                ax.legend(loc='upper left')
                #fig.savefig(f"{''.join(self.my_negotiators)}.png")
                fig.savefig(f"{PPOTrainAgent.dt_now}-1.png")

                fig,ax=plt.subplots()
                for negotiator_id in self.my_negotiators:
                    negotiator_id = negotiator_id[2:-2]
                    #ax.set_title(f'opp_siuration_total_rewards with {negotiator_id}')
                    ax.set_xlabel('simulation')
                    ax.set_xlim([-0.5,N_SIMULATIONS+0.5])
                    ax.set_ylabel('total reward')
                    ax.set_ylim([-25.1,50.1])
                    ax.plot(range(len(PPOTrainAgent.opp_simulation_total_rewards[negotiator_id])),PPOTrainAgent.opp_simulation_total_rewards[negotiator_id],'--',lw=0.4,label=negotiator_id)
                ax.legend(loc='upper left')
                #fig.savefig(f"{''.join(self.my_negotiators)}.png")
                fig.savefig(f"{PPOTrainAgent.dt_now}-2.png")

                fig,ax=plt.subplots()
                for negotiator_id in self.my_negotiators:
                    negotiator_id = negotiator_id[2:-2]
                    #ax.set_title(f'opp_siuration_total_rewards with {negotiator_id}')
                    ax.set_xlabel('simulation')
                    ax.set_xlim([-0.5,N_SIMULATIONS+0.5])
                    ax.set_ylabel('total reward')
                    ax.set_ylim([-25.1,50.1])
                    ax.plot(range(len(PPOTrainAgent.opp_simulation_total_rewards[negotiator_id])),PPOTrainAgent.opp_simulation_total_rewards[negotiator_id],':',lw=0.4,label=negotiator_id)
                ax.legend(loc='upper left')
                #fig.savefig(f"{''.join(self.my_negotiators)}.png")
                fig.savefig(f"{PPOTrainAgent.dt_now}-3.png")"""

                for negotiator_id in self.my_negotiators:
                    fig, ax = plt.subplots()
                    negotiator_id = negotiator_id[2:-2]
                    # ax.set_title(f'opp_siuration_total_rewards with {negotiator_id}')
                    ax.set_xlabel("simulation", fontsize=20)
                    ax.set_xlim([-0.5, N_SIMULATIONS + 0.5])
                    ax.set_ylabel("total reward", fontsize=20)
                    ax.set_ylim([-25.1, 50.1])
                    ax.plot(
                        range(
                            len(
                                PPOTrainAgent.opp_simulation_total_rewards[
                                    negotiator_id
                                ]
                            )
                        ),
                        PPOTrainAgent.opp_simulation_total_rewards[negotiator_id],
                        color="tab:blue",
                        alpha=0.3,
                        lw=0.7,
                    )

                    smoothed_rewards = [
                        PPOTrainAgent.opp_simulation_total_rewards[negotiator_id][0]
                    ]
                    smoothing_rate = 0.95
                    for total_reward in PPOTrainAgent.opp_simulation_total_rewards[
                        negotiator_id
                    ][1:]:
                        smoothed_rewards.append(
                            smoothed_rewards[-1] * 0.9 + total_reward * 0.1
                        )
                    ax.plot(
                        range(len(smoothed_rewards)), smoothed_rewards, color="tab:blue"
                    )

                    ax.tick_params(labelsize=20)
                    fig.tight_layout()
                    # fig.savefig(f"{''.join(self.my_negotiators)}.png")
                    fig.savefig(
                        self.checkpoint_path + f"/totalreward-{negotiator_id}.png"
                    )
                plt.show()
                pass


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
class PPOTestAgent(OneShotAgent):
    # クラス変数
    _count = -1
    checkpoint_directory = None
    _emb_dim = 32
    AGREE_RATES, AGREE_UTILS = {}, {}
    # AGREE_RATES,AGREE_UTILS=defaultdict(list),defaultdict(list)
    IICR = defaultdict(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        PPOTestAgent._count += 1
        if PPOTestAgent._count == 0:
            from tkinter import filedialog

            PPOTestAgent.checkpoint_directory = filedialog.askdirectory()
            pass  # print(PPOTestAgent.checkpoint_directory)

    # シミュレーションの開始時(awiのセットアップ完了後)
    def init(self):
        super().init()

        self.state_dim, self.action_dim = 4, 1
        self.lr_actor, self.lr_critic = 3e-5, 1e-4  # dafault:0.0003(3e-4),0.001(1e-3)

        self.gamma = 0.99  # default:0.99
        self.K_epochs = 200  # dafault:100
        self.eps_clip = 0.2
        self.reward_for_neg_failure = -0.5
        self.reward_for_opp_concession = 0.01

        # 交渉相手リストの作成，原材料と完成品の量を(無理矢理)揃える(地震と同じ層に他の工場はない設定)
        if self.awi.my_suppliers[0] == "SELLER":
            self.my_negotiators = self.awi.my_consumers
        else:
            self.my_negotiators = self.awi.my_suppliers

        if PPOTestAgent._count == 0:
            PPOTestAgent._emb_dim = int(input("The Dimension of Embbeding Vector:"))

        self.emb_dim = PPOTestAgent._emb_dim
        self.lr_emb = 1e-6
        self.ppo = PPO_Emb(
            self.state_dim,
            self.action_dim,
            self.emb_dim,
            self.lr_actor,
            self.lr_critic,
            self.lr_emb,
            self.gamma,
            self.K_epochs,
            self.eps_clip,
            True,
            self.my_negotiators,
        )
        self.ppo_action_flag = {}  # self.ppo.select_actionを呼び出したか否かk
        self.embs = {}
        self.all_embs = {}
        self.agreement_utils = {}
        self.n_agreements = {}
        self.n_exogenous_contracts = {}

        for i, negotiator_id in enumerate(self.my_negotiators):
            self.ppo.buffer[negotiator_id].clear()
            self.all_embs[negotiator_id] = []
            self.agreement_utils[negotiator_id] = []
            self.embs[negotiator_id] = list(np.random.rand(self.emb_dim))
            self.n_agreements[negotiator_id] = 0
            self.n_exogenous_contracts[negotiator_id] = 0

        self.ppo.load(PPOTestAgent.checkpoint_directory)

    # １日の始まり
    def before_step(self):
        self.secured = 0
        self.obs = {}
        self.action = {}
        self.contracts = {}
        self.opp_pre_offer = {}

        if self.awi.my_suppliers[0] == "SELLER":
            # print(self.awi.current_step,self.awi.agent.id,'current_exogenous_input_quantity',self.awi.current_exogenous_input_quantity)
            for negotiator_id in self.my_negotiators:
                self.obs[negotiator_id] = [
                    self.awi.current_step / (self.awi.n_steps - 1),
                    0,
                    1,
                    0,
                ]  # 日,交渉ラウンド,前回の自分のオファーの単価,直前の相手のオファーの単価
            self.contracts["EXOGENOUS"] = {
                "unit_price": self.awi.current_exogenous_input_price
                / self.awi.current_exogenous_input_quantity
                if self.awi.current_exogenous_input_quantity != 0
                else -1,
                "quantity": self.awi.current_exogenous_input_quantity,
            }
        else:
            # print(self.awi.current_step,self.awi.agent.id,'current_exogenous_output_quantity',self.awi.current_exogenous_output_quantity)
            for negotiator_id in self.my_negotiators:
                self.obs[negotiator_id] = [
                    self.awi.current_step / (self.awi.n_steps - 1),
                    0,
                    0,
                    1,
                ]
            self.contracts["EXOGENOUS"] = {
                "unit_price": self.awi.current_exogenous_output_price
                / self.awi.current_exogenous_output_quantity
                if self.awi.current_exogenous_output_quantity != 0
                else -1,
                "quantity": self.awi.current_exogenous_output_quantity,
            }

        for negotiator_id in self.my_negotiators:
            self.ppo_action_flag[negotiator_id] = False  # その日にselect_actionを行ったらTrueにする
            if (
                negotiator_id in self.ppo.episode_buffer.keys()
                and len(self.ppo.episode_buffer[negotiator_id]) > 0
            ):
                self.embs[negotiator_id] = self.ppo.get_emb(
                    self.ppo.episode_buffer[negotiator_id][-1]
                )
                self.all_embs[negotiator_id].append(self.embs[negotiator_id])

        self.first_proposer = {}
        for negotiator_id in self.my_negotiators:
            self.first_proposer[negotiator_id] = None

            if self.awi.is_bankrupt(negotiator_id):
                pass  # print(f'day{self.awi.current_step} {negotiator_id} is bunkrupt.')

    def propose(self, negotiator_id: str, state) -> "Outcome":
        ami = self.get_nmi(negotiator_id)
        my_needs = self._needed(negotiator_id)
        # 試験的に量に関する条件を撤廃
        if not ami or my_needs <= 0 or self.awi.is_bankrupt(negotiator_id):
            return None

        if self.first_proposer[negotiator_id] == None:  # 自分が先手で提案する日の第一ラウンド
            self.first_proposer[negotiator_id] = 1
            self.action[negotiator_id] = self.ppo.select_action(
                self.obs[negotiator_id] + self.embs[negotiator_id], negotiator_id
            )
            self.ppo.buffer[negotiator_id].rewards.append(
                self.reward_for_opp_concession
            )
            self.ppo.buffer[negotiator_id].is_terminals.append(False)
            self.ppo_action_flag[negotiator_id] = True

        offer = [-1, -1, -1]
        offer[TIME] = self.awi.current_step
        offer[QUANTITY] = my_needs
        if negotiator_id in self.opp_pre_offer.keys():
            offer[QUANTITY] = min(my_needs, self.opp_pre_offer[negotiator_id][QUANTITY])
        offer[UNIT_PRICE] = (
            self.action[negotiator_id]
            * (ami.issues[UNIT_PRICE].max_value - ami.issues[UNIT_PRICE].min_value)
            + ami.issues[UNIT_PRICE].min_value
        )
        # 状態を記録
        self.obs[negotiator_id][2] = self.action[negotiator_id]  # 直前の自分のオファー価格

        return offer

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        if self.first_proposer[negotiator_id] == None:  # 自分が後手で提案する日の第一ラウンド
            self.first_proposer[negotiator_id] = 0

        ami = self.get_nmi(negotiator_id)
        self.obs[negotiator_id][1] = state.step / (ami.n_steps - 1)  # 交渉ラウンド
        self.action[negotiator_id] = self.ppo.select_action(
            self.obs[negotiator_id] + self.embs[negotiator_id], negotiator_id
        )
        self.ppo.buffer[negotiator_id].rewards.append(self.reward_for_opp_concession)
        self.ppo.buffer[negotiator_id].is_terminals.append(False)
        self.ppo_action_flag[negotiator_id] = True

        self.opp_pre_offer[negotiator_id] = offer
        # 状態を記録
        self.obs[negotiator_id][3] = (
            offer[UNIT_PRICE] - ami.issues[UNIT_PRICE].min_value
        ) / (
            ami.issues[UNIT_PRICE].max_value - ami.issues[UNIT_PRICE].min_value
        )  # 直前の相手のオファー価格
        # 試験的に量に関する条件を撤廃
        if offer[QUANTITY] <= self._needed(negotiator_id):
            # if True:
            if self._is_selling(ami):
                if self.obs[negotiator_id][3] >= self.action[negotiator_id]:
                    return ResponseType.ACCEPT_OFFER
            else:
                if self.obs[negotiator_id][3] <= self.action[negotiator_id]:
                    return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    def on_negotiation_end(self, negotiator_id, state):
        if self.ppo_action_flag[negotiator_id]:
            self.ppo.buffer[negotiator_id].is_terminals[-1] = True

        # print(f'\n{self.awi.current_step} {negotiator_id} end')

        return super().on_negotiation_end(negotiator_id, state)

    def on_negotiation_success(self, contract, mechanism):
        my_id = self.awi.agent.id
        negotiator_id = list(contract.partners).copy()
        negotiator_id.remove(my_id)
        negotiator_id = negotiator_id[0]
        ami = self.get_nmi(negotiator_id)
        # 報酬の算出
        if self._is_selling(ami):
            reward = (
                (contract.agreement["unit_price"] - ami.issues[UNIT_PRICE].min_value)
                / (ami.issues[UNIT_PRICE].max_value - ami.issues[UNIT_PRICE].min_value)
            ) ** 1
        else:
            reward = (
                (ami.issues[UNIT_PRICE].max_value - contract.agreement["unit_price"])
                / (ami.issues[UNIT_PRICE].max_value - ami.issues[UNIT_PRICE].min_value)
            ) ** 1
        if self.ppo_action_flag[negotiator_id]:
            self.ppo.buffer[negotiator_id].rewards[-1] = reward

        self.secured += contract.agreement["quantity"]
        self.contracts[negotiator_id] = contract.agreement
        self.n_agreements[negotiator_id] += 1

        self.agreement_utils[negotiator_id].append(reward)

        # print('success')

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        my_id = self.awi.agent.id
        negotiator_id = list(partners).copy()
        negotiator_id.remove(my_id)
        negotiator_id = negotiator_id[0]
        ami = self.get_nmi(negotiator_id)
        my_need = self._needed(negotiator_id)

        if self.ppo_action_flag[negotiator_id]:
            self.ppo.buffer[negotiator_id].rewards[-1] = self.reward_for_neg_failure
            self.agreement_utils[negotiator_id].append(self.reward_for_neg_failure)
        else:
            self.agreement_utils[negotiator_id].append(None)

        # print('failure')

    def step(self):
        for negotiator_id in self.my_negotiators:
            if self.ppo_action_flag[negotiator_id]:
                episode = []
                for i, (state, action, is_terminal) in enumerate(
                    zip(
                        reversed(self.ppo.buffer[negotiator_id].states),
                        reversed(self.ppo.buffer[negotiator_id].actions),
                        reversed(self.ppo.buffer[negotiator_id].is_terminals),
                    )
                ):
                    if i != 0 and is_terminal:
                        break
                    episode.insert(0, list(state[: self.state_dim]) + [action])
                if negotiator_id not in self.ppo.episode_buffer.keys():
                    self.ppo.episode_buffer[negotiator_id] = [episode]
                else:
                    self.ppo.episode_buffer[negotiator_id].append(episode)

        if self.awi.current_step == self.awi.n_steps - 1:
            # 交渉相手の戦略(種類)のリスト(重複は削除)
            my_negotiator_types = list(
                set([negotiator_id[2:-2] for negotiator_id in self.my_negotiators])
            )

            # 交渉相手ごと，相手の戦略(種類)ごとに合意率と合意効用を算出し，クラス変数AGREE_UTLSとAGREE_RATESに追加
            agent_agree_rate = {
                k: 1 - v.count(self.reward_for_neg_failure) / (len(v) - v.count(None))
                for k, v in self.agreement_utils.items()
            }
            agent_util_mean = {
                k: 0
                if len([v_suc for v_suc in v if v_suc and v_suc >= 0]) == 0
                else sum([v_suc for v_suc in v if v_suc and v_suc >= 0])
                / len([v_suc for v_suc in v if v_suc and v_suc >= 0])
                for k, v in self.agreement_utils.items()
            }
            # print('agent agree rate:',agent_agree_rate)
            # print('agent average utility:',agent_util_mean)
            type_agree_rate = {type: [] for type in my_negotiator_types}
            type_util_mean = {type: [] for type in my_negotiator_types}
            for negotiator_id in self.my_negotiators:
                type_agree_rate[negotiator_id[2:-2]].append(
                    agent_agree_rate[negotiator_id]
                )
                type_util_mean[negotiator_id[2:-2]].append(
                    agent_util_mean[negotiator_id]
                )
            type_agree_rate = {
                k: (0 if len(v) == 0 else sum(v) / len(v))
                for k, v in type_agree_rate.items()
            }
            type_util_mean = {
                k: (0 if len(v) == 0 else sum(v) / len(v))
                for k, v in type_util_mean.items()
            }
            # print('type agree rate:',type_agree_rate)
            # print('type average utility:',type_util_mean)

            for agent_type in my_negotiator_types:
                """PPOTestAgent.AGREE_UTILS[agent_type].append(type_util_mean[agent_type])
                PPOTestAgent.AGREE_RATES[agent_type].append(type_agree_rate[agent_type])
                """
                if agent_type in PPOTestAgent.AGREE_UTILS.keys():
                    PPOTestAgent.AGREE_UTILS[agent_type].append(
                        type_util_mean[agent_type]
                    )
                else:
                    PPOTestAgent.AGREE_UTILS[agent_type] = [type_util_mean[agent_type]]
                if agent_type in PPOTestAgent.AGREE_RATES.keys():
                    PPOTestAgent.AGREE_RATES[agent_type].append(
                        type_agree_rate[agent_type]
                    )
                else:
                    PPOTestAgent.AGREE_RATES[agent_type] = [type_agree_rate[agent_type]]
            # 表現関数なしの場合はIICRの算出は行わない
            if self.emb_dim == 0:
                return

            # IICRの算出(相手ごと)
            """numerator,denominator=0,0
            n_negotiators=len(self.all_embs.keys())
            if n_negotiators!=0:
                for neg1 in self.all_embs.keys():
                    tmp=0
                    for emb1 in self.all_embs[neg1]:
                        for emb2 in self.all_embs[neg1]:
                            tmp+=np.sqrt(sum((np.array(emb1)-np.array(emb2))**2))
                    tmp/=len(self.all_embs[neg1])*len(self.all_embs[neg1])
                    numerator+=tmp
                    for neg2 in self.all_embs.keys():
                        if neg1==neg2:
                            continue
                        tmp=0
                        for emb1 in self.all_embs[neg1]:
                            for emb2 in self.all_embs[neg2]:
                                tmp+=np.sqrt(sum((np.array(emb1)-np.array(emb2))**2))
                        tmp/=len(self.all_embs[neg1])*len(self.all_embs[neg2])
                        denominator+=tmp
            iicr=(n_negotiators-1)*numerator/denominator"""

            # IICRの算出(相手の種類(戦略)ごと)
            numerator, denominator = 0, 0
            n_types = len(my_negotiator_types)
            type_all_embs = {agent_type: [] for agent_type in my_negotiator_types}

            for negotiator_id in self.my_negotiators:
                type_all_embs[negotiator_id[2:-2]] += self.all_embs[negotiator_id]
            if n_types >= 2:
                for type1 in my_negotiator_types:
                    tmp = 0
                    for emb1 in type_all_embs[type1]:
                        for emb2 in type_all_embs[type1]:
                            tmp += np.sqrt(sum((np.array(emb1) - np.array(emb2)) ** 2))
                    tmp /= len(type_all_embs[type1]) * len(type_all_embs[type1])

                    numerator += tmp
                    for type2 in my_negotiator_types:
                        if type1 == type2:
                            continue
                        tmp = 0
                        for emb1 in type_all_embs[type1]:
                            for emb2 in type_all_embs[type2]:
                                tmp += np.sqrt(
                                    sum((np.array(emb1) - np.array(emb2)) ** 2)
                                )
                        tmp /= len(type_all_embs[type1]) * len(type_all_embs[type2])
                        denominator += tmp
                iicr = (n_types - 1) * numerator / denominator
                # PPOTestAgent.IICR['-'.join([negotiator_id[2:-2] for negotiator_id in self.my_negotiators])].append(iicr)
                if (
                    "-".join(
                        [negotiator_id[2:-2] for negotiator_id in self.my_negotiators]
                    )
                    in PPOTestAgent.IICR.keys()
                ):
                    PPOTestAgent.IICR[
                        "-".join(
                            [
                                negotiator_id[2:-2]
                                for negotiator_id in self.my_negotiators
                            ]
                        )
                    ].append(iicr)
                else:
                    PPOTestAgent.IICR[
                        "-".join(
                            [
                                negotiator_id[2:-2]
                                for negotiator_id in self.my_negotiators
                            ]
                        )
                    ] = [iicr]

                # print('iicr: ',iicr)

            # PCAによる２次元圧縮をして可視化
            X = []
            """for negotiator_id in self.my_negotiators:
                X+=self.all_embs[negotiator_id]"""
            for agent_type in my_negotiator_types:
                X += type_all_embs[agent_type]

            pca = PCA(random_state=0)
            feature = pca.fit_transform(np.array(X))
            # feature=TSNE(random_state=0).fit_transform(np.array(X))
            # print(f'Explained Varience Ratio: first:{pca.explained_variance_ratio_[0]:.3f} second:{pca.explained_variance_ratio_[1]:.3f}')
            fig, ax = plt.subplots()
            ax.set_xlabel(
                f"First Principal Component ({pca.explained_variance_ratio_[0]:.3f})",
                fontsize=20,
            )
            ax.set_ylabel(
                f"Second Principal Component ({pca.explained_variance_ratio_[1]:.3f})",
                fontsize=20,
            )
            ax.tick_params(labelsize=20)
            pos = 0
            """for i,negotiator_id in enumerate(self.my_negotiators):
                ax.scatter(feature[pos:pos+len(self.all_embs[negotiator_id]),0],feature[pos:pos+len(self.all_embs[negotiator_id]),1],label=negotiator_id,edgecolors=['gray' if util==None else ('k' if util<0 else 'w') for util in self.agreement_utils[negotiator_id][-len(self.all_embs[negotiator_id]):]])
                pos+=len(self.all_embs[negotiator_id])
                #print(len(self.all_embs[negotiator_id]),len(self.agreement_utils[negotiator_id][-len(self.all_embs[negotiator_id]):]))
            """
            for i, agent_type in enumerate(my_negotiator_types):
                ax.scatter(
                    feature[pos : pos + len(type_all_embs[agent_type]), 0],
                    feature[pos : pos + len(type_all_embs[agent_type]), 1],
                    label=agent_type,
                )
                pos += len(type_all_embs[agent_type])
                pass  # print(len(type_all_embs[agent_type]), pos)
                # print(len(self.all_embs[negotiator_id]),len(self.agreement_utils[negotiator_id][-len(self.all_embs[negotiator_id]):]))
            """
            colors=['r','g','b','c','m','y','k']
            for i,negotiator_id in enumerate(self.my_negotiators):
                for j in range(len(self.all_embs[negotiator_id])):
                    start=len(self.agreement_utils[negotiator_id])-len(self.all_embs[negotiator_id])
                    if self.agreement_utils[negotiator_id][j]==None:
                        #ax.scatter(feature[pos+j,0],feature[pos+j,1],label=negotiator_id,s=10.0,marker='.')
                        pass
                    elif self.agreement_utils[negotiator_id][j]<0:
                        ax.scatter(feature[pos+j,0],feature[pos+j,1],label=negotiator_id,marker='x',c=colors[i])
                    else:
                        ax.scatter(feature[pos+j,0],feature[pos+j,1],label=negotiator_id,marker=f'${self.agreement_utils[negotiator_id][j]:.2f}$',c=colors[i])
                pos+=len(self.all_embs[negotiator_id])

                #print(len(self.all_embs[negotiator_id]),len(self.agreement_utils[negotiator_id][-len(self.all_embs[negotiator_id]):]))
            """
            ax.legend()
            fig.tight_layout()
            plt.show()
