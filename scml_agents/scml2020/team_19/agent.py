"""
**Submitted to ANAC 2020 SCML**
*Authors* type-your-team-member-names-with-their-emails here


This code is free to use or update given that proper attribution is given to 
the authors and the ANAC 2020 SCML. 

This module implements a factory manager for the SCM 2020 league of ANAC 2019 
competition. This version will use subcomponents. Please refer to the 
[game description](http://www.yasserm.com/scml/scml2020.pdf) for all the 
callbacks and subcomponents available.

Your agent can learn about the state of the world and itself by accessing 
properties in the AWI it has. For example:

- The number of simulation steps (days): self.awi.n_steps  
- The current step (day): self.awi.current_steps
- The factory state: self.awi.state
- Availability for producton: self.awi.available_for_production


Your agent can act in the world by calling methods in the AWI it has. 
For example:

- *self.awi.request_negotiation(...)*  # requests a negotiation with one partner
- *self.awi.request_negotiations(...)* # requests a set of negotiations

 
You can access the full list of these capabilities on the documentation.

- For properties/methods available only to SCM agents, check the list 
  [here](http://www.yasserm.com/scml/scml2020docs/api/scml.scml2020.AWI.html)

"""

# required for development
from scml.scml2020.agents import DoNothingAgent

# required for running the test tournament
import time
from tabulate import tabulate
from scml.scml2020.utils import anac2020_std, anac2020_collusion
from scml.scml2020.agents import DecentralizingAgent, BuyCheapSellExpensiveAgent
from negmas.helpers import humanize_time

# required for typing
from typing import List, Optional, Dict, Any
import numpy as np
from negmas import (
    Issue,
    AgentMechanismInterface,
    Contract,
    Negotiator,
    MechanismState,
    Breach,
)
from scml.scml2020 import Failure
from scml.scml2020 import SCML2020Agent
from scml.scml2020 import PredictionBasedTradingStrategy
from scml.scml2020 import MovingRangeNegotiationManager
from scml.scml2020 import TradeDrivenProductionStrategy

from typing import Tuple
from negmas import LinearUtilityFunction

# my need
from scml.scml2020 import *
from negmas import *
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import seaborn as sns
import math

# my module
from .components.production import MyProductor  # 提出時は.components.productionにする
from .components.negotiation import NewNegotiationManager, MyNegotiationManager
from .components.trading import MyTrader

__all__ = ["Ashgent"]


class NewAshgent(  # 性能出なかったので没
    MyProductor, NewNegotiationManager, MyTrader, SCML2020Agent
):
    """
    This is the only class you *need* to implement. You can create the agent
    by combining the following strategies:
    
    1. A trading strategy that decides the quantities to sell and buy
    2. A negotiation manager that decides which negotiations to engage in and 
       uses some controller(s) to control the behavior of all negotiators
    3. A production strategy that decides what to produce

    """

    def step(self):
        super().step()
        # print(self.awi.reports_of_agent(self.id))  # エージェントのIDからFinancialReportを取得
        # if self.awi.current_step > 10:  # 0はダメ
        #     print(self.awi.reports_at_step(self.awi.current_step - 1))  # ステップ番号からFinancialReportを取得, 指定できるステップ以前に発行された最新のFinancialReportが取得できると思われる, current_step以前のステップを指定
        #     # print(self.awi.reports_at_step(4))  # Noneが返ってくる謎(古いステップのFinantialReportは5ステップずつしか残ってない？)

    def target_quantity(
        self, step: int, sell: bool
    ) -> int:  # MovingRangeNegotiationManagerでは不要
        # """A fixed target quantity of half my production capacity"""
        # return self.awi.n_lines // 2

        ## 改良 ##
        return self.awi.n_lines

        # ## 元 ##
        # if sell:
        #     needed, secured = self.outputs_needed, self.outputs_secured
        # else:
        #     needed, secured = self.inputs_needed, self.inputs_secured

        # return needed[step - 1] - secured[step - 1]  # stepが1から始まるから-1する必要あり．元のやつバグってるわ

    def target_quantities(
        self, steps: Tuple[int, int], sell: bool
    ) -> np.ndarray:  # これがあればtarget_quantityは使わないっぽい
        """Implemented for speed but not really required"""

        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return (
            needed[steps[0] - 1 : steps[1] - 1] - secured[steps[0] - 1 : steps[1] - 1]
        )

    def acceptable_unit_price(
        self, step: int, sell: bool
    ) -> int:  # MovingRangeNegotiationManagerでは不要
        # """The catalog price seems OK"""
        # return self.awi.catalog_prices[self.awi.my_output_product] if sell else self.awi.catalog_prices[self.awi.my_input_product]

        ## 元 ##
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return (
                production_cost + self.input_cost[step]
            )  # そのステップにおける仕入れの予測値と，生産コストより良ければ売る（いくらで仕入れたかは考慮してない？）
        return (
            self.output_price[step] - production_cost
        )  # そのステップにおける売却予測値から，生産コストを差し引いてそれより良ければ買う（現ステップでいくらで取引されてるかは考慮してない？）

    # def create_ufun(self, is_seller: bool, issues=None, outcomes=None):  # IndependentNegotiationsManagerはContorollerを使わないため，ufun関数の定義がここで必要
    #     """A utility function that penalizes high cost and late delivery for buying and and awards them for selling"""
    #     if is_seller:
    #         return LinearUtilityFunction((0, 0.25, 1))
    #     return LinearUtilityFunction((0, -0.5, -0.8))


class Ashgent(MyProductor, MyNegotiationManager, MyTrader, SCML2020Agent):
    def step(self):
        super().step()

    def target_quantity(self, step: int, sell: bool) -> int:
        # if self.awi.current_step < self.awi.n_steps * 0.2:
        #     return math.floor(self.awi.n_lines * 1.5)
        # elif self.awi.current_step < self.awi.n_steps * 0.9:
        #     return self.awi.n_lines
        # else:
        #     return self.awi.n_lines // 3
        return self.awi.n_lines

    def target_quantities(self, steps: Tuple[int, int], sell: bool) -> np.ndarray:
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return (
            needed[steps[0] - 1 : steps[1] - 1] - secured[steps[0] - 1 : steps[1] - 1]
        )

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        production_cost = self.awi.profile.costs[0, self.awi.my_input_product]
        if sell:
            return (
                production_cost + self.input_cost[step]
            )  # そのステップにおける仕入れの予測値と，生産コストより良ければ売る（いくらで仕入れたかは考慮してない？）
        return (
            self.output_price[step] - production_cost
        )  # そのステップにおける売却予測値から，生産コストを差し引いてそれより良ければ買う（現ステップでいくらで取引されてるかは考慮してない？）


########## for test ########################################
from collections import defaultdict


def show_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
    plt.show()


from scml.scml2020 import is_system_agent


def analyze_unit_price(world, agent_type):
    """Returns the average price relative to the negotiation issues"""
    contracts = pd.DataFrame(world.saved_contracts)
    fields = ["seller_type", "buyer_type", "unit_price", "issues", "selling", "buying"]
    # Add fields indicating whether the agent_type is a seller or a buyer
    contracts["selling"] = contracts.seller_type == agent_type
    contracts["buying"] = contracts.buyer_type == agent_type
    # keep only contracts in which agent_type is participating
    contracts = contracts.loc[contracts.selling | contracts.buying, fields]
    # remove all exogenous contracts
    contracts = contracts.loc[contracts.issues.apply(len) > 0, fields]
    # find the minimum and maximum unit price in the negotiation issues
    min_vals = contracts.issues.apply(lambda x: x[UNIT_PRICE].min_value)
    max_vals = contracts.issues.apply(lambda x: x[UNIT_PRICE].max_value)
    # replace the unit price with its fraction of the unit-price issue range
    contracts.unit_price = (contracts.unit_price - min_vals) / (max_vals - min_vals)
    contracts = contracts.drop("issues", 1)
    contracts = contracts.rename(columns=dict(unit_price="price"))
    # group results by whether the agent is selling/buying/both
    if len(contracts) < 1:
        return ""
    return contracts.groupby(["selling", "buying"]).describe().round(1)


def test():
    agent_types = [
        NewAshgent,
        Ashgent,
        DecentralizingAgent,
        IndDecentralizingAgent,
        MovingRangeAgent,
    ]
    world = SCML2020World(
        **SCML2020World.generate(agent_types=agent_types, n_steps=50),
        construct_graphs=True,
    )
    world.run_with_progress()

    # print(world.scores().loc[:, ["agent_name", "agent_type", "score"]].head())
    # print(world.scores())

    # world.scores["level"] = world.scores.agent_name.str.split("@", expand=True).loc[:, 1]
    # sns.lineplot(data=world.scores[["agent_type", "level", "score"]],
    #             x="level", y="score", hue="agent_type")
    # plt.plot([0.0] * len(world.scores["level"].unique()), "b--")
    # plt.show()

    # winner = world.winners[0]

    # stats = pd.DataFrame(data=world.stats)
    # bankruptcy = {a: np.nonzero(stats[f"bankrupt_{a}"].values)[0]
    #     for a in world.non_system_agent_names}
    # pprint({k: "No" if len(v)<1 else f"at: {v[0]}" for k, v in bankruptcy.items()})

    # print("MyNewAgent:\n===========")
    # print(analyze_unit_price(world, "Ashgent"))
    # print("\nMyAgent:\n========")
    # print(analyze_unit_price(world, "LegacyAshgent"))
    # print("\nDecentralizingAgent:\n====================")
    # print(analyze_unit_price(world, "DecentralizingAgent"))

    show_agent_scores(world)

    # fig, axs = plt.subplots(2, 2)
    # for ax, key in zip(axs.flatten().tolist(), ["trading_price", "sold_quantity", "unit_price"]):
    #     for p in range(world.n_products):
    #         ax.plot(world.stats[f"{key}_{p}"], marker="x", label=f"Product {p}")
    #         ax.set_ylabel(key.replace("_", " ").title())
    #         ax.legend().set_visible(False)
    # axs[-1, 0].legend(bbox_to_anchor=(1, -.5), ncol=3)
    # fig.show()
    # plt.show()

    # world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    # plt.show()

    # from pathlib import Path
    # print(Path.home() /"negmas" / "logs" / world.name / "log.txt", "r") as f:
    #     [print(_) for _ in f.readlines()[:10]]


def run(
    competition="std",
    reveal_names=True,
    n_steps=50,
    n_configs=2,
    max_n_worlds_per_config=None,
    n_runs_per_world=1,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competition: The competition type to run (possibilities are std, 
                     collusion).        
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try. 
                     Different world configurations will correspond to
                     different number of factories, profiles
                     , production graphs etc
        n_runs_per_world: How many times will each world simulation be run.

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value        

    """
    # competitors = [NewAshgent, Ashgent, DecentralizingAgent, IndDecentralizingAgent, MovingRangeAgent]
    competitors = [
        Ashgent,
        DecentralizingAgent,
        IndDecentralizingAgent,
        MovingRangeAgent,
    ]
    start = time.perf_counter()
    if competition == "std":
        results = anac2020_std(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
        )
    elif competition == "collusion":
        results = anac2020_collusion(
            competitors=competitors,
            verbose=True,
            n_steps=n_steps,
            n_configs=n_configs,
            n_runs_per_world=n_runs_per_world,
        )
    else:
        raise ValueError(f"Unknown competition type {competition}")
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")


if __name__ == "__main__":
    run()
