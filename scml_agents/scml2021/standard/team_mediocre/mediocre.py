import sys

sys.path.append("./")
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from negmas import (
    AgentMechanismInterface,
    Contract,
    Issue,
    MechanismState,
    Negotiator,
    ResponseType,
    SAONegotiator,
    SAOState,
)
from scml.scml2020 import SCML2020Agent
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE
from scml.scml2020.components.negotiation import NegotiationManager
from scml.scml2020.components.production import ProductionStrategy
from scml.scml2020.components.trading import TradingStrategy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from .Bidding import BuyBidding, SellBidding

warnings.filterwarnings("ignore")

MAX_ROUNDS = 20


__all__ = [
    "Mediocre",
]


class Mediocre(SCML2020Agent, ProductionStrategy, TradingStrategy, NegotiationManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_step = None
        self.is_last_level = None
        self.is_first_level = None
        self.prod_cap = None
        self.negotiators = {}
        self.negotiators_engaged = {}
        self.current_neediness = None
        self.supplier_states: Dict[str, Dict] = {}
        self.consumer_states: Dict[str, Dict] = {}
        self.eagerness_for_suppliers: pd.Series = None
        self.eagerness_for_consumers: pd.Series = None
        self.allocations: Dict[str, Dict] = None  # updated by self.do_allocations
        self.identifier = None  # for logging
        self.STOP_PRODUCTION_LAST_N_STEP = None
        self.buy_pmin = (
            None  # minumum price level defined at current step for buy trade
        )

    def init(self):
        """
        Tournament parameters are read and data structures are initiated.
        """
        global MAX_ROUNDS
        super().init()

        MAX_ROUNDS = self.awi.settings["neg_n_steps"]

        self.prod_cap = self.awi.n_lines
        self.is_last_level = self.awi.my_output_product == self.awi.n_products - 1
        self.is_first_level = self.awi.my_input_product == 0
        self.production_cost = np.max(
            self.awi.profile.costs[:, self.awi.my_input_product]
        )
        self.input_catalog_price = self.awi.catalog_prices[self.awi.my_input_product]
        self.output_catalog_price = self.awi.catalog_prices[self.awi.my_output_product]
        self.buy_pmin = self.input_catalog_price

        self.my_suppliers = self.awi.my_suppliers
        self.my_consumers = self.awi.my_consumers

        self.eagerness_for_suppliers = pd.Series(0.5, index=self.my_suppliers)
        self.eagerness_for_consumers = pd.Series(0.5, index=self.my_consumers)

        self.partners = self.my_suppliers + self.my_consumers

        self.trade_stats = pd.DataFrame(
            -1,
            columns=[
                "sign_rate",
                "agree_rate",
                "hist_quantity",
                "hist_utility",
                "last_quantity",
                "last_utility",
            ],
            index=[p for p in self.partners if p not in ["SELLER", "BUYER"]],
        )

        self.past_negotiations = pd.DataFrame(
            columns=[
                "is_sell",
                "step",
                "nego_no",
                "round",
                "delivery_t",
                "quantity",
                "price",
                "utility",
                "norm_utility",
            ]
        )
        self.past_contracts = pd.DataFrame(
            columns=["cid", "partner", "sign_step", "is_sell", "q", "t", "p", "utility"]
        ).set_index(["partner", "cid"])

        # agent identifier for log data
        self.identifier = "Med_" + "_".join(
            [
                str(self.awi.my_input_product),
                str(self.prod_cap),
                str(self.production_cost),
                str(self.input_catalog_price),
                str(self.output_catalog_price),
            ]
        )

        self.STOP_PRODUCTION_LAST_N_STEP = (
            max(1, int(self.awi.n_steps * 0.1)) if not self.is_last_level else 1
        )

    def before_step(self):
        """
        This method defines the behavior of Mediocre agent.
        """

        self.current_step = int(self.awi.current_step)

        self.calculate_neediness_level()

        if self.current_step > 0:
            lst = [
                negotiator.received_bids
                for negotiator_lst in self.negotiators_engaged[
                    self.current_step - 1
                ].values()
                for negotiator in negotiator_lst
            ]
            if lst:
                all_received_bids = pd.concat(lst)
                all_received_bids["step"] = self.current_step - 1
                self.past_negotiations = pd.concat(
                    [self.past_negotiations, all_received_bids]
                )

            self.update_trade_stats()

        self.supplier_states = {
            partner: self.get_state(partner)
            for partner in self.my_suppliers
            if partner != "SELLER"
        }
        self.consumer_states = {
            partner: self.get_state(partner)
            for partner in self.my_consumers
            if partner != "BUYER"
        }

        if not self.is_first_level:
            self.eagerness_for_suppliers = self.calculate_eagerness_rates(
                self.supplier_states
            )

        if not self.is_last_level:
            self.eagerness_for_consumers = self.calculate_eagerness_rates(
                self.consumer_states
            )

        self.allocations = self.do_allocations()

        self.initialize_negotiators()

        self.request_negotiations()

    def step(self):
        if (
            self.awi.current_step < self.awi.n_steps - self.STOP_PRODUCTION_LAST_N_STEP
            and len(self.my_suppliers)
        ):
            self.production_order()

    def update_trade_stats(self):
        """
        Trading statistics with each partner are updated at each step based on contracts and past negotiations.
        """

        for partner in self.partners:
            if (
                partner
                in [
                    "SELLER",
                    "BUYER",
                ]
            ):  # no negotiation with global players, so, trade_stats is not needed for them
                continue

            n_negotiations, n_agree, n_signs = 0, 0, 0

            if partner in self.past_negotiations.index:
                past_negotiations = self.past_negotiations.loc[partner]
                if isinstance(past_negotiations, pd.Series):
                    n_negotiations = 1
                    n_agree = int(past_negotiations["agree"])
                else:
                    n_negotiations = past_negotiations.shape[0]
                    n_agree = past_negotiations["agree"].sum()

            if partner in self.past_contracts.index.get_level_values(0):
                contracts = self.past_contracts.loc[partner]
                n_signs = self.past_contracts.loc[partner].shape[0]
                hist_utility = contracts["utility"].mean()
                hist_quantity = contracts["q"].mean()

                last_contract_step = contracts["sign_step"].max()
                last_utility = contracts[contracts["sign_step"] == last_contract_step][
                    "utility"
                ].mean()
                last_quantity = contracts[contracts["sign_step"] == last_contract_step][
                    "q"
                ].mean()

                self.trade_stats.loc[partner, "hist_quantity"] = hist_quantity
                self.trade_stats.loc[partner, "hist_utility"] = hist_utility
                self.trade_stats.loc[partner, "last_quantity"] = last_quantity
                self.trade_stats.loc[partner, "last_utility"] = last_utility

            self.trade_stats.loc[partner, "sign_rate"] = (
                n_signs / n_agree if n_agree else 0
            )
            self.trade_stats.loc[partner, "agree_rate"] = (
                n_agree / n_negotiations if n_negotiations else 0
            )

    def get_state(self, partner):
        """
        It prodives the state of partner as breach score, last cash information, and wealth score based cash trend.
        """
        state = {}
        reports = self.awi.reports_of_agent(partner)
        balance_trend = None

        if reports:
            reports = list(reports.values())
            report = reports[-1]  # get the last report
            breach_score = (
                1.1 ** (self.current_step - report.step)
                * report.breach_level
                * report.breach_prob
            )
            cash = report.cash

            interval = len(reports)
            if interval > 1:
                n_reports = min(5, interval)
                x = np.arange(n_reports)
                y = np.array([report.cash for report in reports])[-n_reports:]
                balance_trend = np.polyfit(x, y, 1)[0]

        else:  # reports object is None at the beginning, so just imputing, not important
            breach_score = 0
            cash = self.awi.current_balance

        state["breach_score"] = breach_score
        state["cash"] = cash
        state["wealth"] = state["wealth"] = (
            balance_trend / y.mean() if balance_trend != None else "unknown"
        )

        return state

    def calculate_neediness_level(self):
        """
        Defines the needines level (between -1 and 1) of more buying or selling quantity.
        It is negative if needing buy, and positive if needing sell.
        """

        current_inputs = self.awi.state.inventory[self.awi.my_input_product]
        current_outputs = self.awi.state.inventory[self.awi.my_output_product]

        signed_future_inputs = self.past_contracts[
            (self.past_contracts.t >= self.awi.current_step)
            & (self.past_contracts.is_sell == False)
        ]["q"].sum()
        signed_future_outputs = self.past_contracts[
            (self.past_contracts.t >= self.awi.current_step)
            & (self.past_contracts.is_sell)
        ]["q"].sum()

        future_outputs = current_inputs + signed_future_inputs + current_outputs
        neediness = (
            (future_outputs - signed_future_outputs) / future_outputs
            if future_outputs
            else 0
        )

        self.current_neediness = round(
            max(-1, min(1, neediness)), 3
        )  # between -1 and 1

    def calculate_eagerness_rates(self, partner_states):
        """
        This method calculates eagerness to negoatiate rate for each partner according partner's state and trading utility with that partner.
        Args:
            partner_states (Dict):  For each opponent's market state information is found.
        Returns:
            pandas.Series: For each opponent, eagerness rate is found.
        """

        if self.current_step > 0:
            eagerness = pd.DataFrame(index=partner_states.keys())

            eagerness["nego_parameters"] = [
                (
                    self.trade_stats.loc[partner, "hist_utility"],
                    self.trade_stats.loc[partner, "last_utility"],
                    self.trade_stats.loc[partner, "sign_rate"],
                    self.trade_stats.loc[partner, "agree_rate"],
                )
                for partner in eagerness.index
            ]

            eagerness["nego_score"] = eagerness.nego_parameters.apply(
                lambda row: round((row[0] * row[1]) ** 0.5 * row[2] * row[3], 3)
                if -1 not in row
                else 0
            )

            average_score = eagerness.nego_score.mean()
            average_score = average_score if average_score > 0 else 1

            eagerness.nego_score = eagerness.nego_score.apply(
                lambda score: average_score if score == 0 else score
            )

            eagerness["score"] = 0.0
            eagerness["breach_score"] = 0.0
            eagerness["cash"] = 0.0

            for partner, state in partner_states.items():
                eagerness.loc[partner, "breach_score"] = state["breach_score"]
                eagerness.loc[partner, "cash"] = state["cash"]

                partner_score = state["cash"] / (1 + state["breach_score"] ** 0.5)

                partner_score = partner_score if partner_score > 0 else 0
                eagerness.loc[partner, "score"] = (
                    partner_score * eagerness.loc[partner, "nego_score"]
                )

            eagerness_rates = eagerness.score / eagerness.score.sum()
            eagerness_rates.fillna(0.5, inplace=True)  # to avoid any bug
        else:
            eagerness_rates = pd.Series(0.5, index=partner_states.keys())

        return eagerness_rates.round(3)

    def initialize_negotiators(self):
        """
        For each partner, a negotiatior specialized for that partner is created based on issue bounds and bid space.
        """
        self.negotiators[self.current_step] = {}
        self.negotiators_engaged[self.current_step] = {}
        for partner in self.my_suppliers:
            if partner == "SELLER":
                continue
            state = self.supplier_states[partner]
            allocation = self.allocations[partner]
            params = self.get_nego_params_for(partner, state, allocation, to_sell=False)
            negotiator = MediocreNegotiator(
                self.current_step,
                self.id,
                self.identifier,
                partner,
                params,
                to_sell=False,
            )  # buyer negotiator
            self.negotiators[self.current_step][partner] = negotiator
            self.negotiators_engaged[self.current_step][partner] = []

        for partner in self.my_consumers:
            if partner == "BUYER":
                continue
            state = self.consumer_states[partner]
            allocation = self.allocations[partner]
            params = self.get_nego_params_for(partner, state, allocation, to_sell=True)
            negotiator = MediocreNegotiator(
                self.current_step,
                self.id,
                self.identifier,
                partner,
                params,
                to_sell=True,
            )  # seller negotiator
            self.negotiators[self.current_step][partner] = negotiator
            self.negotiators_engaged[self.current_step][partner] = []

    def production_order(self):
        """
        Converts all inputs.
        """
        self.awi.schedule_production(
            process=self.awi.my_input_product,
            repeats=self.prod_cap + 1,
            step=(self.awi.current_step, self.awi.current_step + 1),
            line=-1,
            partial_ok=True,
        )

    def request_negotiations(self):
        for partner, negotiator_ in self.negotiators[self.awi.current_step].items():
            is_sell = True if partner in self.my_consumers else False
            product = (
                self.awi.my_output_product if is_sell else self.awi.my_input_product
            )
            negotiator = deepcopy(negotiator_)

            # do not request with the negotiatiors allocated 0 quantity
            if negotiator.q_bounds[1] > 0:
                negotiator.valid_q_bounds = negotiator.q_bounds
                negotiator.valid_t_bounds = negotiator.t_bounds
                negotiator.valid_p_bounds = negotiator.p_bounds
                negotiator.issues = None
                negotiator.initiator = self.id
                offer_availability = negotiator.init()

                if offer_availability:
                    try:
                        acceptance = self.awi.request_negotiation(
                            is_buy=not is_sell,
                            product=product,
                            quantity=negotiator.q_bounds,
                            unit_price=negotiator.p_bounds,
                            time=negotiator.t_bounds,
                            partner=partner,
                            negotiator=negotiator,
                        )
                    except:
                        acceptance = False
                    if acceptance:
                        negotiator_.nego_no += 1
                        self.negotiators_engaged[self.awi.current_step][partner].append(
                            negotiator
                        )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        if self.awi.current_step not in self.negotiators:
            return None

        qvalues, tvalues, pvalues = issues[QUANTITY], issues[TIME], issues[UNIT_PRICE]
        negotiator_ = self.negotiators[self.awi.current_step][initiator]
        negotiator = deepcopy(negotiator_)

        # do not allow the negotiatiors allocated 0 quantity to engage in negotiators
        if negotiator.q_bounds[1] > 0:
            negotiator.params[
                "alpha"
            ] = -1  # counter agenda might not be suitable for us, therefore all desirable bids are allowed to propose
            negotiator.valid_q_bounds = [qvalues.min_value, qvalues.max_value]
            negotiator.valid_t_bounds = [tvalues.min_value, tvalues.max_value]
            negotiator.valid_p_bounds = [pvalues.min_value, pvalues.max_value]

            negotiator.issues = issues
            negotiator.initiator = initiator
            offer_availability = negotiator.init()

            if offer_availability:
                negotiator_.nego_no += 1
                self.negotiators_engaged[self.awi.current_step][initiator].append(
                    negotiator
                )
                return negotiator

        return None

    def do_allocations(self):
        """
        The method returns resource allocations for each partner.

        For suppliers, min and max quantity of inputs that negotiator that can buy is determined.
        The resource allocated for buy trade is calculated repect to remained production capacity,
        the eagerness rate of the negotiator for that supplier, and the number of step left.

        For customers, max quantity of outputs that negotiator can sell is determined.
        Quantity allocated depends on the output inventory that is not already signed to sell yet.

        Returns:
            Dict: For each partner, resource identifers that a negotiator can use are returned.
        """

        input_inventory = self.awi.state.inventory[self.awi.my_input_product]
        output_inventory = self.awi.state.inventory[self.awi.my_output_product]

        signed_future_outputs = self.past_contracts[
            (self.past_contracts.t >= self.awi.current_step)
            & (self.past_contracts.is_sell)
        ].q.sum()
        signed_future_inputs = self.past_contracts[
            (self.past_contracts.t >= self.awi.current_step)
            & (self.past_contracts.is_sell == False)
        ].q.sum()

        n_steps_to_produce = (
            self.awi.n_steps - self.awi.current_step - self.STOP_PRODUCTION_LAST_N_STEP
        )
        targetQ, remained_production_capacity = 0, 0
        if n_steps_to_produce > 0:
            remained_production_capacity = max(
                0,
                n_steps_to_produce * self.awi.n_lines
                - input_inventory
                - signed_future_inputs,
            )
            targetQ = remained_production_capacity // n_steps_to_produce

        if len(self.eagerness_for_suppliers) > 0:
            suppliers_allocations = pd.DataFrame(
                {
                    "eagerness": self.eagerness_for_suppliers,  # used for logging
                    "qmin": targetQ // len(self.eagerness_for_suppliers),
                    "qmax": (targetQ * 4 * self.eagerness_for_suppliers).astype(int),
                },
                index=self.eagerness_for_suppliers.index,
            ).T.to_dict()
        else:  # all suppliers were bankrupted!
            suppliers_allocations = {}

        outputs_to_sell = max(0, output_inventory - signed_future_outputs)

        customers_allocations = pd.DataFrame(
            {
                "eagerness": self.eagerness_for_consumers,  # used for logging
                "quantity": outputs_to_sell,
                "output_inventory": output_inventory,  # used for logging
                "signed_future_outputs": signed_future_outputs,
            },  # used for logging
            index=self.eagerness_for_consumers.index,
        ).T.to_dict()

        allocations = {**suppliers_allocations, **customers_allocations}

        return allocations

    def observed_trade_price(self, is_sell, partner=None):
        """
        Calculates own trading price as sqrt(weighted historical price * weighted last price) based on contracts.
        """
        if partner == None or partner not in self.past_contracts.index.get_level_values(
            0
        ):
            pcs = self.past_contracts
        else:
            pcs = self.past_contracts.loc[partner]

        catalog_price = (
            self.output_catalog_price if is_sell else self.input_catalog_price
        )

        if pcs[pcs.is_sell == is_sell].shape[0] > 0:
            try:
                last_sign_step = pcs[pcs.is_sell == is_sell].sign_step.max()

                historical_q_p = pcs.loc[pcs.is_sell == is_sell, ["q", "p"]]
                historical_weighted_p = (
                    historical_q_p["q"] * historical_q_p["p"]
                ).sum() / (historical_q_p["q"]).sum()

                last_q_p = pcs.loc[
                    (pcs.is_sell == is_sell) & (pcs.sign_step == last_sign_step),
                    ["q", "p"],
                ]
                last_weighted_p = (last_q_p["q"] * last_q_p["p"]).sum() / (
                    last_q_p["q"]
                ).sum()

                traded_price = int((historical_weighted_p * last_weighted_p) ** 0.5)
            except:  # to surpass any bug
                traded_price = catalog_price

        else:
            traded_price = catalog_price

        return traded_price

    def get_nego_params_for(self, partner, partner_state, allocation, to_sell):
        "This method sets issue bounds based on partner state, self-state, and market state."

        hist_quantity = self.trade_stats.loc[partner, "hist_quantity"]
        last_quantity = self.trade_stats.loc[partner, "last_quantity"]
        last_utility = self.trade_stats.loc[partner, "last_utility"]

        params = {}
        params["eagerness"] = (
            self.eagerness_for_consumers[partner]
            if to_sell
            else self.eagerness_for_suppliers[partner]
        )
        params["neediness"] = self.current_neediness
        params["prod_cap"] = self.prod_cap
        params["input_catalog_price"] = self.input_catalog_price
        params["output_catalog_price"] = self.output_catalog_price
        params["traded_price"] = self.observed_trade_price(to_sell, partner)
        traded_input_price = self.observed_trade_price(is_sell=False)
        traded_output_price = self.observed_trade_price(is_sell=True)

        if to_sell:
            if self.is_first_level:
                pmin = (
                    self.production_cost
                    + self.input_catalog_price
                    + self.output_catalog_price
                ) // 2
            else:
                r = 1 if self.current_neediness > 0.5 else 1.1
                pmin = (
                    r
                    * (
                        self.production_cost
                        + traded_input_price
                        + self.output_catalog_price
                        + self.awi.trading_prices[self.awi.my_output_product]
                    )
                    // 3
                )

            pmax = max(
                (self.output_catalog_price + pmin + traded_output_price) // 2,
                int(1.5 * pmin),
            )

            if self.eagerness_for_consumers[partner] > 0.5:
                pmax = int(pmax * 1.25)

        else:
            if self.current_neediness < -0.05:  # needs to buy
                d = 3.1
            else:
                d = 3.5
            pmin = (
                traded_input_price
                + self.input_catalog_price
                + self.awi.trading_prices[self.awi.my_input_product]
            ) // d
            pmax = min(
                int(traded_output_price - self.production_cost),
                min(
                    int(self.input_catalog_price * 1.25), self.input_catalog_price + 10
                ),
            )  # min(int(self.input_catalog_price*1.25), self.input_catalog_price+10)
            if pmax <= pmin:
                pmax = int(pmin * 1.5)

        impoverishment_indicator = (
            traded_input_price + self.production_cost
        ) / traded_output_price - 1

        if impoverishment_indicator > 0:
            if to_sell:
                pmin += impoverishment_indicator * pmin // 4
                pmax += impoverishment_indicator * pmax // 2
            else:
                pmin = max(1, pmin - int(impoverishment_indicator / 4 * pmin))
                pmax -= int(impoverishment_indicator / 2 * pmax)
                if pmax <= pmin:
                    pmax = int(pmin * 1.5)

        # no negotiaton agreement, maybe due to acceptable price conflict
        if self.current_step / self.awi.n_steps >= 0.25 and last_utility == -1:
            if to_sell:
                pmax = int(pmax * 0.9)
                if pmax <= pmin:
                    pmax = int(pmin * 1.25)
            else:
                pmax = int(pmax * 1.1)

        qmin, qmax = 0, 0
        if to_sell:
            if allocation["quantity"] > 0:
                # qmin = max(1, int(allocation['quantity'] * params['eagerness']))
                # qmax = max(qmin, int(allocation['quantity']))
                qmin = allocation["quantity"] // len(self.my_suppliers)
                qmax = int(allocation["quantity"])

        elif allocation["qmax"] > 0:
            qmin = allocation["qmin"]
            qmax = max(
                allocation["qmin"], allocation["qmax"]
            )  # using max() to avoid any bug

        signed_buying_inputs = (
            self.past_contracts[self.past_contracts.is_sell == False].q.sum()
            if self.past_contracts.shape[0] > 0
            else 0
        )
        signed_selling_outputs = (
            self.past_contracts[self.past_contracts.is_sell].q.sum()
            if self.past_contracts.shape[0] > 0
            else 0
        )

        if self.is_last_level:
            if (
                signed_buying_inputs > signed_selling_outputs
                and self.current_step / self.awi.n_steps > 0.5
            ):
                qmin, qmax = 0, 0

        if to_sell:
            tmin, tmax = (
                int(self.awi.current_step + 1),
                min(int(self.awi.current_step + 3), self.awi.n_steps - 1),
            )
        else:
            tmin, tmax = (
                int(self.awi.current_step + 1),
                min(
                    int(self.awi.current_step + 6),
                    self.awi.n_steps - self.STOP_PRODUCTION_LAST_N_STEP,
                ),
            )
            if tmax < tmin:
                qmin, qmax = 0, 0

            if self.is_last_level:
                tmax = min(
                    int(self.awi.current_step + 10),
                    self.awi.n_steps - self.STOP_PRODUCTION_LAST_N_STEP,
                )

        alpha = -0.5  # default
        beta = 0.5 * (1 + (self.current_step / self.awi.n_steps) ** 0.5)  # default

        wealth = partner_state["wealth"]

        if wealth != "unknown":
            if wealth >= -0.05:  # good
                alpha = -1
            elif wealth >= -0.1:  # not bad
                alpha = -0.75
            elif wealth >= -0.2:  # not too bad
                alpha = -0.6
            elif wealth <= -0.3:  # too bad, maybe irrational partner
                alpha = -1

        if self.trade_stats.loc[partner, "agree_rate"] > 0:
            contract_rate = (
                self.trade_stats.loc[partner, "sign_rate"]
                * self.trade_stats.loc[partner, "agree_rate"]
            ) ** 0.5

            if contract_rate <= 0.125:
                beta = 1
            elif contract_rate <= 0.25:
                beta = 0.75
            elif contract_rate >= 0.75:
                beta = 0.25
        else:
            beta = 1

        if self.is_first_level or self.is_last_level:  # needs more contract_rate
            beta = min(1, beta * 1.5)

        if not to_sell:
            self.buy_pmin = min(pmin, self.buy_pmin)

        params["q_bounds"] = [int(qmin), int(qmax)]
        params["t_bounds"] = [int(tmin), int(tmax)]
        params["p_bounds"] = [int(pmin), int(pmax)]
        params["alpha"], params["beta"] = round(alpha, 3), round(beta, 3)
        params["reasonable_q"] = (
            (hist_quantity * last_quantity) ** 0.5
            if last_quantity != -1
            else (qmin + qmax) // 2
        )

        return params

    def sell_utility_func(self, p, q, t, breach_score=0):
        delivery_delay = t - self.awi.current_step
        if self.is_last_level:
            return q * min(p**1.1, 1.5) * (1 + delivery_delay**0.1)

        return (
            q * min(p**1.1, 1.5) / (1 + delivery_delay**0.1) / (1 + breach_score**0.5)
        )

    def buy_utility_func(self, p, q, t, breach_score=0):
        delivery_delay = t - self.awi.current_step

        return (
            q
            / (max(0, p - self.buy_pmin) + 1)
            / (1 + delivery_delay**0.1)
            / (1 + breach_score**0.5)
        )

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """
        In the method, all contracts are sorted with respect to its utility.
        Then, according to agent's signing constraints as much as possible from the sorted contracts are signed.
        Args:
            contracts (List[Contract]): Contracts that are come from the negotiation processes.
        Returns:
            List[Optional[str]]: List of signatures for contracts
        """

        signatures = [None] * len(contracts)

        input_inventory = self.awi.state.inventory[self.awi.my_input_product]
        output_inventory = self.awi.state.inventory[self.awi.my_output_product]

        buying_contracts = self.past_contracts[
            (self.past_contracts.t >= self.awi.current_step)
            & (self.past_contracts.is_sell == False)
        ]
        signed_future_inputs = buying_contracts["q"].sum()

        to_be_produced = signed_future_inputs + input_inventory
        total_production_capacity = (
            self.awi.n_steps - self.awi.current_step - self.STOP_PRODUCTION_LAST_N_STEP
        ) * self.awi.n_lines - to_be_produced
        total_production_capacity = max(0, total_production_capacity)

        suppliers = pd.DataFrame(
            [
                {
                    "sign_indx": sign_indx,
                    "price": contract.agreement["unit_price"],
                    "quantity": contract.agreement["quantity"],
                    "time": contract.agreement["time"],
                    "breach_score": self.supplier_states[contract.annotation["seller"]][
                        "breach_score"
                    ]
                    if contract.annotation["seller"] != "SELLER"
                    else 0,
                    "partner": contract.annotation["seller"],
                }
                for sign_indx, contract in enumerate(contracts)
                if contract.annotation["seller"] != self.id
            ]
        )

        consumers = pd.DataFrame(
            [
                {
                    "sign_indx": sign_indx,
                    "price": contract.agreement["unit_price"],
                    "quantity": contract.agreement["quantity"],
                    "time": contract.agreement["time"],
                    "breach_score": self.consumer_states[contract.annotation["buyer"]][
                        "breach_score"
                    ]
                    if contract.annotation["buyer"] != "BUYER"
                    else 0,
                    "partner": contract.annotation["buyer"],
                }
                for sign_indx, contract in enumerate(contracts)
                if contract.annotation["seller"] == self.id
            ]
        )

        total_sell = self.past_contracts.loc[self.past_contracts.is_sell, "q"].sum()
        total_buy = self.past_contracts.loc[
            self.past_contracts.is_sell == False, "q"
        ].sum()
        flow_ratio = total_sell / total_buy if total_buy else 0

        if (
            self.awi.n_steps - self.awi.current_step <= self.STOP_PRODUCTION_LAST_N_STEP
            or len(self.my_consumers) == 0
            or total_buy >= total_production_capacity
        ):
            volume_to_buy = 0
        elif (
            self.awi.current_step / self.awi.n_steps > 0.5
            and flow_ratio < 0.5
            and total_buy > 0
        ):
            volume_to_buy = total_production_capacity / 2
        else:
            volume_to_buy = total_production_capacity

        if suppliers.shape[0] > 0:
            selected_sup_contracts = pd.DataFrame(
                {
                    "sign_indx": suppliers.sign_indx,
                    "cost": [
                        p * q + q * self.production_cost
                        for p, q in zip(suppliers.price, suppliers.quantity)
                    ],
                    "quantity": suppliers.quantity,
                    "delivery_t": suppliers.time,
                    "utility": [
                        self.buy_utility_func(p, q, t, bs)
                        for p, q, t, bs in zip(
                            suppliers.price,
                            suppliers.quantity,
                            suppliers.time,
                            suppliers.breach_score,
                        )
                    ],
                }
            )

            selected_sup_contracts = selected_sup_contracts[
                selected_sup_contracts.delivery_t
                < self.awi.n_steps - self.STOP_PRODUCTION_LAST_N_STEP
            ]
            selected_sup_contracts = selected_sup_contracts.sort_values(
                "utility", ascending=False
            ).T.to_dict()

            for _, contract in selected_sup_contracts.items():
                if contract["quantity"] <= volume_to_buy:
                    signatures[int(contract["sign_indx"])] = self.id
                    volume_to_buy -= contract["quantity"]

        if consumers.shape[0] > 0:
            selected_cons_contracts = pd.DataFrame(
                {
                    "sign_indx": consumers.sign_indx,
                    "quantity": consumers.quantity,
                    "delivery_t": consumers.time,
                    "utility": [
                        self.sell_utility_func(p, q, t, bs)
                        for p, q, t, bs in zip(
                            consumers.price,
                            consumers.quantity,
                            consumers.time,
                            consumers.breach_score,
                        )
                    ],
                }
            )

            selected_cons_contracts = selected_cons_contracts.sort_values(
                "utility", ascending=False
            )

            signed_sell_quantities = (
                self.past_contracts[
                    (self.past_contracts.t >= self.awi.current_step)
                    & (self.past_contracts.is_sell)
                ]
                .groupby("t")["q"]
                .sum()
            )
            stepwise_future_input_inventory = (
                np.ones(self.awi.n_steps) * input_inventory
            )

            if buying_contracts.shape[0] > 0:
                if not self.is_first_level:
                    for (partner, cid), contract in buying_contracts.iterrows():
                        if partner not in self.supplier_states:  # bankrupted
                            buying_contracts.loc[(partner, cid), "q"] = 0

                        elif (
                            self.supplier_states[partner]["cash"]
                            > contract.q
                            * self.awi.trading_prices[self.awi.my_input_product]
                            * 2
                        ):
                            continue

                        else:
                            buying_contracts.loc[(partner, cid), "q"] = int(
                                buying_contracts.loc[(partner, cid), "q"]
                                * (1 - self.supplier_states[partner]["breach_score"])
                            )

                signed_future_inputs = (
                    buying_contracts.set_index("t").groupby("t", axis=0)["q"].sum()
                )

                for t, q in signed_future_inputs.items():
                    stepwise_future_input_inventory[t:] += q

                stepwise_future_output_inventory = (
                    np.ones(self.awi.n_steps) * output_inventory
                )

                for t in range(self.awi.current_step, self.awi.n_steps):
                    if t <= self.awi.n_steps - self.STOP_PRODUCTION_LAST_N_STEP:
                        production_amount = max(
                            0, min(stepwise_future_input_inventory[t], self.prod_cap)
                        )
                        stepwise_future_input_inventory[t:] -= production_amount
                        stepwise_future_output_inventory[t:] += production_amount

                for t, q in signed_sell_quantities.items():
                    stepwise_future_output_inventory[t:] -= q

                signeds = []
                extra_q = 0
                for _, contract in selected_cons_contracts.iterrows():
                    delivery_t = contract["delivery_t"]
                    if isinstance(delivery_t, complex):  # bug
                        delivery_t = int(delivery_t.real)
                    if delivery_t < self.awi.current_step:  # bug
                        continue

                    if (
                        int(contract["quantity"])
                        <= stepwise_future_output_inventory[int(delivery_t)]
                    ):
                        signatures[int(contract["sign_indx"])] = self.id
                        stepwise_future_output_inventory[int(delivery_t) :] -= int(
                            contract["quantity"]
                        )
                        signeds.append((delivery_t, contract["quantity"]))

                    elif (
                        self.is_last_level
                        and len(self.my_suppliers)
                        and delivery_t - self.awi.current_step > 5
                        and int(contract["quantity"])
                        <= self.prod_cap * (delivery_t - self.awi.current_step)
                        - extra_q
                    ):
                        signatures[int(contract["sign_indx"])] = self.id
                        extra_q += int(contract["quantity"])

        return signatures

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        """
        Adds the signed contracts to past_contracts.
        """
        contract_list = []
        for contract in signed:
            is_sell = contract.annotation["seller"] == self.id
            partner = (
                contract.annotation["buyer"]
                if is_sell
                else contract.annotation["seller"]
            )
            cid = contract.id
            sign_step = contract.signed_at
            q = int(contract.agreement["quantity"])
            t = int(contract.agreement["time"])
            p = int(contract.agreement["unit_price"])

            if partner not in ["SELLER", "BUYER"]:
                agreement = (q, t, p)
                utility = None
                if contract.signed_at != 0:
                    for negotiator in self.negotiators_engaged[
                        int(contract.signed_at) - 1
                    ][partner]:
                        if agreement in negotiator.agreement_utility:
                            utility = negotiator.agreement_utility[agreement]

                if not utility:
                    for negotiator in self.negotiators_engaged[int(contract.signed_at)][
                        partner
                    ]:
                        if agreement in negotiator.agreement_utility:
                            utility = negotiator.agreement_utility[agreement]

            else:
                utility = np.nan

            contract_list.append([cid, partner, sign_step, is_sell, q, t, p, utility])

        self.past_contracts.reset_index(inplace=True)
        new_contracts = pd.DataFrame(
            contract_list,
            columns=[
                "cid",
                "partner",
                "sign_step",
                "is_sell",
                "q",
                "t",
                "p",
                "utility",
            ],
        )
        self.past_contracts = pd.concat((self.past_contracts, new_contracts))
        self.past_contracts.set_index(["partner", "cid"], inplace=True)

    def contract_utility(self, contract):
        q = int(contract.agreement["quantity"])
        t = int(contract.agreement["time"])
        p = int(contract.agreement["unit_price"])

        utility = (
            self.sell_utility_func(p, q, t)
            if contract.annotation["seller"] == self.id
            else self.buy_utility_func(p, q, t)
        )

        return round(utility, 3)

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: int,
        compensation_money: int,
    ) -> None:
        try:
            if agent in self.partners:
                self.partners.remove(agent)

                if agent in self.my_consumers:
                    self.my_consumers.remove(agent)
                else:
                    self.my_suppliers.remove(agent)

                if contracts:
                    self.past_contracts.drop(
                        [c.id for c in contracts], level="cid", inplace=True
                    )
        except:  # bug
            pass


class MediocreNegotiator(SAONegotiator):
    def __init__(
        self, current_step, agent_id, agent_id_long, opponent, params, to_sell, **kwargs
    ):
        super().__init__(**kwargs)
        self.nego_start_time = None
        self.negotiator_id = None
        self.initiator = None
        self.issues = None
        self.current_step = current_step
        self.agent_id = agent_id
        self.agent_id_long = agent_id_long
        self.opponent = opponent
        self.to_sell = to_sell
        self.nego_no = 0

        self.params = params
        self.eagerness = params["eagerness"]
        self.neediness = params["neediness"]
        self.t_bounds = params["t_bounds"]
        self.q_bounds = params["q_bounds"]
        self.p_bounds = params["p_bounds"]
        self.prod_cap = params["prod_cap"]
        self.reasonable_q = params["reasonable_q"]

        # valid bounds are explicitly set before self.init based on negotiation agenda
        self.valid_t_bounds = None
        self.valid_q_bounds = None
        self.valid_p_bounds = None
        self.valid_bounds = []
        self.preffered_delivery_t = None
        self.bidding = None
        self.agreement_utility = {}
        self.max_utility = None
        self.offer_availability = None
        self.last_decision = None
        self.next_bid = None
        self.received_bids = pd.DataFrame(
            columns=[
                "round",
                "bid",
                "quantity",
                "delivery_t",
                "price",
                "utility",
                "norm_utility",
            ]
        )
        self.offered_bids = pd.DataFrame(columns=self.received_bids.columns)

    def __str__(self):
        return self.negotiator_id if self.negotiator_id else str(self.q_bounds)

    def __repr__(self):
        return str(self)

    def utility_func(self, bid):
        if len(bid) == 3:
            q, t, p = bid
        else:
            q, p = bid

        if self.to_sell:
            utility = q * min((p**1.1), p * 1.5)
        else:
            if p >= self.valid_p_bounds[0]:
                utility = q / (p - self.valid_p_bounds[0] + 1)
            else:
                utility = q

        return round(utility, 3)

    def init(self):
        self.valid_bounds = [
            self.valid_q_bounds,
            self.valid_t_bounds,
            self.valid_p_bounds,
        ]

        bidding = SellBidding if self.to_sell else BuyBidding

        self.bidding = bidding(
            self.current_step,
            self.initiator,
            self.eagerness,
            self.neediness,
            self.params,
            self.valid_bounds,
            utility_func=self.utility_func,
            use_bid_func=True,
        )

        self.offer_availability = self.bidding.generate_offerable_set()

        if self.offer_availability:
            p = self.valid_p_bounds[1] if self.to_sell else self.valid_p_bounds[0]
            self.max_utility = self.utility_func(
                (self.valid_q_bounds[1], self.valid_t_bounds[0], p)
            )

        self.preffered_delivery_t = (
            self.valid_t_bounds[0]
            if self.to_sell
            else int(np.mean(self.valid_t_bounds))
        )

        return self.offer_availability

    def on_negotiation_start(self, state: MechanismState) -> None:
        self.nego_start_time = datetime.now().time().strftime("%H-%M-%S-%f")
        self.negotiator_id = self.agent_id + "@" + "step=" + str(self.current_step)
        self.nego_no += 1  # there can be multiple negotiations with the same partner in the same step

    def respond(self, state: SAOState):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        offer_utility = self.utility_func(offer)
        norm_offer_utility = round(offer_utility / self.max_utility, 3)

        self.received_bids.loc[len(self.received_bids)] = [
            state.step,
            offer,
            offer[0],
            offer[1],
            offer[2],
            offer_utility,
            norm_offer_utility,
        ]

        opponent_behavior_params = self.get_opponent_params()
        acceptance, my_bid = self.bidding.evaluate(
            offer, state.relative_time, opponent_behavior_params
        )

        if acceptance:
            if my_bid:
                my_bid_utility = self.utility_func(my_bid)
                norm_my_bid_utility = round(my_bid_utility / self.max_utility, 3)
                self.offered_bids.loc[len(self.offered_bids)] = [
                    state.step,
                    my_bid,
                    my_bid[0],
                    my_bid[1],
                    my_bid[2],
                    my_bid_utility,
                    norm_my_bid_utility,
                ]
            self.last_decision = "accept"
            return ResponseType.ACCEPT_OFFER
        else:
            self.next_bid = my_bid
            self.last_decision = "reject"
            return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState):
        my_bid = (
            self.next_bid
            if self.next_bid
            else self.bidding.bid_func(state.relative_time, self.valid_t_bounds[0])
        )
        my_bid_utility = self.utility_func(my_bid)
        norm_my_bid_utility = round(my_bid_utility / self.max_utility, 3)
        self.offered_bids.loc[len(self.offered_bids)] = [
            state.step,
            my_bid,
            my_bid[0],
            my_bid[1],
            my_bid[2],
            my_bid_utility,
            norm_my_bid_utility,
        ]
        self.last_decision = "reject"
        return my_bid

    def on_negotiation_end(self, state: MechanismState) -> None:
        if state.agreement:
            self.agreement_utility[state.agreement] = round(
                self.utility_func(state.agreement) / self.max_utility, 3
            )
        self.negotiator_id += "-agree=" + str(state.agreement)
        self.received_bids[["opponent", "is_sell", "step", "nego_no", "agree"]] = (
            self.opponent,
            self.to_sell,
            self.current_step,
            self.nego_no,
            bool(state.agreement),
        )
        self.offered_bids[["opponent", "is_sell", "step", "nego_no", "agree"]] = (
            self.opponent,
            self.to_sell,
            self.current_step,
            self.nego_no,
            bool(state.agreement),
        )

        self.received_bids.set_index("opponent", inplace=True)
        self.offered_bids.set_index("opponent", inplace=True)

    def get_opponent_params(self):
        behavior_params = {}
        data = self.received_bids[
            ["round", "quantity", "delivery_t", "price", "norm_utility"]
        ]

        if data.shape[0] > 1:
            nego_rounds, q, t, p, utility = data.values[-MAX_ROUNDS // 5 :].T
            scaled_y = MinMaxScaler().fit_transform(utility.reshape(-1, 1))
            scaled_rounds = MinMaxScaler().fit_transform(nego_rounds.reshape(-1, 1))
            reg = LinearRegression()
            reg.fit(X=scaled_rounds, y=scaled_y)

            concession_rate = reg.coef_[0][0]  # a value between -1 and 1

            mean_q, mean_t, mean_p = data[["quantity", "delivery_t", "price"]].mean()
            std_q, std_t, std_p = data[["quantity", "delivery_t", "price"]].std()

            behavior_params["concession_rate"] = concession_rate
            behavior_params["t_bounds"] = [
                max(0, int(mean_t - std_t)),
                int(mean_t + std_t),
            ]
            behavior_params["q_bounds"] = [
                max(0, int(mean_q - std_q)),
                int(mean_q + std_q),
            ]
            behavior_params["p_bounds"] = [
                max(0, int(mean_p - std_p)),
                int(mean_q + std_p),
            ]

        else:
            q, t, p = data[["quantity", "delivery_t", "price"]].values[0]
            behavior_params["concession_rate"] = -1
            behavior_params["t_bounds"] = [t, t]
            behavior_params["q_bounds"] = [q, q]
            behavior_params["p_bounds"] = [p, p]

        return behavior_params

    # deactivated
    def target_quantity(self, step: int, sell: bool) -> int:
        return None

    # deactivated
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        return None
