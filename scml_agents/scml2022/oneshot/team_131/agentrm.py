from scml.oneshot import *
from negmas import ResponseType
from scml.oneshot import *
import math
import random

random.seed(0)

str_seller = {True: "Seller", False: "Buyer", None: "Unknown"}

__all__ = ["AgentRM"]


class AgentRM(OneShotAgent):
    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._e = concession_exponent
        self.your_needs = dict()
        self._opp_best_success_selling, self._opp_best_success_buying = dict(), dict()

    def init(self):
        self.n_consumers = len(self.awi.my_consumers)  # for Seller
        self.n_supliers = len(self.awi.my_suppliers)  # for Buyer
        super().init()

    def before_step(self):
        self.secured = 0
        self.your_needs = dict()
        self._rjc_best_price_selling = 0
        self._rjc_best_price_buying = float("inf")
        self.n_end_negotiators = 0
        self.end_negotiators = []
        self.accepted_offers = dict()  # アクセプトしたオファー
        self.first_proposer = None

    def step(self):
        super().step()

    def on_negotiation_success(self, contract, mechanism):
        my_id = self.awi.agent.id
        negotiator_id = list(contract.partners).copy()
        negotiator_id.remove(my_id)
        negotiator_id = negotiator_id[0]
        ami = self.get_nmi(negotiator_id)

        if self._is_selling(ami):
            if negotiator_id in self._opp_best_success_selling.keys():
                self._opp_best_success_selling[negotiator_id] = max(
                    self._opp_best_success_selling[negotiator_id],
                    contract.agreement["unit_price"],
                )
            else:
                self._opp_best_success_selling[negotiator_id] = contract.agreement[
                    "unit_price"
                ]
        else:
            if negotiator_id in self._opp_best_success_buying.keys():
                self._opp_best_success_buying[negotiator_id] = min(
                    self._opp_best_success_buying[negotiator_id],
                    contract.agreement["unit_price"],
                )
            else:
                self._opp_best_success_buying[negotiator_id] = contract.agreement[
                    "unit_price"
                ]

        self.secured += contract.agreement["quantity"]
        if negotiator_id in self.your_needs:
            self.your_needs.pop(negotiator_id)

    def on_negotiation_end(self, negotiator_id: str, state) -> None:
        self.n_end_negotiators += 1
        self.end_negotiators.append(negotiator_id)
        return super().on_negotiation_end(negotiator_id, state)

    def propose(self, negotiator_id: str, state) -> "Outcome":
        if self.first_proposer == None:
            self.first_proposer = True

        ami = self.get_nmi(negotiator_id)
        offer = self.best_offer(negotiator_id)

        if not offer:
            return None
        offer = list(offer)

        if negotiator_id in self.your_needs.keys():
            offer[QUANTITY] = max(
                min(offer[QUANTITY], self.your_needs[negotiator_id]),
                ami.issues[QUANTITY].min_value,
            )
        offer[UNIT_PRICE] = self._find_good_price(negotiator_id, state, offer[QUANTITY])

        if (
            not (self._is_selling(ami))
            and offer[QUANTITY] * (offer[UNIT_PRICE] + self.awi.profile.cost)
            > self.awi.current_balance
        ):
            offer[QUANTITY] = math.floor(
                self.awi.current_balance / (offer[UNIT_PRICE] + self.awi.profile.cost)
            )

        return tuple(offer)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        if self.first_proposer == None:
            self.first_proposer = False

        self.your_needs[negotiator_id] = offer[QUANTITY]
        ami = self.get_nmi(negotiator_id)
        my_needs = self._needed(negotiator_id)

        # 必要量を既に満たしていたら交渉を終了する
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION

        # 基本のレスポンス
        if self._is_good_price(
            negotiator_id, ami, state, offer[UNIT_PRICE], offer[QUANTITY]
        ):
            response = (
                ResponseType.ACCEPT_OFFER
                if offer[QUANTITY] <= my_needs
                else ResponseType.REJECT_OFFER
            )
        else:
            response = ResponseType.REJECT_OFFER
            # その日に受けたオファーのうち，価格要因でREJECTした最も良い単価を記憶しておく
            if self._is_selling(ami):
                self._rjc_best_price_selling = max(
                    self._rjc_best_price_selling, offer[UNIT_PRICE]
                )
            else:
                self._rjc_best_price_buying = min(
                    self._rjc_best_price_buying, offer[UNIT_PRICE]
                )

        # 自分の応答でその日の交渉が全て終了の場合の特別な処理
        if state.step == ami.n_steps - 1 and self.first_proposer == False:
            if self.n_consumers - self.n_end_negotiators == 1:
                if self._is_selling(ami):
                    if offer[QUANTITY] > my_needs:
                        if (
                            -(offer[UNIT_PRICE] - self.awi.profile.cost) * my_needs
                            + self.awi.current_shortfall_penalty
                            * self.awi.trading_prices[self.awi.my_output_product]
                            * (offer[QUANTITY] - my_needs)
                            < self.awi.current_disposal_cost
                            * self.awi.trading_prices[self.awi.my_input_product]
                            * my_needs
                        ):
                            response = ResponseType.ACCEPT_OFFER
                        else:
                            response = ResponseType.REJECT_OFFER
                    else:
                        if (
                            offer[UNIT_PRICE]
                            + self.awi.current_disposal_cost
                            * self.awi.trading_prices[self.awi.my_input_product]
                            > self.awi.profile.cost
                        ):
                            response = ResponseType.ACCEPT_OFFER
                        else:
                            response = ResponseType.REJECT_OFFER
                else:
                    if offer[QUANTITY] > my_needs:
                        if (
                            offer[UNIT_PRICE] * offer[QUANTITY]
                            + self.awi.current_disposal_cost
                            * self.awi.trading_prices[self.awi.my_input_product]
                            * (offer[QUANTITY] - my_needs)
                            - offer[QUANTITY]
                            * self.awi.current_exogenous_output_price
                            / self.awi.current_exogenous_output_quantity
                            < self.awi.current_shortfall_penalty
                            * self.awi.trading_prices[self.awi.my_output_product]
                            * my_needs
                        ):
                            return ResponseType.ACCEPT_OFFER
                        else:
                            return ResponseType.REJECT_OFFER
                    else:
                        if (
                            self.awi.current_exogenous_output_price
                            / self.awi.current_exogenous_output_quantity
                            + self.awi.current_shortfall_penalty
                            * self.awi.trading_prices[self.awi.my_output_product]
                            > offer[UNIT_PRICE]
                        ):
                            return ResponseType.ACCEPT_OFFER
                        else:
                            return ResponseType.REJECT_OFFER

        # 破産防止
        if (
            not (self._is_selling(ami))
            and offer[QUANTITY] * (offer[UNIT_PRICE] + self.awi.profile.cost)
            > self.awi.current_balance
        ):
            response = ResponseType.REJECT_OFFER

        if response == ResponseType.ACCEPT_OFFER:
            self.accepted_offers[negotiator_id] = offer

        return response

    def best_offer(self, negotiator_id):
        ami = self.get_nmi(negotiator_id)
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3

        offer[QUANTITY] = my_needs
        # 売り手の提案に対する交渉相手の応答は，交渉ラウンドを跨ぐので他の交渉相手に提案する際にヤババンてなる
        offer[QUANTITY] = max(
            min(offer[QUANTITY], quantity_issue.max_value), quantity_issue.min_value
        )

        if negotiator_id in self.your_needs.keys():
            offer[QUANTITY] = max(
                min(offer[QUANTITY], self.your_needs[negotiator_id]),
                quantity_issue.min_value,
            )

        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    def _get_negotiator_id(self, ami):
        if self._is_selling(ami):
            return ami.annotation["buyer"]
        else:
            return ami.annotation["seller"]

    def _is_good_price(self, negotiator_id, ami, state, price, quantity):
        mn, mx = self._price_range(negotiator_id, ami, quantity)
        if self._is_selling(ami):
            th = self._th(state.step, ami.n_steps)
            return (price - mn) >= th * (mx - mn)
        else:
            th = self._th(state.step, ami.n_steps)
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, negotiator_id, state, quantity):
        ami = self.get_nmi(negotiator_id)
        # 後から提案したエージェントの提案は，相手がアクセプトしたか否かで量を調整できないので，頑固めにする
        if self._is_selling(ami):
            if self.first_proposer == False:
                mx = ami.issues[UNIT_PRICE].max_value
                mn = (
                    self.awi.current_exogenous_input_price
                    / self.awi.current_exogenous_input_quantity
                    + self.awi.profile.cost
                    + self.awi.current_disposal_cost
                    * self.awi.trading_prices[self.awi.my_input_product]
                    * max(self._needed(negotiator_id) - quantity, 0)
                    / quantity
                )
                mn = max(
                    min(mn, ami.issues[UNIT_PRICE].max_value),
                    ami.issues[UNIT_PRICE].min_value,
                )
                mn = (mx + mn) / 2
            else:
                mn, mx = self._price_range(negotiator_id, ami, quantity)
            th = self._th(state.step, ami.n_steps)
            return mn + th * (mx - mn)
        else:
            mn, mx = self._price_range(negotiator_id, ami, quantity)
            mn = min(mn * 1.1, ami.issues[UNIT_PRICE].max_value)
            th = self._th(state.step, ami.n_steps)
            return mx - th * (mx - mn)

    def _price_range(self, negotiator_id, ami, quantity):
        if self._is_selling(ami):
            # 交渉相手とそれまでに合意した最高値よりちょっと高い額から，その日に誰かが提案して自分がリジェクトした中の最高値へ譲歩
            mx = (
                (
                    ami.issues[UNIT_PRICE].max_value
                    + self._opp_best_success_selling[negotiator_id]
                )
                / 2
                if negotiator_id in self._opp_best_success_selling.keys()
                else ami.issues[UNIT_PRICE].max_value
            )
            mn = (
                self.awi.current_exogenous_input_price
                / self.awi.current_exogenous_input_quantity
                + self.awi.profile.cost
            )
            mn = max(
                min(
                    max(mn, self._rjc_best_price_selling),
                    ami.issues[UNIT_PRICE].max_value,
                ),
                ami.issues[UNIT_PRICE].min_value,
            )
            if mx < mn:
                mx = ami.issues[UNIT_PRICE].max_value
        else:
            # 交渉相手とそれまでに合意した最安値よりちょっと安い額から，その日に誰かが提案して自分がリジェクトした中の最安値へ譲歩
            mn = (
                (
                    ami.issues[UNIT_PRICE].min_value
                    + self._opp_best_success_buying[negotiator_id]
                )
                / 2
                if negotiator_id in self._opp_best_success_buying.keys()
                else ami.issues[UNIT_PRICE].min_value
            )
            mx = (
                self.awi.current_exogenous_output_price
                / self.awi.current_exogenous_output_quantity
                - self.awi.profile.cost
            )
            mx = max(
                min(
                    min(mx, self._rjc_best_price_buying),
                    ami.issues[UNIT_PRICE].max_value,
                ),
                ami.issues[UNIT_PRICE].min_value,
            )
            if mx < mn:
                mn = ami.issues[UNIT_PRICE].min_value
        return mn, mx

    def _th(self, step, n_steps, e=None):
        if e == None:
            e = self._e
        if n_steps <= 1:
            return 1
        return ((n_steps - step - 1) / (n_steps - 1)) ** e
