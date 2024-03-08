"""
An implementation of SCML Oneshot game's agent
Author: Yuchen Liu
Last modified: July-29-2021
negmas version: 0.8.4
scml version: 0.4.6
"""

import time

from negmas import SAONMI, ResponseType, SAOState
from negmas.common import MechanismState
from negmas.situated import Contract
from scml.oneshot import QUANTITY, UNIT_PRICE
from scml.oneshot.agent import OneShotAgent

__all__ = ["UcOneshotAgent3_4"]


class UcOneshotAgent3_4(OneShotAgent):
    def init(self) -> None:
        """Called once after the agent-world interface is initialized"""
        # secured quantity value of each step
        self.quantity_secured = 0

        # record partners, partner's name should be like "08Dec@2"
        # index of each partner's name, c_rate, upper and lower limit should be same
        self.partners = []
        self.concession_rates = []
        self.upper_limits = []
        self.lower_limits = []
        self.th_matrix = []
        self.actual_price = []

        # record the partner/competitor ratio, if there is many partners but less competitors we can have a greedier strategy when negotiating and vise versa
        # the more the value, the greeder the negotiation strategy
        self.p_c_ratio = (
            len(self.awi.all_consumers[0]) / len(self.awi.all_consumers[1])
            if self.awi.level == 1
            else len(self.awi.all_consumers[1]) / len(self.awi.all_consumers[0])
        )
        self.p_c_ratio = self.p_c_ratio**0.8

        # default value of concession_rate, upper and lower th limits
        self.default_concession_rate = 0.35
        self.default_upper_limits = 0.15
        self.default_lower_limits = 0.15
        # negotiation step in oneshot game is 20, so just use 20 here
        self.default_th_value = [
            ((19 - a) / 19) ** (self.default_concession_rate)
            * (1.0 - self.default_lower_limits - self.default_upper_limits)
            + self.default_lower_limits
            for a in range(20)
        ]

        # define max and min value of concession rate, upper lower limit
        self.CONCESSION_RATE = {"min_value": 0.1, "max_value": 0.5}
        self.UPPER_LIMIT = {"min_value": 0.0, "max_value": 0.35}
        self.LOWER_LIMIT = {"min_value": 0.2, "max_value": 0.35}

        # coefficient for adjusting concession rate, upper and lower limits (1 - range(20)/20 ** 0.3 except the first number)
        self.success_step_factor = [
            0.59290947,
            0.59290947,
            0.49881277,
            0.43398573,
            0.38296614,
            0.34024604,
            0.30315470,
            0.27017218,
            0.24034221,
            0.21301989,
            0.18774760,
            0.16418761,
            0.14208280,
            0.12123246,
            0.10147656,
            0.08268525,
            0.06475155,
            0.04758620,
            0.03111384,
            0.01527020,
        ]

        # for collecting negotiation result information
        # key: partner; value: (success_or_not: bool,
        # negotiation_steps: int,
        # accepted/end_neg by: int (0: accepted/end_neg by me, 1: accepted/end_neg by partner, 2: time_over)
        self.neg_results = {}

        # record negotiation mechanism's min and max price value of this simulation
        self.min_value = None
        self.max_value = None

        # record past exo_quantity and negotiation-reached quantity history
        self.exo_quantity_history = [-1 for _ in range(self.awi.n_steps)]
        self.neg_quantity_history = [-1 for _ in range(self.awi.n_steps)]

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""
        pass

    def step(self) -> None:
        """Called at at the END of every production step (day)"""
        # sleep 0.1ms to wait on_negotiation_success/failure() functions to finish
        time.sleep(0.0001)

        # record exo_quantity and negotiation-reached quantity
        self.exo_quantity_history[self.awi.current_step] = (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
        )
        self.neg_quantity_history[self.awi.current_step] = self.quantity_secured

        # reset quantity_secured value for next step
        self.quantity_secured = 0

        # if negotiated with a new partner, add this partner to our data set
        for partner in self.neg_results.keys():
            if partner not in self.partners:
                self.partners.append(partner)
                self.concession_rates.append(self.default_concession_rate)
                self.upper_limits.append(self.default_upper_limits)
                self.lower_limits.append(self.default_lower_limits)
                th = [((19 - a) / 19) ** (0.3) * 0.65 + 0.15 for a in range(20)]
                self.th_matrix.append(th)
                step_price = []
                for t in th:
                    p = (
                        round(self.min_value + t * (self.max_value - self.min_value))
                        if self.awi.level == 0
                        else round(
                            self.max_value - t * (self.max_value - self.min_value)
                        )
                    )
                    step_price.append(p)
                self.actual_price.append(step_price)

        # adjust strategy(concession rate, upper price limit and lower price limit) of each partner
        # according to negotiation results, adjust upper_limit, lower_limit and concession rate of each partners
        for partner in self.neg_results.keys():
            partner_index = self.partners.index(partner)
            if self.neg_results[partner][0]:
                # successed negotiations
                self._th_coefficient_adjust_success(
                    partner_index, self.neg_results[partner][1]
                )
            else:
                # failed negotiations
                self._th_coefficient_adjust_failure(
                    partner_index,
                    self.neg_results[partner][1],
                    self.neg_results[partner][2],
                )

        # clear last step's negotiation results
        self.neg_results.clear()

        # adjust values if out of limit
        for i in range(len(self.partners)):
            self.concession_rates[i] = max(
                self.CONCESSION_RATE["min_value"], self.concession_rates[i]
            )
            self.concession_rates[i] = min(
                self.CONCESSION_RATE["max_value"], self.concession_rates[i]
            )
            self.upper_limits[i] = max(
                self.UPPER_LIMIT["min_value"], self.upper_limits[i]
            )
            self.upper_limits[i] = min(
                self.UPPER_LIMIT["max_value"], self.upper_limits[i]
            )
            self.lower_limits[i] = max(
                self.LOWER_LIMIT["min_value"], self.lower_limits[i]
            )
            self.lower_limits[i] = min(
                self.LOWER_LIMIT["max_value"], self.lower_limits[i]
            )

        # calculate basic th value for each partner
        for i in range(len(self.partners)):
            self.th_matrix[i] = [
                ((19 - a) / 19) ** (self.concession_rates[i]) for a in range(20)
            ]
            for j in range(20):
                self.th_matrix[i][j] = min(
                    self.th_matrix[i][j], (1 - self.upper_limits[i])
                )
                self.th_matrix[i][j] = max(self.th_matrix[i][j], self.lower_limits[i])

        # based on the number of partners, adjust lowest utility to second(third ... etc.) lowest utility of each step
        round_up_coefficient = 5
        round_up_number = round(len(self.partners) / round_up_coefficient)
        if round_up_number == 0:
            pass
        elif self.awi.level == 0:
            for i in range(20):
                temp = []
                for j in range(len(self.partners)):
                    temp.append(self.th_matrix[j][i])
                temp.sort()
                lower_limit = temp[round_up_number]
                for k in range(len(self.partners)):
                    self.th_matrix[k][i] = max(lower_limit, self.th_matrix[k][i])
        else:  # self.awi.level == 1
            for i in range(20):
                temp = []
                for j in range(len(self.partners)):
                    temp.append(self.th_matrix[j][i])
                temp.sort(reverse=True)
                upper_limit = temp[round_up_number]
                for k in range(len(self.partners)):
                    self.th_matrix[k][i] = min(upper_limit, self.th_matrix[k][i])

        # calculate actual step price for each partner
        # adjust tx values based on self.p_c_ratio & all price come close to catalogue price
        catalog_price = self.awi.catalog_prices[1]
        self.actual_price = [[0 for _ in range(20)] for _ in range(len(self.partners))]
        quantity_adjust_ratio = self._past_quantity_adjust_coefficient()
        for i in range(len(self.partners)):
            for j in range(20):
                th = self.th_matrix[i][j] * self.p_c_ratio * quantity_adjust_ratio
                th = max(0.1, th)
                th = min(0.9, th)
                p = (
                    self.min_value + th * (self.max_value - self.min_value)
                    if self.awi.level == 0
                    else self.max_value - th * (self.max_value - self.min_value)
                )
                self.actual_price[i][j] = p
                self.actual_price[i][j] = self._adjust_with_catalog(
                    catalog_price, self.actual_price[i][j], 10
                )

    def propose(self, negotiator_id: str, state: MechanismState):
        """Called when the agent is asking to propose in one negotiation"""
        # collect info
        ami = self.get_ami(negotiator_id)
        partner = (
            ami.annotation["buyer"] if self.awi.level == 0 else ami.annotation["seller"]
        )

        if self._required_quantity() <= 0:
            return None
        else:
            if partner in self.partners:
                target_price = self.actual_price[self.partners.index(partner)][
                    state.step
                ]
            else:
                th = self.default_th_value[state.step]
                target_price = (
                    round(
                        ami.issues[UNIT_PRICE].min_value
                        + th
                        * (
                            ami.issues[UNIT_PRICE].max_value
                            - ami.issues[UNIT_PRICE].min_value
                        )
                    )
                    if self.awi.level == 0
                    else round(
                        ami.issues[UNIT_PRICE].max_value
                        - th
                        * (
                            ami.issues[UNIT_PRICE].max_value
                            - ami.issues[UNIT_PRICE].min_value
                        )
                    )
                )

            return (
                max(self._required_quantity(), 0),
                self.awi.current_step,
                target_price,
            )

    def respond(self, negotiator_id: str, state: SAOState) -> ResponseType:
        """Called when the agent is asked to respond to an offer"""
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        # collect info
        ami = self.get_ami(negotiator_id)
        partner = (
            ami.annotation["buyer"] if self.awi.level == 0 else ami.annotation["seller"]
        )
        offer_quantity = offer[QUANTITY]
        offer_unit_price = offer[UNIT_PRICE]

        if partner in self.partners:
            target_price = self.actual_price[self.partners.index(partner)][state.step]
        else:
            min_price = ami.issues[UNIT_PRICE].min_value
            max_price = ami.issues[UNIT_PRICE].max_value
            th = self.default_th_value[state.step]
            target_price = (
                round(min_price + th * (max_price - min_price))
                if self.awi.level == 0
                else round(max_price - th * (max_price - min_price))
            )

        while True:
            # record required_quantity
            # when respond with ACCEPT re-check the required_quantity
            required_quantity = self._required_quantity()
            if required_quantity <= 0:
                return ResponseType.END_NEGOTIATION

            if self.awi.level == 0:
                if (
                    offer_quantity <= required_quantity
                    and offer_unit_price >= target_price
                ):
                    # quantity and price meet requirement
                    if required_quantity == self._required_quantity():
                        return ResponseType.ACCEPT_OFFER
                    else:
                        continue
                elif offer_unit_price < target_price:
                    # too cheap
                    return ResponseType.REJECT_OFFER
                else:
                    # over quantity mechanism
                    over_quantity_ratio = (
                        offer_quantity - required_quantity
                    ) / required_quantity
                    target_price += round((over_quantity_ratio * 10) ** 1.9)
                    if offer_unit_price >= target_price:
                        if required_quantity == self._required_quantity():
                            return ResponseType.ACCEPT_OFFER
                        else:
                            continue
                    else:
                        return ResponseType.REJECT_OFFER
            else:  # self.awi.level == 1
                if (
                    offer_quantity <= required_quantity
                    and offer_unit_price <= target_price
                ):
                    # quantity and price meet requirement
                    if required_quantity == self._required_quantity():
                        return ResponseType.ACCEPT_OFFER
                    else:
                        continue
                elif offer_unit_price > target_price:
                    # too expensive
                    return ResponseType.REJECT_OFFER
                else:
                    # over quantity mechanism
                    over_quantity_ratio = (
                        offer_quantity - required_quantity
                    ) / required_quantity
                    target_price -= round((over_quantity_ratio * 10) ** 1.9)
                    if offer_unit_price <= target_price:
                        if required_quantity == self._required_quantity():
                            return ResponseType.ACCEPT_OFFER
                        else:
                            continue
                    else:
                        return ResponseType.REJECT_OFFER

    def on_negotiation_success(self, contract: Contract, mechanism: SAONMI) -> None:
        """Called when a negotiation the agent is a party of ends with agreement"""
        # update required_quantity
        quantity = contract.agreement["quantity"]
        self.quantity_secured += quantity

        # assign negotiation min value and max value
        if (self.min_value is None) or (self.max_value is None):
            self.min_value = mechanism.issues[2].min_value
            self.max_value = mechanism.issues[2].max_value

        # record data
        partner = (
            mechanism.annotation["buyer"]
            if self.awi.level == 0
            else mechanism.annotation["seller"]
        )
        accepted_by = int(partner == mechanism.state.current_proposer_agent)
        no_of_step = mechanism.state.step
        # no_of_step = len(mechanism["mechanism"]._history)
        # if (
        #     partner
        #     == mechanism["mechanism"]._history[no_of_step - 1].current_proposer_agent
        # ):
        #     accepted_by = 1
        # else:
        #     accepted_by = 0
        #
        self.neg_results[partner] = (True, no_of_step, accepted_by)

    def on_negotiation_failure(
        self,
        partners,
        annotation,
        mechanism: SAONMI,
        state: MechanismState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""
        # assign negotiation min value and max value
        if (self.min_value is None) or (self.max_value is None):
            self.min_value = mechanism.issues[2].min_value
            self.max_value = mechanism.issues[2].max_value

        # record data
        partner = (
            mechanism.annotation["buyer"]
            if self.awi.level == 0
            else mechanism.annotation["seller"]
        )
        no_of_step = mechanism.state.step
        # no_of_step = len(mechanism["_mechanism"].history)
        if no_of_step >= 19:
            self.neg_results[partner] = (False, 19, 2)  # 2:time(step) over
            return
        rejected_by = int(partner == mechanism.state.current_proposer_agent)
        # if (
        #     partner
        #     == mechanism["_mechanism"].history[no_of_step - 1].current_proposer_agent
        # ):
        #     rejected_by = 1
        # else:
        #     rejected_by = 0

        self.neg_results[partner] = (False, no_of_step, rejected_by)

    """
    self defined functions
    """

    def _th_coefficient_adjust_success(self, partner_index, negotiation_step) -> None:
        adjust_coefficient = self.success_step_factor[negotiation_step - 1]
        self.concession_rates[partner_index] -= adjust_coefficient * 0.075

    def _th_coefficient_adjust_failure(
        self, partner_index, negotiation_step, rejected_by
    ) -> None:
        adjust_coefficient = self.success_step_factor[negotiation_step - 1]
        if rejected_by == 0:
            self.concession_rates[partner_index] += adjust_coefficient * 0.0056
        else:
            self.concession_rates[partner_index] += adjust_coefficient * 0.023

    def _required_quantity(self) -> int:
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.quantity_secured
        )

    def _past_quantity_adjust_coefficient(self) -> float:
        try:
            if self.awi.current_step == 0:
                neg_exo_ratio = (
                    self.neg_quantity_history[0] / self.exo_quantity_history[0]
                )
            elif self.awi.current_step == 1:
                neg_exo_ratio = (
                    0.6 * self.neg_quantity_history[1] / self.exo_quantity_history[1]
                    + 0.4 * self.neg_quantity_history[0] / self.exo_quantity_history[0]
                )
            else:
                neg_exo_ratio = (
                    0.5
                    * self.neg_quantity_history[self.awi.current_step]
                    / self.exo_quantity_history[self.awi.current_step]
                    + 0.3
                    * self.neg_quantity_history[self.awi.current_step - 1]
                    / self.exo_quantity_history[self.awi.current_step - 1]
                    + 0.2
                    * self.neg_quantity_history[self.awi.current_step - 2]
                    / self.exo_quantity_history[self.awi.current_step - 2]
                )
            return (
                1.0
                + (2 * (1 / (1 + 2.71828 ** -(8 * neg_exo_ratio - 8))) - 1) ** 5 * 0.20
            )  # modification of sigmoid function
        except:  # divide by 0
            return 1.0

    @staticmethod
    def _adjust_with_catalog(catalog_price: int, price: float, abs_limit: int) -> int:
        diff = price - catalog_price
        if abs(diff) >= abs_limit:
            return (
                round(catalog_price + abs_limit)
                if diff >= 0
                else round(catalog_price - abs_limit)
            )
        else:
            return round(price - (diff * 0.37))
