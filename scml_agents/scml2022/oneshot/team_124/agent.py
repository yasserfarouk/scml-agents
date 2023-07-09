from scml.oneshot import OneShotAgent

QUANTITY = 0
TIME = 1
UNIT_PRICE = 2

from negmas import Outcome, ResponseType

__all__ = ["LearningAdaptiveAgent"]


class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def before_step(self):
        self.secured = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state) -> "Outcome":
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs
            else ResponseType.REJECT_OFFER
        )

    def best_offer(self, negotiator_id):
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
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


class BetterAgent(SimpleAgent):
    """A greedy agent based on OneShotAgent with more sane strategy"""

    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._e = concession_exponent

    def propose(self, negotiator_id: str, state) -> "Outcome":
        offer = super().propose(negotiator_id, state)
        if not offer:
            return None
        offer = list(offer)
        offer[UNIT_PRICE] = self._find_good_price(self.get_nmi(negotiator_id), state)
        return tuple(offer)

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        if response != ResponseType.ACCEPT_OFFER:
            return response
        ami = self.get_nmi(negotiator_id)
        return (
            response
            if self._is_good_price(ami, state, offer[UNIT_PRICE])
            else ResponseType.REJECT_OFFER
        )

    def _is_good_price(self, ami, state, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step, ami.n_steps)
        # a good price is one better than the threshold
        if self._is_selling(ami):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, ami, state):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step, ami.n_steps)
        # offer a price that is around th of your best possible price
        if self._is_selling(ami):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _price_range(self, ami):
        """Finds the minimum and maximum prices"""
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
        return mn, mx

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e


class AdaptiveAgent(BetterAgent):
    """Considers best price offers received when making its decisions"""

    def before_step(self):
        super().before_step()
        self._best_selling, self._best_buying = 0.0, float("inf")

    def respond(self, negotiator_id, state):
        """Save the best price received"""
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        ami = self.get_nmi(negotiator_id)
        if self._is_selling(ami):
            self._best_selling = max(offer[UNIT_PRICE], self._best_selling)
        else:
            self._best_buying = min(offer[UNIT_PRICE], self._best_buying)
        return response

    def _price_range(self, ami):
        """Limits the price by the best price received"""
        mn, mx = super()._price_range(ami)
        if self._is_selling(ami):
            mn = max(mn, self._best_selling)
        else:
            mx = min(mx, self._best_buying)
        return mn, mx


GreedynessWeight = [-0.15, -0.075, 0.075, 0.15]
GreedLevel1 = 0
GreedLevel2 = 1
GreedLevel3 = 2
GreedLevel4 = 3


class LearningAdaptiveAgent(AdaptiveAgent):
    def before_step(self):
        # Save the last buying/selling prices
        super().before_step()
        self._selling_by_id = {}
        self._buying_by_id = {}
        self._selling = []
        self._buying = []

    def respond(self, negotiator_id, state):
        offer = state.current_offer
        if not offer:
            return ResponseType.REJECT_OFFER
        response = super().respond(negotiator_id, state)
        nmi = self.get_nmi(negotiator_id)
        if self._is_selling(nmi):
            if not negotiator_id in self._selling_by_id.keys():
                self._selling_by_id[negotiator_id] = []
            val1 = max(offer[UNIT_PRICE], max(self._selling, default=0))
            val2 = max(
                offer[UNIT_PRICE], max(self._selling_by_id[negotiator_id], default=0)
            )
            self._selling_by_id[negotiator_id].append(val2)
            self._selling.append(val1)
        else:
            if not negotiator_id in self._buying_by_id.keys():
                self._buying_by_id[negotiator_id] = []
            self._buying_by_id[negotiator_id].append(offer[UNIT_PRICE])
            val1 = min(offer[UNIT_PRICE], min(self._buying, default=float("inf")))
            val2 = min(
                offer[UNIT_PRICE],
                min(self._buying_by_id[negotiator_id], default=float("inf")),
            )
            self._buying_by_id[negotiator_id].append(val2)
            self._buying.append(val1)
        return response

    def _price_range(self, nmi):
        mn, mx = super()._price_range(nmi)
        unit_price_issue = nmi.issues[UNIT_PRICE]
        if self._is_selling(nmi):
            if nmi.negotiator_ids[1] in self._selling_by_id.keys():
                avg = average(self._selling_by_id[nmi.negotiator_ids[1]], mn)
                type = self._find_negotiator_greed_level(avg, True)
                w = 0
                if type != -1:
                    w = GreedynessWeight[type]
                mn = max(max((1 - w) * mn + (w) * avg, avg), unit_price_issue.min_value)
        else:
            if nmi.negotiator_ids[0] in self._buying_by_id.keys():
                avg = average(self._buying_by_id[nmi.negotiator_ids[0]], mx)
                w = 0
                type = self._find_negotiator_greed_level(avg, False)
                if type != -1:
                    w = GreedynessWeight[type]
                mx = min(min((1 - w) * mx + (w) * avg, avg), unit_price_issue.max_value)
        return mn, mx

    def _find_negotiator_greed_level(self, negotiator_avg, is_seeling):
        if negotiator_avg == 0:
            return -1
        if is_seeling:
            total_avg = average(self._selling, 0)
            if total_avg == 0:
                return -1
            if negotiator_avg < total_avg:
                if negotiator_avg / total_avg < 0.5:
                    return GreedLevel4
                else:
                    return GreedLevel3
            else:
                if total_avg / negotiator_avg < 0.5:
                    return GreedLevel1
                else:
                    return GreedLevel2
        else:
            total_avg = average(self._buying, 0)
            if total_avg == 0:
                return -1
            if negotiator_avg < total_avg:
                if negotiator_avg / total_avg < 0.5:
                    return GreedLevel1
                else:
                    return GreedLevel2
            else:
                if total_avg / negotiator_avg < 0.5:
                    return GreedLevel4
                else:
                    return GreedLevel3


def average(list_items, n):
    t = 0.95  # time factor
    if len(list_items) == 0:
        return n
    else:
        sum = 0
        divider = 0
        for i, item in enumerate(list_items):
            w = t ** (len(list_items) - i)
            sum += w * item
            divider += w
        return sum / divider
