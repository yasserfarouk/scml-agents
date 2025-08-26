from collections import defaultdict
from scml.oneshot.common import *
from .simple_agent import SimpleAgent

__all__ = ["OptimisticAgent"]


class OptimisticAgent(SimpleAgent):
    """A greedy agent based on SimpleAgent with more sane strategy"""

    def propose(self, negotiator_id, state):
        offer = self.good_offer(negotiator_id, state)
        if offer is None:
            return offer
        offered = self._offered(negotiator_id)
        offered[negotiator_id] = {offer[TIME]: offer[QUANTITY]}
        return offer

    def before_step(self):
        self.offered_sales = defaultdict(lambda: defaultdict(int))
        self.offered_supplies = defaultdict(lambda: defaultdict(int))

    def on_negotiation_success(self, contract, mechanism):
        partner = [_ for _ in contract.partners if _ != self.id][0]
        offered = self._offered(partner)
        offered[partner] = dict()

    def _offered(self, partner):
        if self.is_consumer(partner):
            return self.offered_sales
        return self.offered_supplies

    def _needs(self, partner, t):
        n = super()._needs(partner, t)
        offered = self._offered(partner)
        for k, v in offered[partner].items():
            if k > t:
                continue
            n = max(0, n - v)
        return int(n)
