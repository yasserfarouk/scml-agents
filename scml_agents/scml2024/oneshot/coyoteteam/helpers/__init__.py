import math
from typing import Iterable

import requests
from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.std import *
from scml.runner import WorldRunner
import pandas as pd
from rich.jupyter import print
import warnings

from scml_agents.scml2020.monty_hall.nvm_lib2.ilpnvm import current_inventory

warnings.filterwarnings("ignore")
# setup disply parameters
from matplotlib import pylab as plt
from matplotlib.ticker import StrMethodFormatter

float_formatter = StrMethodFormatter("{x:0.03f}")
from IPython.core.display import display, HTML
import random

random.seed(10)
import numpy as np
from scml import AWI
np.random.seed(10)
from rich.jupyter import print



class SimpleAgent(StdAgent):
    """A greedy agent based on StdAgent"""

    def __init__(self, *args, production_level=0.25, future_concession=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.production_level = production_level
        self.future_concession = future_concession

    def propose(self, negotiator_id: str, state):
        return self.good_offer(negotiator_id, state)

    def respond(self, negotiator_id, state, source=""):
        # accept any quantity needed at a good price
        offer = state.current_offer
        return (
            ResponseType.ACCEPT_OFFER
            if self.is_needed(negotiator_id, offer)
            and self.is_good_price(negotiator_id, offer, state)
            else ResponseType.REJECT_OFFER
        )

    def is_needed(self, partner, offer):
        if offer is None:
            return False
        return offer[QUANTITY] <= self._needs(partner, offer[TIME])

    def is_good_price(self, partner, offer, state):
        # ending the negotiation is bad
        if offer is None:
            return False
        nmi = self.get_nmi(partner)
        if not nmi:
            return False
        issues = nmi.issues
        minp = issues[UNIT_PRICE].min_value
        maxp = issues[UNIT_PRICE].max_value
        # use relative negotiation time to concede
        # for offers about today but conede less for
        # future contracts
        r = state.relative_time
        if offer[TIME] > self.awi.current_step:
            r *= self.future_concession
        # concede linearly
        if self.is_consumer(partner):
            return offer[UNIT_PRICE] >= minp + (1 - r) * (maxp - minp) 
        return -offer[UNIT_PRICE] >= -minp + (1 - r) * (minp - maxp)

    def good_offer(self, partner, state):
        nmi = self.get_nmi(partner)
        if not nmi:
            return None
        issues = nmi.issues
        qissue = issues[QUANTITY]
        pissue = issues[UNIT_PRICE]
        for t in sorted(list(issues[TIME].all)):
            # find my needs for this day
            needed = self._needs(partner, t)
            if needed <= 0:
                continue
            offer = [-1] * 3
            # ask for as much as I need for this day
            offer[QUANTITY] = max(min(needed, qissue.max_value), qissue.min_value)
            offer[TIME] = t
            # use relative negotiation time to concede
            # for offers about today but concede less for
            # future contracts
            r = state.relative_time
            if t > self.awi.current_step:
                r *= self.future_concession
            # concede linearly on price
            minp, maxp = pissue.min_value, pissue.max_value
            if self.is_consumer(partner):
                offer[UNIT_PRICE] = int(minp + (maxp - minp) * (1 - r) + 0.5)
            else:
                offer[UNIT_PRICE] = int(minp + (maxp - minp) * r + 0.5)
            return tuple(offer)
        # just end the negotiation if I need nothing
        return None

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def _needs(self, partner, t):
        # find my needs today
        if self.awi.is_first_level:
            total_needs = self.awi.needed_sales
        elif self.awi.is_last_level:
            total_needs = self.awi.needed_supplies
        else:
            total_needs = self.production_level * self.awi.n_lines
        # estimate future needs
        if self.is_consumer(partner):
            total_needs += (
                self.production_level * self.awi.n_lines * (t - self.awi.current_step)
            )
            total_needs -= self.awi.total_sales_until(t)
        else:
            total_needs += (
                self.production_level * self.awi.n_lines * (self.awi.n_steps - t - 1)
            )
            total_needs -= self.awi.total_supplies_between(t, self.awi.n_steps - 1)
        # subtract already signed contracts
        return int(total_needs)

class CoyoteAgent(SimpleAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.other_stock = None

    def respond(self, negotiator_id, state, source=""):
        # accept any quantity needed at a good price
        offer = state.current_offer
        partner = self.get_nmi(negotiator_id)
        self.other_stock = current_inventory(partner)
        return (
            ResponseType.ACCEPT_OFFER
            if self.is_needed(negotiator_id, offer)
            and self.is_good_price(negotiator_id, offer, state)
            else ResponseType.REJECT_OFFER
        )



