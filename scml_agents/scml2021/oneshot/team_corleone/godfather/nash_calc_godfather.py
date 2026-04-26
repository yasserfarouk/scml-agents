from scml.oneshot import *

from .offer import Offer


class NashCalcGodfather:
    """NashCalc, adapted to the Godfather interface"""

    def __init__(self, my_ufun, opp_ufun):
        self.ufun = my_ufun
        self.opp_ufun = opp_ufun
        self.max_quant = self.ufun.offer_space.max_q

        self.pareto_frontier = self.calc_pareto_frontier()
        self.nbp = self.calc_nash_point()

        self.range1 = self.my_max(0) - self.my_min(0)
        self.range2 = self.my_max(1) - self.my_min(1)

    def price_range(self):
        return self.ufun.offer_space.min_p, self.ufun.offer_space.max_p

    def calc_nash_point(self):
        own_disagree_util, opp_disagree_util = self.calc_utility(0, 0)

        max_value = float("-inf")
        curr_max = None
        for point in self.pareto_frontier:
            nash_value = (point[2] - own_disagree_util) * (point[3] - opp_disagree_util)
            if nash_value > max_value:
                curr_max = point
                max_value = nash_value

        return curr_max

    def calc_utility(self, quant, price):
        o = Offer(price, quant)
        own = self.ufun(o)
        opp = self.opp_ufun(o)
        return own, opp

    def calc_pareto_frontier(self):
        """Returns pareto frontier as a list of [q, p, own_util, opp_util] lists"""
        pareto_frontier = []
        utils = {}
        for o in self.ufun.outcome_space.outcome_set():
            p = o.price
            q = o.quantity
            a, b = self.calc_utility(q, p)
            utils[(p, q)] = [a, b]

        for o in self.ufun.outcome_space.outcome_set():
            p = o.price
            q = o.quantity
            own_util, opp_util = self.calc_utility(q, p)
            pareto = True
            for combo in utils.values():
                a = combo[0]
                b = combo[1]
                if (a >= own_util and b > opp_util) or (a > own_util and b >= opp_util):
                    pareto = False
            if pareto:
                pareto_frontier.append([q, p, own_util, opp_util])

        return pareto_frontier

    def my_max(self, bit):
        max_val = float("-inf")
        for element in self.pareto_frontier:
            if element[2 + bit] > max_val:
                max_val = element[2 + bit]
        return max_val

    def my_min(self, bit):
        min_val = float("inf")
        for element in self.pareto_frontier:
            if element[2 + bit] < min_val:
                min_val = element[2 + bit]
        return min_val

    def frontier_offers(self):
        return [Offer(e[1], e[0]) for e in self.pareto_frontier]

    def nash_point_offer(self):
        return Offer(self.nbp[1], self.nbp[0])

    def nash_point_my_util(self):
        return self.nbp[2]
