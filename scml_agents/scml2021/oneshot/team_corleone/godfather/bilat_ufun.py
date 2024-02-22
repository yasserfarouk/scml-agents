from typing import Dict, List

import numpy as np
from scipy.optimize import curve_fit
from scml.oneshot import OneShotUFun
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE

from .negotiation_history import BilateralHistory
from .offer import Offer
from .spaces import *

# import matplotlib
# from matplotlib import animation
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm


class BilatUFun:
    def __init__(self, offer_space: OfferSpace):
        self.offer_space = offer_space
        self.move_space = MoveSpace(offer_space)
        self.outcome_space = OutcomeSpace(offer_space)

    def __call__(self, outcome: Outcome) -> float:
        raise NotImplementedError

    def vis(self, fig, ax):
        normalize = matplotlib.colors.Normalize(vmin=-100, vmax=200)
        x, y, z = [], [], []
        for o in self.outcome_space.outcome_set():
            x.append(o.price)
            y.append(o.quantity)
            z.append(self(o))
        plot = ax.scatter(x, y, c=z, norm=normalize)
        ax.set_title("utility by p/q")
        ax.set_xlabel("price")
        ax.set_ylabel("quant")
        fig.colorbar(plot, ax=ax)
        with open("visualizations/ufuns.txt", "a") as f:
            f.write(f"{x}\n{y}\n{z}\n\n")

    def vis_3d(self, fig, ax):
        X = np.arange(self.offer_space.min_p, self.offer_space.max_p, 1)
        Y = np.arange(self.offer_space.min_q, self.offer_space.max_q, 1)
        X, Y = np.meshgrid(X, Y)
        vec_func = np.vectorize(lambda x, y: self(Offer(x, y)))
        Z = vec_func(X, Y)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title("utility by p/q")
        ax.set_xlabel("price")
        ax.set_ylabel("quant")
        ax.set_zlabel("util")

    def poly_fit_4th_correct(self) -> List[float]:
        """Returns a list of more parameters for a 4th-order polynomial surface fitting the ufun"""
        offers = self.offer_space.offer_set()
        x = [o.price for o in offers]
        y = [o.quantity for o in offers]
        z = [self(o) for o in offers]

        def poly_4c(xy, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):
            x = xy[:, 0]
            y = xy[:, 1]
            return (
                a * x**4
                + b * x**3 * y
                + c * x**3
                + d * x**2 * y**2
                + e * x**2 * y
                + f * x**2
                + g * x * y**3
                + h * x * y**2
                + i * x * y
                + j * x
                + k * y**4
                + l * y**3
                + m * y**2
                + n * y
                + o
            )

        xy = np.zeros((len(x), 2))
        xy[:, 0] = x
        xy[:, 1] = y
        popt, pcov = curve_fit(poly_4c, xy, z)

        # mse = sum([(true - approx) ** 2 for true, approx in zip(z, poly_4c(xy, *popt))]) / len(z)
        # print("mean sq err poly fit 4th", mse)
        return popt

    def poly_fit_3rd_correct(self) -> List[float]:
        """Returns a list of more parameters for a 3rd-order polynomial surface fitting the ufun"""
        offers = self.offer_space.offer_set()
        x = [o.price for o in offers]
        y = [o.quantity for o in offers]
        z = [self(o) for o in offers]

        def poly_3c(xy, a, b, c, d, e, f, g, h, i, j):
            x = xy[:, 0]
            y = xy[:, 1]
            return (
                a * x**3
                + b * x**2 * y
                + c * x**2
                + d * x * y**2
                + e * x * y
                + f * x
                + g * y**3
                + h * y**2
                + i * y
                + j
            )

        xy = np.zeros((len(x), 2))
        xy[:, 0] = x
        xy[:, 1] = y
        popt, pcov = curve_fit(poly_3c, xy, z)

        # mse = sum([(true - approx) ** 2 for true, approx in zip(z, poly_3c(xy, *popt))]) / len(z)
        # print("mean sq err poly fit 3rd", mse)
        return popt

    def poly_fit_3rd(self) -> List[float]:
        """Returns a list of more parameters for a 3rd-order polynomial surface fitting the ufun"""
        offers = self.offer_space.offer_set()
        x = [o.price for o in offers]
        y = [o.quantity for o in offers]
        z = [self(o) for o in offers]

        def poly_3(xy, a, b, c, d, e, f, g, h, i, j, k, l, m, n):
            x = xy[:, 0]
            y = xy[:, 1]
            return (
                a * x**3 * y**3
                + b * x**3 * y**2
                + c * x**3 * y
                + d * x**3
                + e * x**2 * y**3
                + f * x**2 * y**2
                + e * x**2 * y
                + f * x**2
                + g * x * y**3
                + h * x * y**2
                + i * x * y
                + j * x
                + k * y**3
                + l * y**2
                + m * y
                + n
            )

        xy = np.zeros((len(x), 2))
        xy[:, 0] = x
        xy[:, 1] = y
        popt, pcov = curve_fit(poly_3, xy, z)

        # mse = sum([(true - approx) ** 2 for true, approx in zip(z, poly_3(xy, *popt))]) / len(z)
        # print("mean sq err poly fit 3rd", mse)
        return popt

    def poly_fit(self) -> List[float]:
        """Returns a list of 9 parameters for a 2nd-order polynomial surface fitting the ufun"""
        offers = self.offer_space.offer_set()
        x = [o.price for o in offers]
        y = [o.quantity for o in offers]
        z = [self(o) for o in offers]

        def poly_2(xy, a, b, c, d, e, f, g, h, i):
            x = xy[:, 0]
            y = xy[:, 1]
            return (
                a * x**2 * y**2
                + b * x**2 * y
                + c * x**2
                + d * x * y**2
                + e * x * y
                + f * x
                + g * y**2
                + h * y
                + i
            )

        xy = np.zeros((len(x), 2))
        xy[:, 0] = x
        xy[:, 1] = y
        popt, pcov = curve_fit(poly_2, xy, z)

        # mse = sum([(true - approx) ** 2 for true, approx in zip(z, poly_2(xy, *popt))]) / len(z)
        # print("mean sq err poly fit", mse)
        return popt

    def __eq__(self, other):
        os1 = self.offer_space.offer_set()
        os2 = other.offer_space.offer_set()
        if os1 != os2:
            return False
        return all(abs(self(o) - other(o)) < 0.01 for o in os1)


class BilatUFunDummy(BilatUFun):
    """Represents a null/degenerate ufun. Really just a wrapper for an outcome space."""

    def __call__(self):
        raise RuntimeError("tried to call dummy ufun")


class BilatUFunUniform(BilatUFun):
    """Uniform ufun"""

    def __init__(self, offer_space: OfferSpace):
        super().__init__(offer_space)

    def __call__(self, outcome: Outcome) -> float:
        return 0


class BilatUFunMarginalTable(BilatUFun):
    """Builds a marginal ufun from a lookup table, where the lookup table util_table
    maps outcomes (including the reserve) to utilities"""

    def __init__(self, offer_space: OfferSpace, util_table: Dict[Outcome, float]):
        self._util_table = util_table
        super().__init__(offer_space)

    def __call__(self, outcome: Outcome) -> float:
        return self._util_table[outcome] - self._util_table[self.offer_space.reserve]

    def __repr__(self) -> str:
        return repr(self._util_table)


class BilatUFunAvg(BilatUFun):
    def __init__(
        self,
        offer_space: OfferSpace,
        opp_history: BilateralHistory,
        expected_exog_quant=None,
    ):
        super().__init__(offer_space)

        self.level = 1 if opp_history.world_info.my_level == 0 else 0
        self.n_partners = (
            opp_history.world_info.n_competitors + 1
        )  # include opp, which is not its own partner

        self.expected_prod_cost = (self.level + 1) * 5.5 * 2.5
        self.expected_disp_cost = 0.1
        self.expected_shortfall_cost = 0.6

        self.expected_exog = {0: 8.9992004, 1: 9.02894599}[self.level]
        self.needed_exog = self.expected_exog / self.n_partners
        ex_pin = 1 if self.level == 0 else 0
        ex_pout = 0
        ex_qin = self.needed_exog if self.level == 0 else 0
        ex_qout = self.needed_exog if self.level == 1 else 0
        input_agent = True if self.level == 0 else False
        output_agent = False if self.level == 0 else True
        input_product = None
        self.scml_ufun = OneShotUFun(
            ex_pin=ex_pin,
            ex_qin=ex_qin,
            ex_pout=ex_pout,
            ex_qout=ex_qout,
            input_product=input_product,
            input_agent=input_agent,
            output_agent=output_agent,
            production_cost=self.expected_prod_cost,
            disposal_cost=self.expected_disp_cost,
            storage_cost=0.0,
            shortfall_penalty=self.expected_shortfall_cost,
            input_penalty_scale=None,
            output_penalty_scale=None,
            storage_penalty_scale=None,
            n_input_negs=1,
            n_output_negs=1,
            current_step=0,
            agent_id="",
            time_range=(0, 0),
        )

    def __call__(self, o: Offer):
        offer = [-1] * 3
        offer[TIME] = 1
        offer[UNIT_PRICE] = o.price
        offer[QUANTITY] = o.quantity
        result = self.scml_ufun(tuple(offer))
        return result

    # def __init__(self, offer_space: OfferSpace, opp_history: BilateralHistory, expected_exog_quant=None):
    #     # n.b.: opp_history and ufun have opposite POVs
    #     super().__init__(offer_space)

    #     self.level = 1 if opp_history.world_info.my_level == 0 else 0
    #     self.n_partners = opp_history.world_info.n_competitors + 1  # include opp, which is not its own partner

    #     self.expected_prod_cost = (self.level+1) * 5.5 * 2.5
    #     self.expected_disp_cost = 0.1
    #     self.expected_shortfall_cost = 0.6

    #     self.expected_exog = { 0: 8.9992004, 1: 9.02894599 }[self.level]

    # def __call__(self, o: Offer):
    #     needed_exog = self.expected_exog / self.n_partners
    #     costs = 0.0
    #     if self.level == 0:
    #         costs += self.expected_prod_cost * min(o.quantity, self.expected_exog)
    #     else:
    #         costs += self.expected_prod_cost * min(o.quantity, self.expected_exog)

    #     if o.quantity > needed_exog:
    #         diff = o.quantity - needed_exog
    #         if self.level == 0:
    #             costs += self.expected_shortfall_cost * diff * self.awi.trading_prices[1]
    #         else:
    #             costs += self.expected_disp_cost * diff * self.awi.trading_prices[1]
    #     else:
    #         diff = needed_exog - o.quantity
    #         if self.level == 0:
    #             costs += self.expected_shortfall_cost * diff * self.awi.trading_prices[1]
    #         else:
    #             costs += self.expected_disp_cost * diff * self.awi.trading_prices[1]

    #     if self.level == 0:
    #         return (o.price * min(o.quantity, self.expected_exog)) - costs  # TODO: revisit expected vs needed exog
    #     else:
    #         return 0 - (o.price * o.quantity) - costs
