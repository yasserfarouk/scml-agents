import random
from itertools import product

import numpy as np
import pandas as pd


class SellBidding:
    def __init__(
        self,
        current_step,
        initiator,
        eagerness,
        neediness,
        params,
        valid_bounds,
        utility_func,
        use_bid_func=True,
    ):
        self.current_step = current_step
        self.initiator = initiator
        self.output_catalog_price = params["output_catalog_price"]
        self.traded_price = params["traded_price"]
        self.utility_func = utility_func
        self.eagerness = eagerness
        self.neediness = neediness
        self.concess_rates = np.linspace(1, 0.8 - max(0, self.neediness) / 10, 100)
        self.t_bounds = params["t_bounds"]
        self.q_bounds = params["q_bounds"]
        self.p_bounds = params["p_bounds"]

        self.t_range = np.arange(self.t_bounds[0], self.t_bounds[1] + 1)
        self.q_range = np.arange(self.q_bounds[0], self.q_bounds[1] + 1)
        self.p_range = np.arange(self.p_bounds[0], self.p_bounds[1] + 1)

        if len(self.q_range) > 100:
            self.q_range = np.unique(
                np.linspace(self.q_bounds[0], self.q_bounds[1] + 1, 100).round()
            )

        self.valid_bounds = valid_bounds
        self.valid_q_range = np.arange(valid_bounds[0][0], valid_bounds[0][1] + 1)
        self.valid_t_range = np.arange(valid_bounds[1][0], valid_bounds[1][1] + 1)
        self.valid_p_range = np.arange(valid_bounds[2][0], valid_bounds[2][1] + 1)

        self.alpha = params["alpha"]
        self.beta = params["beta"]

        self.bid_func = self.get_good_bid if use_bid_func else self.generate_random_bid

    def generate_offerable_set(self):
        # defining reference points for points for offerable set
        self.center = np.mean([self.q_bounds, self.p_bounds], axis=1)
        self.r_1 = np.array([self.q_range[0], self.p_range[0]])
        self.r_2 = np.array([self.q_range[0], self.p_range[-1]])
        self.r_3 = np.array([self.q_range[-1], self.p_range[-1]])
        self.r_4 = np.array([self.q_range[-1], self.p_range[0]])
        self.r_a = -self.alpha * self.r_2 + (1 + self.alpha) * self.center
        self.r_b = self.beta * self.r_4 + (1 - self.beta) * self.center

        self.M_alpha = [self.r_1, self.r_a, self.r_3]
        self.M_beta = [self.r_1, self.r_b, self.r_3]
        self.M_zero = [self.r_1, self.r_3]

        bids = list(product(self.q_range, self.p_range))
        bid_df = pd.DataFrame(
            np.array([bids, list(map(self.find_bid_zone, bids))], dtype="object").T,
            columns=["bid", "zone"],
        )
        bid_df["utility"] = bid_df["bid"].apply(self.utility_func)
        self.bid_dict = {
            zone: bids.values for zone, bids in bid_df.groupby("zone")["bid"]
        }
        arr = np.array(list(bid_df["bid"]))
        bid_df["q"] = arr[:, 0]
        bid_df["p"] = arr[:, 1]
        bid_df.index = pd.MultiIndex.from_tuples(bid_df.bid)
        self.bid_df = bid_df

        self.offerable_zones = ["M_alpha", "M_beta", "M_zero"]
        self.acceptable_zones = self.offerable_zones + ["above_M_alpha"]

        self.offerable_bids = self.bid_df[
            (self.bid_df["zone"].isin(self.offerable_zones))
            & (self.bid_df["q"].isin(self.valid_q_range))
            & (self.bid_df["p"].isin(self.valid_p_range))
        ]
        self.offerable_bids.sort_values("utility", ascending=False, inplace=True)
        self.max_utility = self.offerable_bids["utility"].max()

        return self.offerable_bids.shape[0] > 0  # to control offer availability

    def find_bid_zone(self, bid):  # finds the zone where bid places
        q, p = bid

        d1 = (q - self.M_alpha[0][0]) * (self.M_alpha[1][1] - self.M_alpha[0][1]) - (
            p - self.M_alpha[0][1]
        ) * (self.M_alpha[1][0] - self.M_alpha[0][0])
        d2 = (q - self.M_alpha[1][0]) * (self.M_alpha[2][1] - self.M_alpha[1][1]) - (
            p - self.M_alpha[1][1]
        ) * (self.M_alpha[2][0] - self.M_alpha[1][0])
        d3 = (q - self.M_zero[0][0]) * (self.M_zero[1][1] - self.M_zero[0][1]) - (
            p - self.M_zero[0][1]
        ) * (self.M_zero[1][0] - self.M_zero[0][0])
        d4 = (q - self.M_beta[0][0]) * (self.M_beta[1][1] - self.M_beta[0][1]) - (
            p - self.M_beta[0][1]
        ) * (self.M_beta[1][0] - self.M_beta[0][0])
        d5 = (q - self.M_beta[1][0]) * (self.M_beta[2][1] - self.M_beta[1][1]) - (
            p - self.M_beta[1][1]
        ) * (self.M_beta[2][0] - self.M_beta[1][0])

        if d3 < 0:
            if d1 >= 0 and d2 >= 0:
                return "M_alpha"
            elif d1 < 0 or d2 < 0:
                return "above_M_alpha"
        elif d3 > 0:
            if d4 <= 0 and d5 <= 0:
                return "M_beta"
            elif d4 > 0 or d5 > 0:
                return "below_M_beta"
        else:
            return "M_zero"

    def evaluate(self, opponent_bid, nego_relative_time, opponent_behaviour_params):
        q, t, p = opponent_bid

        o_delivery_time = np.ceil(np.mean(opponent_behaviour_params["t_bounds"]))
        o_delivery_delay = o_delivery_time - self.current_step
        o_delivery_time = (
            o_delivery_time if o_delivery_delay >= 1 else o_delivery_time + 1
        )

        preferred_delivery_time = (
            self.valid_t_range[0] if nego_relative_time <= 0.5 else o_delivery_time
        )

        opponent_bid_zone = self.find_bid_zone((q, p))
        oppo_bid_utility = self.utility_func(opponent_bid)

        if self.q_bounds[0] <= q and q <= self.q_bounds[1] and self.p_bounds[0] <= p:
            if opponent_bid_zone == "above_M_alpha" and self.t_bounds[0] <= t:
                return (True, None)  # Accept

            elif self.eagerness >= 0.5:
                my_bid = self.generate_responsive_bid(
                    opponent_bid,
                    nego_relative_time,
                    o_delivery_time,
                    opponent_behaviour_params,
                )

            else:
                my_bid = self.bid_func(nego_relative_time, preferred_delivery_time)
        else:
            my_bid = self.bid_func(nego_relative_time, preferred_delivery_time)

        if self.neediness > 0:
            more_q = min(int(my_bid[0] * (1 + self.neediness)), self.valid_q_range[-1])
            my_bid = (more_q, my_bid[1], my_bid[2])

        my_bid_utility = self.utility_func(my_bid)

        if p >= self.p_bounds[0] and q <= self.q_bounds[1]:
            if abs(my_bid_utility) > 1e-12 and (
                nego_relative_time > 0.75
                and (my_bid_utility - oppo_bid_utility) / my_bid_utility
                <= max(0, self.neediness) / 10
            ):
                return (True, my_bid)  # Accept

            elif my_bid_utility < oppo_bid_utility:
                return (True, my_bid)  # Accept

        return (False, my_bid)

    def generate_responsive_bid(
        self, opponent_bid, nego_relative_time, delivery_time, opponent_behaviour_params
    ):
        opponent_concession = (
            opponent_behaviour_params["concession_rate"]
            if nego_relative_time >= 0.25
            else 0
        )
        time_concession = (
            1
            if nego_relative_time < 0.75
            else self.concess_rates[int(len(self.concess_rates) * nego_relative_time)]
        )
        restricted_q = (
            self.q_bounds
            if nego_relative_time <= 0.25
            else opponent_behaviour_params["q_bounds"]
        )
        upper_p = (
            self.traded_price
            if nego_relative_time >= 0.9 and self.traded_price in self.valid_p_range
            else self.valid_p_range[1]
        )

        oppo_bid_utility = self.utility_func(opponent_bid)

        delta = (
            (self.max_utility - oppo_bid_utility)
            * time_concession
            * (1 + max(0, opponent_concession))
        )

        offer_set = self.offerable_bids.loc[
            (oppo_bid_utility <= self.offerable_bids.utility)
            & (self.offerable_bids.utility <= oppo_bid_utility + delta)
            & (restricted_q[0] <= self.offerable_bids.q)
            & (restricted_q[1] * 1.5 >= self.offerable_bids.q)
            & (self.offerable_bids.p <= upper_p)
        ]

        if not isinstance(offer_set, pd.Series) and offer_set.shape[0] > 0:
            bid = self.bid_func(nego_relative_time, delivery_time, offer_set)
            return bid

        elif isinstance(offer_set, pd.Series):
            bid = offer_set["bid"]
            bid = (int(bid[0]), int(delivery_time), int(bid[1]))
            return bid

        # generate a bid indifferent to opponent's bid
        bid = self.bid_func(nego_relative_time, delivery_time)
        return bid

    def get_good_bid(self, nego_relative_time, delivery_time, offer_set=None):
        """
        Based on needines indicator and remaning negotiation time, a bid to be acceptable for opponent is generated.
        """

        if type(offer_set) == type(None) or len(offer_set) == 0:
            offer_set = self.offerable_bids

        offer_set.sort_values("utility", ascending=False, inplace=True)

        len_bids = offer_set.shape[0]

        if self.neediness > 0:  # sell need
            good_bids = offer_set.iloc[int(len_bids * self.neediness / 2) :]
        else:
            good_bids = offer_set

        if not isinstance(good_bids, pd.Series):
            # Added by yasser for the unlikely case that all weights are negative
            if "utility" not in good_bids.columns or good_bids.utility.sum() <= 0:
                subset_good_bids = good_bids.loc[
                    random.choices(good_bids.bid.values, k=min(len(good_bids), 5))
                ]
            else:
                subset_good_bids = good_bids.loc[
                    random.choices(
                        good_bids.bid.values,
                        weights=good_bids.utility,
                        k=min(len(good_bids), 5),
                    )
                ]
            subset_good_bids.sort_values("utility", ascending=False, inplace=True)
            bid = subset_good_bids.iloc[
                max(0, int(len(subset_good_bids) * nego_relative_time) - 1)
            ]["bid"]
        else:
            bid = good_bids["bid"]

        bid = (int(bid[0]), int(delivery_time), int(bid[1]))

        return bid

    def generate_random_bid(
        self, nego_relative_time, delivery_time, offer_set=None
    ):  # NOT USED
        if type(offer_set) == type(None):
            offer_set = self.offerable_bids

        bid = offer_set.loc[
            random.choices(offer_set.bid.values, weights=offer_set.utility, k=1)[0],
            "bid",
        ]
        bid = (int(bid[0]), int(delivery_time), int(bid[1]))
        return bid


class BuyBidding:
    def __init__(
        self,
        current_step,
        initiator,
        eagerness,
        neediness,
        params,
        valid_bounds,
        utility_func,
        use_bid_func=True,
    ):
        self.current_step = current_step
        self.initiator = initiator
        self.input_catalog_price = params["input_catalog_price"]
        self.traded_price = params["traded_price"]
        self.utility_func = utility_func

        self.eagerness = eagerness
        self.neediness = neediness
        self.concess_rates = np.linspace(1, 0.8 - abs(min(0, self.neediness)) / 10, 100)
        self.reasonable_q = params["reasonable_q"]
        self.prod_cap = params["prod_cap"]
        self.t_bounds = params["t_bounds"]
        self.q_bounds = params["q_bounds"]
        self.p_bounds = params["p_bounds"]

        self.t_range = np.arange(self.t_bounds[0], self.t_bounds[1] + 1)  # NOT USED YET
        self.q_range = np.arange(self.q_bounds[0], self.q_bounds[1] + 1)
        self.p_range = np.arange(self.p_bounds[0], self.p_bounds[1] + 1)
        if len(self.q_range) > 100:
            self.q_range = np.unique(
                np.linspace(self.q_bounds[0], self.q_bounds[1] + 1, 100).round()
            )

        self.valid_bounds = valid_bounds
        self.valid_q_range = np.arange(valid_bounds[0][0], valid_bounds[0][1] + 1)
        self.valid_t_range = np.arange(valid_bounds[1][0], valid_bounds[1][1] + 1)
        self.valid_p_range = np.arange(valid_bounds[2][0], valid_bounds[2][1] + 1)

        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.bid_func = self.get_good_bid if use_bid_func else self.generate_random_bid

    def generate_offerable_set(self):
        self.center = np.mean([self.q_bounds, self.p_bounds], axis=1)
        self.r_1 = np.array([self.q_range[0], self.p_range[0]])
        self.r_2 = np.array([self.q_range[0], self.p_range[-1]])
        self.r_3 = np.array([self.q_range[-1], self.p_range[-1]])
        self.r_4 = np.array([self.q_range[-1], self.p_range[0]])
        self.r_a = -self.alpha * self.r_1 + (1 + self.alpha) * self.center
        self.r_b = self.beta * self.r_3 + (1 - self.beta) * self.center

        self.M_alpha = [self.r_2, self.r_a, self.r_4]
        self.M_beta = [self.r_2, self.r_b, self.r_4]
        self.M_zero = [self.r_2, self.r_4]

        bids = list(product(self.q_range, self.p_range))
        bid_df = pd.DataFrame(
            np.array([bids, list(map(self.find_bid_zone, bids))], dtype="object").T,
            columns=["bid", "zone"],
        )
        bid_df["utility"] = bid_df["bid"].apply(self.utility_func)
        self.bid_dict = {
            zone: bids.values for zone, bids in bid_df.groupby("zone")["bid"]
        }
        arr = np.array(list(bid_df["bid"]))
        bid_df["q"] = arr[:, 0]
        bid_df["p"] = arr[:, 1]
        bid_df.index = pd.MultiIndex.from_tuples(bid_df.bid)
        self.bid_df = bid_df

        self.offerable_zones = ["M_alpha", "M_beta", "M_zero"]
        self.acceptable_zones = self.offerable_zones + ["below_M_alpha"]

        self.offerable_bids = self.bid_df[
            (self.bid_df["zone"].isin(self.offerable_zones))
            & (self.bid_df["q"].isin(self.valid_q_range))
            & (self.bid_df["p"].isin(self.valid_p_range))
        ]
        self.offerable_bids.sort_values("utility", ascending=False, inplace=True)
        self.max_utility = self.offerable_bids["utility"].max()

        return self.offerable_bids.shape[0] > 0

    def find_bid_zone(self, bid):
        q, p = bid

        d1 = (q - self.M_alpha[0][0]) * (self.M_alpha[1][1] - self.M_alpha[0][1]) - (
            p - self.M_alpha[0][1]
        ) * (self.M_alpha[1][0] - self.M_alpha[0][0])
        d2 = (q - self.M_alpha[1][0]) * (self.M_alpha[2][1] - self.M_alpha[1][1]) - (
            p - self.M_alpha[1][1]
        ) * (self.M_alpha[2][0] - self.M_alpha[1][0])
        d3 = (q - self.M_zero[0][0]) * (self.M_zero[1][1] - self.M_zero[0][1]) - (
            p - self.M_zero[0][1]
        ) * (self.M_zero[1][0] - self.M_zero[0][0])
        d4 = (q - self.M_beta[0][0]) * (self.M_beta[1][1] - self.M_beta[0][1]) - (
            p - self.M_beta[0][1]
        ) * (self.M_beta[1][0] - self.M_beta[0][0])
        d5 = (q - self.M_beta[1][0]) * (self.M_beta[2][1] - self.M_beta[1][1]) - (
            p - self.M_beta[1][1]
        ) * (self.M_beta[2][0] - self.M_beta[1][0])

        if d3 > 0:
            if d1 <= 0 and d2 <= 0:
                return "M_alpha"
            elif d1 > 0 or d2 > 0:
                return "below_M_alpha"
        elif d3 < 0:
            if d4 >= 0 and d5 >= 0:
                return "M_beta"
            elif d4 < 0 or d5 < 0:
                return "above_M_beta"
        else:
            return "M_zero"

    def evaluate(self, opponent_bid, nego_relative_time, opponent_behaviour_params):
        q, t, p = opponent_bid
        o_delivery_time = np.ceil(np.mean(opponent_behaviour_params["t_bounds"]))
        o_delivery_delay = o_delivery_time - self.current_step

        preferred_delivery_time = (
            o_delivery_time
            if nego_relative_time >= 0.5 and o_delivery_time <= self.t_bounds[1]
            else np.mean(self.valid_t_range)
        )

        opponent_bid_zone = self.find_bid_zone((q, p))
        oppo_bid_utility = self.utility_func(opponent_bid)

        if self.q_bounds[0] <= q and q <= self.q_bounds[1] and p <= self.p_bounds[1]:
            if opponent_bid_zone == "below_M_alpha" and t <= self.t_bounds[1]:
                return (True, None)  # Accept

            elif self.eagerness >= 0.5 and opponent_bid_zone in self.acceptable_zones:
                my_bid = self.generate_responsive_bid(
                    opponent_bid,
                    nego_relative_time,
                    o_delivery_time,
                    opponent_behaviour_params,
                )

            else:
                my_bid = self.bid_func(nego_relative_time, preferred_delivery_time)
        else:
            my_bid = self.bid_func(nego_relative_time, preferred_delivery_time)

        if self.neediness < 0:
            more_q = min(
                int(my_bid[0] * (1 + abs(self.neediness))), self.valid_q_range[-1]
            )
            my_bid = (more_q, my_bid[1], my_bid[2])

        my_bid_utility = self.utility_func(my_bid)

        if p <= self.p_bounds[1] and q <= self.q_bounds[1]:
            if (
                nego_relative_time > 0.75
                and (my_bid_utility - oppo_bid_utility) / my_bid_utility
                < abs(min(0, self.neediness)) / 10
                and t <= self.t_bounds[1]
            ):
                return (True, my_bid)  # Accept

            elif my_bid_utility < oppo_bid_utility and t <= self.t_bounds[1]:
                return (True, my_bid)  # Accept

        return (False, my_bid)  # Reject

    def generate_responsive_bid(
        self, opponent_bid, nego_relative_time, delivery_time, opponent_behaviour_params
    ):
        opponent_concession = (
            opponent_behaviour_params["concession_rate"]
            if nego_relative_time >= 0.25
            else 0
        )
        time_concession = (
            1
            if nego_relative_time < 0.75
            else self.concess_rates[int(len(self.concess_rates) * nego_relative_time)]
        )
        restricted_q = (
            self.q_bounds
            if nego_relative_time <= 0.25
            else opponent_behaviour_params["q_bounds"]
        )
        lower_p = (
            self.traded_price
            if nego_relative_time >= 0.9 and self.traded_price in self.valid_p_range
            else self.valid_p_range[0]
        )

        oppo_bid_utility = self.utility_func(opponent_bid)

        delta = (
            (self.max_utility - oppo_bid_utility)
            * time_concession
            * (1 + max(0, opponent_concession))
        )

        offer_set = self.offerable_bids.loc[
            (oppo_bid_utility <= self.offerable_bids.utility)
            & (self.offerable_bids.utility <= oppo_bid_utility + delta)
            & (self.reasonable_q * 0.75 <= self.offerable_bids.q)
            & (self.reasonable_q * 1.25 >= self.offerable_bids.q)
            & (lower_p <= self.offerable_bids.p)
            # (restricted_q[0] <= self.offerable_bids.q) & (restricted_q[1] >= self.offerable_bids.q)
        ]

        if not isinstance(offer_set, pd.Series) and offer_set.shape[0] > 0:
            bid = self.bid_func(nego_relative_time, delivery_time, offer_set)
            return bid

        elif isinstance(offer_set, pd.Series):
            bid = offer_set["bid"]
            bid = (int(bid[0]), int(delivery_time), int(bid[1]))
            return bid

        # generate a bid indifferent to opponent' bid
        bid = self.bid_func(nego_relative_time, delivery_time)
        return bid

    def get_good_bid(self, nego_relative_time, delivery_time, offer_set=None):
        if type(offer_set) == type(None) or len(offer_set) == 0:
            offer_set = self.offerable_bids

        offer_set.sort_values("utility", ascending=False, inplace=True)

        len_bids = offer_set.shape[0]

        if self.neediness < 0:  # buy need
            abs_neediness = abs(self.neediness)
            good_bids = offer_set.iloc[int(len_bids * abs_neediness / 2) :]
        else:
            good_bids = offer_set

        if not isinstance(good_bids, pd.Series):
            subset_good_bids = good_bids.loc[
                random.choices(
                    good_bids["bid"].values,
                    weights=good_bids["utility"],
                    k=min(len(good_bids), 5),
                )
            ]
            subset_good_bids.sort_values("utility", ascending=False, inplace=True)
            bid = subset_good_bids.iloc[
                max(0, int(len(subset_good_bids) * nego_relative_time) - 1)
            ]["bid"]
        else:
            bid = good_bids["bid"]

        self.prod_cap
        bid = (int(bid[0]), int(delivery_time), int(bid[1]))

        return bid

    def generate_random_bid(
        self, nego_relative_time, delivery_time, offer_set=None
    ):  # NOT USED
        if type(offer_set) == type(None):
            offer_set = self.offerable_bids

        bid = offer_set.loc[
            random.choices(offer_set.bid.values, weights=offer_set.utility, k=1)[0],
            "bid",
        ]
        bid = (int(bid[0]), int(delivery_time), int(bid[1]))

        return bid
