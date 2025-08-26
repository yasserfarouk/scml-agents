import random
from scml.oneshot.agents.rand import OneShotSyncAgent
import numpy as np

from itertools import chain, combinations
from scml.oneshot import UNIT_PRICE, QUANTITY
from negmas import SAOResponse, ResponseType

__all__ = ["HoriYamaAgent"]


def make_random_price(n):
    # nは0~1の範囲の値
    # 値が大きいほど1に近づくように重みづけさせたもの
    xs = np.linspace(0, 1, 100)
    skew = (n - 0.5) * 2  # -1〜1 の範囲
    if abs(skew) < 1e-8:  # 厳密に中立にする
        weights = np.ones_like(xs)
    elif skew > 0:
        weights = np.power(xs - xs.min(), 1 + 2 * skew)
    else:
        weights = np.power(xs.max() - xs, 1 - 2 * skew)
    weights /= weights.sum()
    return np.random.choice(xs, p=weights)


def needs_parameter_sigmoid(needs, k=1.5, needs0=1, param_max=6.0, param_min=1.4):
    """
    needs: 必要量（1以上）
    k: シグモイドの傾き（大きいほど急激に切り替わる）
    needs0: シグモイドの中心（このneedsで中間値）
    param_max: needsが最小のときのパラメータ最大値
    param_min: needsが大きいときのパラメータ最小値（1.2以上にする）
    """
    s = 1 / (1 + np.exp(k * (needs - needs0)))
    param = param_min + (param_max - param_min) * s
    return param


def distribute(q: int, n: int) -> list[int]:
    """Distributes n values over m bins with at
    least one item per bin assuming q > n"""
    from numpy.random import choice
    from collections import Counter

    q = q * needs_parameter_sigmoid(q)
    q = int(np.ceil(q))  # 切り上げ
    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# BestSyncAgent1_4MarketSigmoidPrice
class HoriYamaAgent(OneShotSyncAgent):
    """An agent that distributes its needs over its partners randomly."""

    def distribute_needs(self) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # find suppliers and consumers still negotiating with me
            partner_ids = [_ for _ in all_partners if _ in self.negotiators.keys()]
            partners = len(partner_ids)

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partner_ids, [0] * partners)))
                continue

            # distribute my needs over my (remaining) partners.
            dist.update(dict(zip(partner_ids, distribute(needs, partners))))
        return dist

    def first_proposals(self):
        # just randomly distribute my needs over my partners (with best price for me).
        s, p = self._step_and_price(best_price=True)
        distribution = self.distribute_needs()
        d = {k: (q, s, p) if q > 0 else None for k, q in distribution.items()}
        return d

    def counter_all(self, offers, states):
        response = dict()
        # process for sales and supplies independently
        for needs, all_partners, issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            # get a random price
            price = self._step_and_price(best_price=False)[1]
            # find active partners
            partners = {_ for _ in all_partners if _ in offers.keys()}

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))
            best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                others = partners.difference(partner_ids)
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                if diff < best_diff:
                    best_diff, best_indx = diff, i
                if diff == 0:
                    break

            # If the best combination of offers is good enough, accept them and end all
            # other negotiations
            th = self._current_threshold(
                min([_.relative_time for _ in states.values()])
            )
            if best_diff <= th:
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in others}
                continue

            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            distribution = self.distribute_needs()
            response.update(
                {
                    k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                    if q == 0
                    else SAOResponse(
                        ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                    )
                    for k, q in distribution.items()
                }
            )
        return response

    def _current_threshold(self, r: float):
        mn, mx = 0, self.awi.n_lines // 2
        base_threshold = mn + (mx - mn) * (r**4.0)

        # 市場の人数比を取得
        n_competitors = self.awi.n_competitors
        seller = self.awi.is_first_level

        if seller:
            # 売り手の場合
            my_partners = len(self.awi.my_consumers)  # 買い手の数
            ratio = my_partners / (n_competitors + 1)  # 買い手/売り手の比率

            if ratio > 1.0:
                # 売り手市場（自分に有利）-> 閾値を厳しく（小さく）
                market_factor = 0.5 + 0.3 / (1 + ratio - 1.0)  # 0.5-0.8の範囲
            else:
                # 買い手市場（自分に不利）-> 閾値を緩く（大きく）
                market_factor = 1.0 + (1.0 - ratio) * 0.5  # 1.0-1.5の範囲
        else:
            # 買い手の場合
            my_partners = len(self.awi.my_suppliers)  # 売り手の数
            ratio = (
                (n_competitors + 1) / my_partners if my_partners > 0 else float("inf")
            )  # 買い手/売り手の比率

            if ratio > 1.0:
                # 買い手市場（自分に不利）-> 閾値を緩く（大きく）
                market_factor = 1.0 + (ratio - 1.0) * 0.3  # 1.0-1.3の範囲
            else:
                # 売り手市場（自分に有利）-> 閾値を厳しく（小さく）
                market_factor = 0.7 + ratio * 0.3  # 0.7-1.0の範囲

        return base_threshold * market_factor

    def _step_and_price(self, best_price=False, *, bias=0.3):
        """
        bias: 0   → 従来と同じ
            0.3 → 少し強気 (売り手は高め／買い手は安め)
            1.0 → かなり強気
        """
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin, pmax = issues[UNIT_PRICE].min_value, issues[UNIT_PRICE].max_value

        if best_price:
            return s, (pmax if seller else pmin)

        # ―― 市場状況 ――#
        n_competitors = self.awi.n_competitors
        if seller:
            ratio = len(self.awi.my_consumers) / (n_competitors + 1)
            mapped = ratio / (1 + ratio)  # 0-1 に正規化 (従来)
            mapped = mapped ** (1 - bias)  # ★指数で押し上げ
        else:
            ratio = (n_competitors + 1) / max(1, len(self.awi.my_suppliers))
            mapped = ratio / (1 + ratio)
            mapped = mapped ** (1 + bias)  # ★指数で押し下げ

        price = pmin + (pmax - pmin) * make_random_price(mapped)
        return s, round(price)


if __name__ == "__main__":
    import sys

    from .helpers.runner import run

    run(
        [HoriYamaAgent],
        sys.argv[1] if len(sys.argv) > 1 else "oneshot",
    )
