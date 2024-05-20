import random

from negmas import *
from negmas.sao import SAOResponse

from scml.common import distribute
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE
from scml.std.agent import StdSyncAgent

from typing import *
from itertools import combinations
import copy

__all__ = ["CautiousStdAgent"]
MAX_INVENTORY = 100
MIN_INVENTORY = 0


def distribute(
    q: int,
    n: int,
    *,
    mx: int | None = None,
    equal=False,
    concentrated=False,
    allow_zero=False,
    concentrated_idx: list[int] = [],
) -> list[int]:
    """Distributes q values over n bins.

    Args:
        q: Quantity to distribute
        n: number of bins to distribute q over
        mx: Maximum allowed per bin. `None` for no limit
        equal: Try to make the values in each bins as equal as possible
        concentrated: If true, will try to concentrate offers in few bins. `mx` must be passed in this case
        allow_zero: Allow some bins to be zero even if that is not necessary
    """
    from collections import Counter

    from numpy.random import choice

    q, n = int(q), int(n)

    if mx is not None and q > mx * n:
        q = mx * n

    if concentrated:
        assert mx is not None
        lst = [0] * n
        if not allow_zero:
            for i in range(min(q, n)):
                lst[i] = 1
        q -= sum(lst)
        if q == 0:
            random.shuffle(lst)
            return lst
        for i in range(n):
            q += lst[i]
            lst[i] = min(mx, q)
            q -= lst[i]
        concentrated_lst = sorted(lst, reverse=True)[: len(concentrated_idx)]
        for x in concentrated_lst:
            lst.remove(x)
        random.shuffle(lst)
        for i, x in zip(concentrated_idx, concentrated_lst):
            lst.insert(i, x)
        return lst

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst
    if q == n:
        return [1] * n
    if allow_zero:
        per = 0
    else:
        per = (q // n) if equal else 1
    q -= per * n
    r = Counter(choice(n, q))
    return [r.get(_, 0) + per for _ in range(n)]


def powerset(iterable):
    """冪集合"""
    from itertools import chain

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class CautiousStdAgent(StdSyncAgent):
    """An agent that distributes its needs over its partners randomly."""

    def __init__(
        self,
        *args,
        equal: bool = False,
        overordering_max_selling: float = 0.0,
        overordering_max_buying: float = 0.2,
        overordering_min: float = 0.0,
        overordering_exp: float = 0.4,
        mismatch_exp: float = 4.0,
        today_exp: float = 0.5,
        future_exp: float = 0.2,
        overmismatch_max_selling: float = 0,
        overmismatch_max_buying: float = 0.3,
        undermismatch_min_selling: float = -0.4,
        undermismatch_min_buying: float = -0.2,
        **kwargs,
    ):
        self.equal_distribution = equal
        self.overordering_max_selling = overordering_max_selling
        self.overordering_max_buying = overordering_max_buying
        self.overordering_min = overordering_min
        self.overordering_exp = overordering_exp
        self.mismatch_exp = mismatch_exp
        self.today_exp = today_exp
        self.future_exp = future_exp
        self.overmismatch_max_selling = overmismatch_max_selling
        self.overmismatch_max_buying = overmismatch_max_buying
        self.undermismatch_min_selling = undermismatch_min_selling
        self.undermismatch_min_buying = undermismatch_min_buying
        super().__init__(*args, **kwargs)

    def init(self):
        self.average_prices = [
            self.awi.catalog_prices[self.awi.my_input_product],
            self.awi.catalog_prices[self.awi.my_output_product],
        ]
        self.future_selling_contracts = []
        self.cost = self.awi.profile.cost
        self.opp_acc_prices = [
            {
                partner: [
                    self.awi.catalog_prices[self.awi.my_input_product],
                    self.awi.catalog_prices[self.awi.my_input_product],
                    self.awi.catalog_prices[self.awi.my_input_product],
                ]
                for partner in self.awi.my_suppliers
            },
            {
                partner: [
                    self.awi.catalog_prices[self.awi.my_output_product],
                    self.awi.catalog_prices[self.awi.my_output_product],
                    self.awi.catalog_prices[self.awi.my_output_product],
                ]
                for partner in self.awi.my_consumers
            },
        ]
        self.pre_inventory = [0, 0]
        self.p_min_for_selling = self.awi.catalog_prices[self.awi.my_output_product]
        self.p_max_for_buying = self.awi.catalog_prices[self.awi.my_input_product]

        self.total_agreed_quantity = {
            k: 0 for k in self.awi.my_consumers + self.awi.my_suppliers
        }

    def before_step(self):
        super().before_step()

        self.max_inventory = min(
            MAX_INVENTORY,
            (self.awi.n_steps - 1 - self.awi.current_step) * self.awi.n_lines / 2,
        )
        self.min_inventory = MIN_INVENTORY

        if self.awi.my_consumers == ["BUYER"]:
            self.p_max_for_buying = min(
                (
                    self.awi.current_exogenous_output_price
                    / self.awi.current_exogenous_output_quantity
                    if self.awi.current_exogenous_output_quantity != 0
                    else self.awi.catalog_prices[self.awi.my_output_product]
                )
                - self.awi.profile.cost
                - 0.1,
                self.awi.current_input_issues[UNIT_PRICE].max_value,
            )
        else:
            if sum(self.awi.current_inventory) > self.max_inventory:
                self.p_max_for_buying = self.awi.current_input_issues[
                    UNIT_PRICE
                ].min_value + 0.1 * (
                    self.awi.current_input_issues[UNIT_PRICE].max_value
                    - self.awi.current_input_issues[UNIT_PRICE].min_value
                )
            elif self.awi.current_inventory[0] >= self.awi.n_lines or sum(
                self.awi.current_inventory
            ) > sum(self.pre_inventory):
                self.p_max_for_buying = max(
                    self.p_max_for_buying * 0.8,
                    self.awi.current_input_issues[UNIT_PRICE].min_value
                    + 0.1
                    * (
                        self.awi.current_input_issues[UNIT_PRICE].max_value
                        - self.awi.current_input_issues[UNIT_PRICE].min_value
                    ),
                )
            elif (
                self.awi.current_step < 0.8 * self.awi.n_steps
                and sum(self.awi.current_inventory) == 0
                or sum(self.awi.current_inventory) < sum(self.pre_inventory)
            ):
                self.p_max_for_buying = min(
                    self.p_max_for_buying * 1.1,
                    self.awi.catalog_prices[self.awi.my_output_product]
                    - self.awi.profile.cost
                    - 0.1,
                )

        self.today_agreed_buying_contracts = []

        # print(f"=== day {self.awi.current_step} {self.id} ===")
        # print(f"inventory {self.awi.current_inventory}")
        # print(f"input need {max(self.awi.needed_supplies,0)}")

        # print("==============")
        # print("input issue (catalog price, trading price): ",self.awi.current_input_issues[UNIT_PRICE],self.awi.catalog_prices[self.awi.my_input_product],self.awi.trading_prices[self.awi.my_input_product])
        # print("output issue (catalog price, trading price): ",self.awi.current_output_issues[UNIT_PRICE],self.awi.catalog_prices[self.awi.my_output_product],self.awi.trading_prices[self.awi.my_output_product])
        # print("production cost: ",self.awi.profile.cost)
        # print("==============")
        # if self.awi.my_suppliers == ["SELLER"]:
        #     print(self.id, "inventory:",sum(self.awi.current_inventory),"exogenous contract:",self.awi.current_exogenous_input_price/self.awi.current_exogenous_input_quantity if self.awi.current_exogenous_input_quantity>0 else 0,"output issues:",self.awi.current_output_issues[UNIT_PRICE])
        #     print("       future_selling_contracts: ",[contract.agreement for contract in self.future_selling_contracts])

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        super().on_negotiation_success(contract, mechanism)
        q, t, p, is_seller = (
            contract.agreement["quantity"],
            contract.agreement["time"],
            contract.agreement["unit_price"],
            self.id == contract.annotation["seller"],
        )
        partner_id = (
            contract.annotation["buyer"] if is_seller else contract.annotation["seller"]
        )
        if is_seller and self.awi.current_step < t < self.awi.n_steps:
            self.future_selling_contracts.append(contract)
        if not is_seller and t == self.awi.current_step:
            self.today_agreed_buying_contracts.append(contract)
        self.opp_acc_prices[is_seller][partner_id][0] = min(
            self.opp_acc_prices[is_seller][partner_id][0], p
        )
        self.opp_acc_prices[is_seller][partner_id][-1] = max(
            self.opp_acc_prices[is_seller][partner_id][-1], p
        )
        self.total_agreed_quantity[partner_id] += q

    def step(self):
        super().step()
        self.pre_inventory = list(self.awi.current_inventory)
        # 実行済みの契約と相手が破産した契約を削除しておく
        for contract in self.future_selling_contracts:
            if contract.agreement[
                "time"
            ] <= self.awi.current_step or self.awi.is_bankrupt(
                contract.annotation["buyer"]
            ):
                self.future_selling_contracts.remove(contract)

    def first_proposals(self):  # type: ignore
        # just randomly distribute my needs over my partners (with best price for me).
        # remaining partners get random future offers

        offers = {}
        unneeded = (
            None if not self.awi.allow_zero_quantity else (0, self.awi.current_step, 0)
        )

        # Buying (Propose to suppliers)
        # 出荷先がBUYERなら、現在の在庫でexogenous_output_quantityに対する不足分＋0~n_lines個(後半ほど減らす)の入荷を目指す
        if self.awi.my_consumers == ["BUYER"]:
            todays_input_needed = (
                self.awi.current_exogenous_output_quantity
                - (
                    min(self.awi.current_inventory_input, self.awi.n_lines)
                    + self.awi.current_inventory_output
                )
                + int(
                    self.awi.n_lines
                    * (1 - (self.awi.current_step + 1) / self.awi.n_steps)
                )
            )
        # 出荷先がBUYER以外なら、前半はn_lines個、後半は最終日に近づくほど在庫数を0に近づけていく
        else:
            if sum(self.awi.current_inventory) > self.max_inventory:
                todays_input_needed = 0
            elif (
                self.awi.my_suppliers != ["SELLER"]
                and self.awi.current_input_issues[UNIT_PRICE].min_value
                + self.awi.profile.cost
                > self.awi.trading_prices[self.awi.my_output_product]
            ):
                todays_input_needed = 0
            else:
                todays_input_needed = max(
                    self.awi.n_lines - self.awi.current_inventory_input, 0
                )

        # valid_suppliers, not_valid_suppliers = [],[]
        # for k in self.awi.my_suppliers:
        #     if self.awi.is_bankrupt(k) or (self.awi.current_step > min(self.awi.n_steps*0.5, 50) and self.total_agreed_quantity[k] == 0):
        #         not_valid_suppliers.append(k)
        #     else:
        #         valid_suppliers.append(k)

        # if self.awi.my_consumers == ["BUYER"] and len(valid_suppliers) == 0:
        #     valid_suppliers, not_valid_suppliers = [],[]
        #     for k in self.awi.my_suppliers:
        #         if self.awi.is_bankrupt(k):
        #             not_valid_suppliers.append(k)
        #         else:
        #             valid_suppliers.append(k)

        # not_valid_suppliers = [k for k in self.awi.my_suppliers if self.awi.is_bankrupt(k)]
        # valid_suppliers = list(set(self.awi.my_suppliers).difference(not_valid_suppliers))

        valid_suppliers = self.awi.my_suppliers

        if len(valid_suppliers) > 0:
            distribution = dict(
                zip(
                    valid_suppliers,
                    distribute(
                        todays_input_needed, len(valid_suppliers), mx=self.awi.n_lines
                    ),
                )
            )
            offers |= {
                k: (q, self.awi.current_step, self.best_price(k)) if q > 0 else unneeded
                for k, q in distribution.items()
            }
        # offers |= {k:unneeded for k in not_valid_suppliers}

        # distribution = dict(zip(
        #     self.awi.my_suppliers,
        #     distribute(todays_input_needed,len(self.awi.my_suppliers),mx=self.awi.n_lines)
        # ))
        # offers |= {
        #     k: (q, self.awi.current_step, self.best_price(k)) if q>0 else unneeded
        #     for k, q in distribution.items()
        # }

        # Selling (Propose to consumer)
        # 在庫をできるだけ早く売り切る
        # remained_consumers = self.awi.my_consumers.copy()
        remained_consumers = [
            k for k in self.awi.my_consumers if not self.awi.is_bankrupt(k)
        ]
        secured_output = sum(
            [
                contract.agreement["quantity"]
                for contract in self.future_selling_contracts
            ]
        )
        for t in range(self.awi.current_step, self.awi.n_steps):
            if self.awi.my_consumers == ["BUYER"]:
                break
            # todays_output_needed = min(max(self.awi.needed_sales - secured_output,0),self.awi.n_lines)
            todays_output_needed = max(self.awi.needed_sales - secured_output, 0)
            if todays_output_needed <= self.awi.n_lines or t == self.awi.n_steps - 1:
                # 一日で販売しきれる場合はすべての相手に1個以上提案する
                distribution = dict(
                    zip(
                        remained_consumers,
                        distribute(todays_output_needed, len(remained_consumers)),
                    )
                )
            else:
                # 一日で販売しきれない場合は一人の相手にn_lines個を販売し、残りを翌日以降他の相手に販売する
                todays_output_needed = self.awi.n_lines
                concentrated_ids = sorted(
                    remained_consumers,
                    key=lambda x: self.total_agreed_quantity[x],
                    reverse=True,
                )
                distribution = dict(
                    zip(
                        remained_consumers,
                        distribute(
                            todays_output_needed,
                            len(remained_consumers),
                            mx=self.awi.n_lines,
                            concentrated=True,
                            concentrated_idx=[
                                i
                                for i, p in enumerate(remained_consumers)
                                if p in concentrated_ids
                            ],
                            allow_zero=True,
                        ),
                    )
                )
            # offers |= {k: (q, t, self.best_price(k)) if q > 0 else unneeded for k,q in distribution.items()}
            # if self.awi.my_suppliers == ["SELLER"] and sum(self.awi.current_inventory) > self.max_inventory:
            #     offers |= {k: (q, t, self.awi.current_output_issues[UNIT_PRICE].min_value) if q > 0 else unneeded for k,q in distribution.items()}
            # else:
            offers |= {
                k: (q, t, self.awi.current_output_issues[UNIT_PRICE].max_value)
                if q > 0
                else unneeded
                for k, q in distribution.items()
            }

            remained_consumers = [
                k
                for k in remained_consumers
                if k not in distribution.keys() or distribution[k] <= 0
            ]
            secured_output += todays_output_needed
            if len(remained_consumers) == 0 or todays_output_needed == 0:
                break

        offers |= {k: unneeded for k in remained_consumers}
        offers |= {
            k: unneeded for k in set(self.awi.my_consumers).difference(offers.keys())
        }

        # print("Exogenous Contracts:",self.awi.current_exogenous_input_quantity,self.awi.current_exogenous_output_quantity)
        # print("inventories",self.awi.current_inventory)
        # print("input offers:",{k:v for k,v in offers.items() if k in self.awi.my_suppliers})
        # print("output offers:",{k:v for k,v in offers.items() if k in self.awi.my_consumers})
        # print("future offers:",[contract.agreement["quantity"] for contract in self.future_selling_contracts])

        return offers

    def counter_all(self, offers, states):
        today_offers = {
            k: v for k, v in offers.items() if v[TIME] == self.awi.current_step
        }

        unneeded_response = (
            SAOResponse(ResponseType.END_NEGOTIATION, None)
            if not self.awi.allow_zero_quantity
            else SAOResponse(ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0))
        )
        response = {}

        ################### Decide Response for Buying #######################

        # 供給先がBUYERの場合は組合せによる数量重視
        if self.awi.my_consumers == ["BUYER"]:
            valid_suppliers = [_ for _ in self.awi.my_suppliers if _ in offers.keys()]
            today_partners = [
                _ for _ in self.awi.my_suppliers if _ in today_offers.keys()
            ]
            today_partners = set(today_partners)
            plist = list(powerset(today_partners))[::-1]
            price_good_plist = [
                ps
                for ps in plist
                if len(ps) > 0
                and max([offers[p][UNIT_PRICE] for p in ps])
                * self.awi.current_exogenous_output_quantity
                < self.awi.current_exogenous_output_price
                - self.awi.profile.cost * self.awi.current_exogenous_output_quantity
            ]
            if len(price_good_plist) > 0:
                plist = price_good_plist
            plus_best_diff, plus_best_indx = float("inf"), -1
            minus_best_diff, minus_best_indx = -float("inf"), -1
            best_diff, best_indx = float("inf"), -1
            todays_input_needed = (
                self.awi.current_exogenous_output_quantity
                - (
                    min(self.awi.current_inventory_input, self.awi.n_lines)
                    + self.awi.current_inventory_output
                )
                # + int(self.awi.n_lines * (1 - self.awi.current_step / self.awi.n_steps))
            )

            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = offered - todays_input_needed
                if diff >= 0:  # 必要以上の量のとき
                    if diff < plus_best_diff:
                        plus_best_diff, plus_best_indx = diff, i
                    elif diff == plus_best_diff:
                        if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(
                            offers[p][UNIT_PRICE] for p in plist[plus_best_indx]
                        ):
                            plus_best_diff, plus_best_indx = diff, i
                if diff <= 0:  # 必要量に満たないとき
                    if diff > minus_best_diff:
                        minus_best_diff, minus_best_indx = diff, i
                    elif diff == minus_best_diff:
                        if (
                            diff < 0 and len(partner_ids) < len(plist[minus_best_indx])
                        ):  # アクセプトする不足分をCounterOfferできる相手の数が多かったら更新
                            minus_best_diff, minus_best_indx = diff, i
                        elif (
                            diff == 0
                            or diff < 0
                            and len(partner_ids) == len(plist[minus_best_indx])
                        ):
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(
                                offers[p][UNIT_PRICE] for p in plist[minus_best_indx]
                            ):
                                minus_best_diff, minus_best_indx = diff, i

            th_min, th_max = self._allowed_mismatch(
                min(state.relative_time for state in states.values()), False
            )
            if th_min <= minus_best_diff or plus_best_diff <= th_max:
                if plus_best_diff <= th_max:
                    best_diff, best_indx = plus_best_diff, plus_best_indx
                else:
                    best_diff, best_indx = minus_best_diff, minus_best_indx

                response |= {
                    p: SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])
                    for p in plist[best_indx]
                }

                remained_suppliers = set(valid_suppliers).difference(plist[best_indx])
                if best_diff < 0 and len(remained_suppliers) > 0:
                    concentrated_ids = sorted(
                        remained_suppliers,
                        key=lambda x: self.total_agreed_quantity[x],
                        reverse=True,
                    )
                    if (
                        len(concentrated_ids) > 0
                        and self.awi.current_step > self.awi.n_steps * 0.5
                    ):
                        distribution = dict(
                            zip(
                                remained_suppliers,
                                distribute(
                                    -best_diff,
                                    len(remained_suppliers),
                                    mx=min(
                                        max(
                                            offers[concentrated_ids[0]][QUANTITY],
                                            int(-best_diff / len(remained_suppliers))
                                            + 1,
                                        ),
                                        self.awi.n_lines,
                                    ),
                                    concentrated=True,
                                    concentrated_idx=[
                                        i
                                        for i, p in enumerate(remained_suppliers)
                                        if p in concentrated_ids
                                    ],
                                ),
                            )
                        )
                    else:
                        distribution = dict(
                            zip(
                                remained_suppliers,
                                distribute(-best_diff, len(remained_suppliers)),
                            )
                        )

                    response |= {
                        k: SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (
                                q,
                                self.awi.current_step,
                                self.buy_price(
                                    states[k].relative_time,
                                    self.awi.current_input_issues[UNIT_PRICE].min_value,
                                    self.p_max_for_buying,
                                    True,
                                ),
                            ),
                        )
                        if q > 0
                        else unneeded_response
                        for k, q in distribution.items()
                    }
        # 供給先がBUYER以外の場合は安ければ買い、高ければ買わない
        # 翌日以降入荷の契約はリジェクトする
        # 最終日は(供給先がBUYERのとき以外は)買わない
        elif self.awi.current_step == self.awi.n_steps - 1:
            response |= {
                partner_id: unneeded_response
                for partner_id in offers.keys()
                if partner_id in self.awi.my_suppliers
            }
        # 取引しても損しかしない(最安値で仕入れて市場価格で販売しても利益が出ない)ときは買わない
        elif (
            self.awi.current_input_issues[UNIT_PRICE].min_value + self.awi.profile.cost
            > self.awi.trading_prices[self.awi.my_output_product]
        ):
            response |= {
                partner_id: unneeded_response
                for partner_id in offers.keys()
                if partner_id in self.awi.my_suppliers
            }
        else:
            buying_offers = {
                partner_id: offer
                for partner_id, offer in offers.items()
                if partner_id in self.awi.my_suppliers
            }
            if sum(self.awi.current_inventory) > self.max_inventory:
                response |= {k: unneeded_response for k in buying_offers.keys()}
            else:
                input_secured = 0
                for partner_id, offer in sorted(
                    buying_offers.items(), key=lambda x: x[1][UNIT_PRICE]
                ):
                    if offer[UNIT_PRICE] <= self.buy_price(
                        states[partner_id].relative_time,
                        self.awi.current_input_issues[UNIT_PRICE].min_value,
                        self.p_max_for_buying,
                    ):
                        if offer[TIME] == self.awi.current_step:
                            response[partner_id] = SAOResponse(
                                ResponseType.ACCEPT_OFFER, offer
                            )
                        else:
                            response[partner_id] = SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (
                                    offer[QUANTITY],
                                    self.awi.current_step,
                                    offer[UNIT_PRICE],
                                ),
                            )
                    else:
                        response[partner_id] = (
                            SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (
                                    offer[QUANTITY],
                                    self.awi.current_step,
                                    self.buy_price(
                                        states[partner_id].relative_time,
                                        self.awi.current_input_issues[
                                            UNIT_PRICE
                                        ].min_value,
                                        self.p_max_for_buying,
                                    ),
                                ),
                            )
                            if offer[QUANTITY] > 0
                            else unneeded_response
                        )
                    input_secured += offer[QUANTITY]
                    if (
                        sum(self.awi.current_inventory) + input_secured
                        > self.max_inventory
                    ):
                        break

        ################### Decide Response for Selling #######################

        # consumersとの交渉は、とにかく(前日までに入荷した)在庫を売り切れる組み合わせを重視する、在庫を超える量の契約は結ばない
        selling_offers = {
            partner_id: offer
            for partner_id, offer in offers.items()
            if partner_id in self.awi.my_consumers
        }
        plist = list(powerset(selling_offers.keys()))[::-1]
        plist = [
            ps
            for ps in plist
            if sum(
                [
                    self.awi.current_step <= offers[p][TIME] < self.awi.n_steps
                    for p in ps
                ]
            )
            == len(ps)
        ]

        # output productの市場価格が安くなってきてしまっていたら、市場価格より高い価格でしか売らないようにして市場価格の引き上げを図る
        if (
            self.awi.my_suppliers != ["SELLER"]
            and sum(self.awi.current_inventory) < self.max_inventory
        ):
            price_good_plist = []
            if (
                self.awi.current_exogenous_input_quantity > 0
                and self.awi.current_exogenous_input_price
                / self.awi.current_exogenous_input_quantity
                + self.awi.profile.cost
                > self.awi.trading_prices[self.awi.my_output_product]
                or self.awi.current_exogenous_input_quantity == 0
                and self.awi.current_input_issues[UNIT_PRICE].min_value
                + self.awi.profile.cost
                > self.awi.trading_prices[self.awi.my_output_product]
            ):
                for ps in plist:
                    total_price, total_quantity = 0, 0
                    for k in ps:
                        total_price += (
                            selling_offers[k][UNIT_PRICE] * selling_offers[k][QUANTITY]
                        )
                        total_quantity += selling_offers[k][QUANTITY]
                    if (
                        total_price
                        > self.awi.trading_prices[self.awi.my_output_product]
                        * total_quantity
                    ):
                        price_good_plist.append(ps)
            if len(price_good_plist) > 0:
                plist = price_good_plist

        best_diff, best_indx = float("inf"), -1  # このbest_diffは絶対値
        secured_output = 0
        for contract in self.future_selling_contracts:
            if self.awi.is_bankrupt(contract.annotation["buyer"]):
                self.future_selling_contracts.remove(contract)
                continue
            secured_output += contract.agreement["quantity"]
        todays_agreed_input = 0
        for contract in self.today_agreed_buying_contracts:
            if self.awi.is_bankrupt(contract.annotation["seller"]):
                self.today_agreed_buying_contracts.remove(contract)
                continue
            todays_agreed_input += contract.agreement["quantity"]
        todays_output_needed = min(
            max(
                self.awi.needed_sales
                + int(todays_agreed_input / (self.awi.level + 1))
                - secured_output,
                0,
            ),
            self.awi.n_lines,
        )
        for i, partner_ids in enumerate(plist):
            offered = sum(offers[p][QUANTITY] for p in partner_ids)
            diff = offered - todays_output_needed
            if -best_diff < diff <= 0 or (
                -diff == best_diff
                and sum(offers[p][UNIT_PRICE] for p in partner_ids)
                > sum(offers[p][UNIT_PRICE] for p in plist[best_indx])
            ):
                best_diff, best_indx = -diff, i

        response |= {
            p: SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])
            for p in plist[best_indx]
        }

        remained_consumers = list(set(selling_offers).difference(plist[best_indx]))
        # remained_output_needs = best_diff
        remained_output_needs = (
            self.awi.needed_sales
            + int(todays_agreed_input / (self.awi.level + 1))
            - secured_output
            - sum(offers[p][QUANTITY] for p in plist[best_indx])
        )

        for t in range(self.awi.current_step, self.awi.n_steps):
            if len(remained_consumers) == 0:
                break
            if remained_output_needs == 0:
                response |= {k: unneeded_response for k in remained_consumers}
                break
            tmp_output_needs = (
                min(
                    remained_output_needs,
                    self.awi.n_lines - (todays_output_needed - best_diff),
                )
                if t == self.awi.current_step
                else min(remained_output_needs, self.awi.n_lines)
            )
            if tmp_output_needs == 0:
                continue

            # offer_quantities = dict(zip(remained_partners, distribute(tmp_output_needs,len(remained_partners)))) if remained_output_needs==tmp_output_needs\
            #     else dict(zip(remained_partners, distribute(tmp_output_needs,len(remained_partners),mx=self.awi.n_lines,concentrated=True,allow_zero=True)))
            concentrated_ids = sorted(
                remained_consumers,
                key=lambda x: self.total_agreed_quantity[x],
                reverse=True,
            )
            distribution = dict(
                zip(
                    remained_consumers,
                    distribute(
                        tmp_output_needs,
                        len(remained_consumers),
                        # mx=min(max(offers[concentrated_ids[0]][QUANTITY],int(tmp_output_needs/len(remained_consumers))+1),self.awi.n_lines),# これだとremained_output_needs!=tmp_output_needのときに相手を使い切ってしまう可能性あり
                        mx=self.awi.n_lines,
                        concentrated=True,
                        concentrated_idx=[
                            i
                            for i, p in enumerate(remained_consumers)
                            if p in concentrated_ids
                        ],
                        allow_zero=(
                            False if remained_output_needs == tmp_output_needs else True
                        ),
                    ),
                )
            )

            # if self.awi.my_suppliers == ["SELLER"] and sum(self.awi.current_inventory) > self.max_inventory:
            #     response |= {
            #         k:SAOResponse(ResponseType.REJECT_OFFER,(
            #             q, t, self.awi.current_output_issues[UNIT_PRICE].min_value
            #         )) if q>0 else unneeded_response for k,q in distribution.items()
            #     }
            # else:
            mn = (
                max(
                    self.awi.current_exogenous_input_price
                    / self.awi.current_exogenous_input_quantity
                    + self.awi.profile.cost
                    + 1,
                    self.awi.current_output_issues[UNIT_PRICE].min_value,
                )
                if self.awi.current_exogenous_input_quantity > 0
                else max(
                    self.awi.catalog_prices[self.awi.my_input_product]
                    + self.awi.profile.cost
                    + 1,
                    self.awi.current_output_issues[UNIT_PRICE].min_value,
                )
            )
            mx = mn + (self.awi.current_output_issues[UNIT_PRICE].max_value - mn) / 2
            response |= {
                k: SAOResponse(
                    ResponseType.REJECT_OFFER,
                    (
                        q,
                        t,
                        max(
                            self.sell_price(states[k].relative_time, mn, mx),
                            selling_offers[k][UNIT_PRICE],
                        )
                        if sum(self.awi.current_inventory) < self.max_inventory
                        and self.awi.my_suppliers != ["SELLER"]
                        and self.awi.current_input_issues[UNIT_PRICE].min_value
                        + self.awi.profile.cost
                        > self.awi.trading_prices[self.awi.my_output_product]
                        else self.sell_price(states[k].relative_time, mn, mx),
                    ),
                )
                if q > 0
                else unneeded_response
                for k, q in distribution.items()
            }
            remained_output_needs -= tmp_output_needs
            remained_consumers = list(
                set(remained_consumers).difference(
                    {k for k, q in distribution.items() if q > 0}
                )
            )

        # print(f"day {self.awi.current_step} {self.id}")
        # print("Exogenous Contracts:",self.awi.current_exogenous_input_quantity,self.awi.current_exogenous_output_quantity)
        # print("inventories:",self.awi.current_inventory)
        # print("future selling contracts:",[contract.agreement["quantity"] for contract in self.future_selling_contracts])
        # print("Input Responses:",{k:v for k,v in response.items() if k in self.awi.my_suppliers})
        # print("Output Responses:",{k:v for k,v in response.items() if k in self.awi.my_consumers})

        return response

    def _allowed_mismatch(self, r: float, is_selling: True):
        undermismatch_min = self.awi.n_lines * (
            self.undermismatch_min_selling
            if is_selling
            else self.undermismatch_min_buying
        )
        overmismatch_max = self.awi.n_lines * (
            self.overmismatch_max_selling
            if is_selling
            else self.overmismatch_max_buying
        )
        return undermismatch_min * ((1 - r) ** self.mismatch_exp), overmismatch_max * (
            r ** (1 / self.mismatch_exp)
        )

    def _overordering_fraction(self, t: float):
        mn, mx = self.overordering_min, self.overordering_max
        return mx - (mx - mn) * (t**self.overordering_exp)

    def is_supplier(self, negotiator_id):
        return negotiator_id in self.awi.my_suppliers

    def is_consumer(self, negotiator_id):
        return negotiator_id in self.awi.my_consumers

    def best_price(self, partner_id):
        """Best price for a negotiation today"""
        if partner_id == "SELLER":
            try:
                return int(
                    self.awi.current_exogenous_input_price
                    / self.awi.current_exogenous_input_quantity
                )
            except ZeroDivisionError:
                return 0
        if partner_id == "BUYER":
            try:
                return int(
                    self.awi.current_exogenous_output_price
                    / self.awi.current_exogenous_output_quantity
                )
            except ZeroDivisionError:
                return 0
        return (
            self.awi.current_output_issues[UNIT_PRICE].max_value
            if self.is_consumer(partner_id)
            else self.awi.current_input_issues[UNIT_PRICE].min_value
        )

    def good_price(self, partner_id, today: bool):
        """A good price to use"""
        nmi = self.get_nmi(partner_id)
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        if self.is_supplier(partner_id):
            return self.buy_price(nmi.state.relative_time, mn, mx, today=today)
        return self.sell_price(
            self.get_nmi(partner_id).state.relative_time, mn, mx, today=today
        )

    def buy_price(self, t: float, mn: float, mx: float, today: bool = True) -> float:
        """Return a good price to buy at"""
        e = self.today_exp if today else self.future_exp
        return max(mn, min(mx, int(mn + (mx - mn) * (t**e) + 0.5)))

    def sell_price(self, t: float, mn: float, mx: float, today: bool = True) -> float:
        """Return a good price to sell at"""
        e = self.today_exp if today else self.future_exp
        if not today:
            mn = mn + self.fmin * (mx - mn)
        return max(mn, min(mx, int(0.5 + mx - (mx - mn) * (t**e))))

    def good2buy(self, p: float, t: float, mn, mx, today: bool = True):
        """Is p a good price to buy at?"""
        if not today:
            mx = mx - self.fmin * (mx - mn)
        return p - 0.0001 <= self.buy_price(t, mn, mx, today)

    def good2sell(self, p: float, t: float, mn, mx, today: bool = True):
        """Is p a good price to sell at?"""
        return p + 0.0001 >= self.sell_price(t, mn, mx, today)


###########################################################################
###########################################################################


class AgentVSC2024(StdSyncAgent):
    """An agent that distributes its needs over its partners randomly."""

    def __init__(
        self,
        *args,
        equal: bool = False,
        overordering_max_selling: float = 0.0,
        overordering_max_buying: float = 0.2,
        overordering_min: float = 0.0,
        overordering_exp: float = 0.4,
        mismatch_exp: float = 4.0,
        today_exp: float = 0.5,
        future_exp: float = 0.2,
        # mismatch_max: float = 0.3,
        overmismatch_max_selling: float = 0,
        overmismatch_max_buying: float = 0.2,
        undermismatch_min_selling: float = -0.4,
        undermismatch_min_buying: float = -0.3,
        **kwargs,
    ):
        self.equal_distribution = equal
        self.overordering_max_selling = overordering_max_selling
        self.overordering_max_buying = overordering_max_buying
        self.overordering_min = overordering_min
        self.overordering_exp = overordering_exp
        self.mismatch_exp = mismatch_exp
        self.today_exp = today_exp
        self.future_exp = future_exp
        # self.mismatch_max = mismatch_max
        self.overmismatch_max_selling = overmismatch_max_selling
        self.overmismatch_max_buying = overmismatch_max_buying
        self.undermismatch_min_selling = undermismatch_min_selling
        self.undermismatch_min_buying = undermismatch_min_buying
        super().__init__(*args, **kwargs)

    def init(self):
        self.average_prices = [
            self.awi.catalog_prices[self.awi.my_input_product],
            self.awi.catalog_prices[self.awi.my_output_product],
        ]
        self.step_input_quantities = [0] * self.awi.n_steps
        self.step_output_quantities = [0] * self.awi.n_steps
        self.cost = self.awi.profile.cost
        self.opp_acc_prices = [
            {
                partner: [
                    self.awi.catalog_prices[self.awi.my_input_product],
                    self.awi.catalog_prices[self.awi.my_input_product],
                    self.awi.catalog_prices[self.awi.my_input_product],
                ]
                for partner in self.awi.my_suppliers
            },
            {
                partner: [
                    self.awi.catalog_prices[self.awi.my_output_product],
                    self.awi.catalog_prices[self.awi.my_output_product],
                    self.awi.catalog_prices[self.awi.my_output_product],
                ]
                for partner in self.awi.my_consumers
            },
        ]
        self.inventory = 0
        self.max_inventory = MAX_INVENTORY
        self.min_inventory = MIN_INVENTORY
        self.p_min_for_selling = self.awi.catalog_prices[self.awi.my_output_product]
        self.p_max_for_buying = self.awi.catalog_prices[self.awi.my_input_product]

    def before_step(self):
        super().before_step()
        if self.awi.my_suppliers == ["SELLER"]:
            self.step_input_quantities[self.awi.current_step] += (
                self.awi.current_exogenous_input_quantity
            )
        if self.awi.my_consumers == ["BUYER"]:
            self.step_output_quantities[self.awi.current_step] += (
                self.awi.current_exogenous_output_quantity
            )

        # print(f"=== day {self.awi.current_step} {self.id} ===")
        # print(f"inputs {self.step_input_quantities}")
        # print(f"output {self.step_output_quantities}")
        # print(f"inventory {self.awi.current_inventory}")
        # print(f"input need {max(self.awi.needed_supplies,0)}")
        # print(f"output need {sum(self.step_input_quantities)-sum(self.step_output_quantities)}, {max(self.awi.needed_sales,0)}")

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        super().on_negotiation_success(contract, mechanism)
        q, t, p, is_seller = (
            contract.agreement["quantity"],
            contract.agreement["time"],
            contract.agreement["unit_price"],
            self.id == contract.annotation["seller"],
        )
        partner_id = (
            contract.annotation["buyer"] if is_seller else contract.annotation["seller"]
        )
        if partner_id in ["BUEYE", "SELLER"]:
            pass  # print(partner_id)
        if is_seller:
            if self.awi.current_step <= t < self.awi.n_steps:
                self.step_output_quantities[t] += q
        else:
            if self.awi.current_step <= t < self.awi.n_steps:
                self.step_input_quantities[t] += q
        self.opp_acc_prices[is_seller][partner_id][0] = min(
            self.opp_acc_prices[is_seller][partner_id][0], p
        )
        self.opp_acc_prices[is_seller][partner_id][-1] = max(
            self.opp_acc_prices[is_seller][partner_id][-1], p
        )

    def on_contract_executed(self, contract) -> None:
        super().on_contract_executed(contract)
        pass  # print("contract executed")

    def step(self):
        super().step()
        # print("step")

    def first_proposals(self):  # type: ignore
        # just randomly distribute my needs over my partners (with best price for me).
        # remaining partners get random future offers

        offers = {}
        unneeded = (
            None if not self.awi.allow_zero_quantity else (0, self.awi.current_step, 0)
        )

        # Buying (Propose to suppliers)
        # 当日にn_lines個入荷できるよう提案する
        if self.awi.my_consumers == ["SELLER"]:
            todays_input_needed = max(
                self.awi.current_exogenous_output_quantity
                - sum(self.awi.current_inventory)
            )
        else:
            todays_input_needed = (
                self.awi.n_lines
                if self.awi.current_step < self.awi.n_steps / 2
                else int(
                    self.awi.n_lines
                    * (2 - 2 * self.awi.current_step / self.awi.n_steps)
                )
            )
            todays_input_needed = max(
                todays_input_needed - self.awi.current_inventory_input, 0
            )
        distribution = dict(
            zip(
                self.awi.my_suppliers,
                distribute(
                    todays_input_needed, len(self.awi.my_suppliers), mx=self.awi.n_lines
                ),
            )
        )
        offers = {
            k: (q, self.awi.current_step, self.best_price(k)) if q > 0 else unneeded
            for k, q in distribution.items()
        }

        # Selling (Propose to consumer)
        # 在庫をできるだけ早く売り切る
        remained_consumers = self.awi.my_consumers.copy()
        secured_output = 0
        for t in range(self.awi.current_step, self.awi.n_steps):
            if t == self.awi.current_step:
                todays_output_needed = max(self.awi.needed_sales, 0)
            elif t == self.awi.current_step + 1:
                if self.awi.my_suppliers == ["SELLER"]:
                    continue  # 前日の入荷分はexogenous_contractなのですでに出荷のofferを出し済み
                else:
                    todays_output_needed = self.step_input_quantities[t - 1]
            else:
                todays_output_needed = self.step_input_quantities[t - 1]

            distribution = dict(
                zip(
                    remained_consumers,
                    distribute(
                        todays_output_needed,
                        len(remained_consumers),
                        mx=self.awi.n_lines,
                        concentrated=True,
                        allow_zero=True,
                    ),
                )
            )
            offers |= {
                k: (q, t, self.best_price(k)) for k, q in distribution.items() if q > 0
            }
            remained_consumers = [
                p for p in remained_consumers if p not in offers.keys()
            ]

            """todays_output_needed = sum(self.step_input_quantities[:t]) - sum(self.step_output_quantities) - secured_output # t-1日目までの総入荷量ー総出荷量
            distribution = dict(zip(remained_consumers, distribute(todays_output_needed,len(remained_consumers),mx=self.awi.n_lines,concentrated=True)))
            offers |= {
                k: (q, t, self.best_price(k)) for k, q in distribution.items() if q > 0
            }
            remained_consumers = [p for p in remained_consumers if p not in offers.keys()]
            secured_output += todays_output_needed"""

        offers |= {k: unneeded for k in remained_consumers}

        # print(f"==={self.id}, day {self.awi.current_step}===")
        # print("step_input_quantitiies: ",self.step_input_quantities)
        # print("step_output_quantities: ",self.step_output_quantities)
        # print("offers for suppliers: ",{k:v for k,v in offers.items() if k in self.awi.my_suppliers})
        # print("offers for consumers: ",{k:v for k,v in offers.items() if k in self.awi.my_consumers})
        # print("current_inventory", self.awi.current_inventory,self.awi.current_inventory_input,self.awi.current_inventory_output)
        return offers

    def counter_all(self, offers, states):
        today_offers = {
            k: v for k, v in offers.items() if v[TIME] == self.awi.current_step
        }

        tmp_step_input_quantities = copy.deepcopy(self.step_input_quantities)
        tmp_step_output_quantities = copy.deepcopy(self.step_output_quantities)

        unneeded_response = (
            SAOResponse(ResponseType.END_NEGOTIATION, None)
            if not self.awi.allow_zero_quantity
            else SAOResponse(ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0))
        )
        response = {}

        ################### Decide Response for Buying #######################

        # 供給先がSELLERの場合は組合せによる数量重視
        if self.awi.my_consumers == ["SELLER"]:
            today_partners = [
                _ for _ in self.awi.my_suppliers if _ in today_offers.keys()
            ]
            random.shuffle(today_partners)
            today_partners = set(today_partners)
            plist = list(powerset(today_partners))[::-1]
            plist = [
                ps
                for ps in plist
                if max([offers[p][UNIT_PRICE] for p in ps])
                < self.awi.trading_prices[self.awi.my_output_product]
                - self.awi.profile.cost
            ]
            plus_best_diff, plus_best_indx = float("inf"), -1
            minus_best_diff, minus_best_indx = -float("inf"), -1
            best_diff, best_indx = float("inf"), -1

            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = offered - max(self.awi.needed_supplies, 0)
                if diff >= 0:  # 必要以上の量のとき
                    if diff < plus_best_diff:
                        plus_best_diff, plus_best_indx = diff, i
                    elif diff == plus_best_diff:
                        if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(
                            offers[p][UNIT_PRICE] for p in plist[plus_best_indx]
                        ):
                            plus_best_diff, plus_best_indx = diff, i
                if diff <= 0:  # 必要量に満たないとき
                    if diff > minus_best_diff:
                        minus_best_diff, minus_best_indx = diff, i
                    elif diff == minus_best_diff:
                        if (
                            diff < 0 and len(partner_ids) < len(plist[minus_best_indx])
                        ):  # アクセプトする不足分をCounterOfferできる相手の数が多かったら更新
                            minus_best_diff, minus_best_indx = diff, i
                        elif diff == 0 or len(partner_ids) == len(
                            plist[minus_best_indx]
                        ):
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(
                                offers[p][UNIT_PRICE] for p in plist[minus_best_indx]
                            ):
                                minus_best_diff, minus_best_indx = diff, i

            th_min, th_max = self._allowed_mismatch(
                min(state.relative_time for state in states.values()), False
            )
            if th_min <= minus_best_diff or plus_best_diff <= th_max:
                if th_min <= minus_best_diff and plus_best_diff <= th_max:
                    if -minus_best_diff < plus_best_diff:
                        best_diff, best_indx = minus_best_diff, minus_best_indx
                    else:
                        best_diff, best_indx = plus_best_diff, plus_best_indx
                elif minus_best_diff < th_min and plus_best_diff <= th_max:
                    best_diff, best_indx = plus_best_diff, plus_best_indx
                else:
                    best_diff, best_indx = minus_best_diff, minus_best_indx

                response |= {
                    p: SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])
                    for p in plist[best_indx]
                }
                if best_diff < 0:
                    remained_todays_partners = today_partners.difference(
                        plist[best_indx]
                    )
                    offer_quantities = dict(
                        zip(
                            remained_todays_partners,
                            distribute(-best_diff, len(remained_todays_partners)),
                        )
                    )
                    t = min(_.relative_time for _ in states.values())
                    response |= {
                        k: SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (
                                q,
                                self.awi.current_step,
                                self.buy_price(
                                    t,
                                    self.awi.current_input_issues[UNIT_PRICE].min_value,
                                    self.p_max_for_buying,
                                    True,
                                ),
                            ),
                        )
                        if q > 0
                        else unneeded_response
                        for k, q in offer_quantities.items()
                    }
        # 供給先がSELLER以外の場合は安ければ買い、高ければ買わない
        else:
            buying_offers = {
                partner_id: offer
                for partner_id, offer in offers.items()
                if partner_id in self.awi.my_suppliers
            }
            for partner_id, offer in sorted(
                buying_offers.items(), key=lambda x: x[1][UNIT_PRICE]
            ):
                today = offer[TIME] == self.awi.current_step
                neg_step_relative = states[partner_id].relative_time
                if (
                    offer[TIME] < self.awi.current_step
                    or self.awi.n_steps <= offer[TIME]
                ):
                    response[partner_id] = unneeded_response
                    continue
                if offer[QUANTITY] + tmp_step_input_quantities[
                    offer[TIME]
                ] <= self.awi.n_lines and offer[UNIT_PRICE] <= self.buy_price(
                    neg_step_relative,
                    self.awi.current_input_issues[UNIT_PRICE].min_value,
                    self.p_max_for_buying,
                    today,
                ):
                    response[partner_id] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    tmp_step_input_quantities[offer[TIME]] += offer[QUANTITY]
                else:
                    q = max(
                        self.awi.n_lines - tmp_step_input_quantities[offer[TIME]], 0
                    )
                    response[partner_id] = (
                        SAOResponse(
                            ResponseType.REJECT_OFFER,
                            (
                                q,
                                offer[TIME],
                                self.buy_price(
                                    neg_step_relative,
                                    self.awi.current_input_issues[UNIT_PRICE].min_value,
                                    self.p_max_for_buying,
                                    today,
                                ),
                            ),
                        )
                        if q > 0
                        else unneeded_response
                    )

        # consumersとの交渉は、とにかく在庫を売り切れる組合せを重視する
        selling_offers = {
            partner_id: offer
            for partner_id, offer in offers.items()
            if partner_id in self.awi.my_consumers
        }
        plist = list(powerset(selling_offers.keys()))[::-1]
        best_diff, best_indx = float("inf"), -1
        for i, partner_ids in enumerate(plist):
            # tmp_offers = {p:offers[p] for p in partner_ids}
            offered = sum(offers[p][QUANTITY] for p in partner_ids)
            diff = offered - max(self.awi.needed_sales, 0)
            if -best_diff < diff <= 0 or (
                -diff == best_diff
                and sum(offers[p][UNIT_PRICE] for p in plist[best_indx])
                < sum(offers[p][UNIT_PRICE] for p in partner_ids)
            ):
                best_diff, best_indx = -diff, i

        response |= {
            p: SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])
            for p in plist[best_indx]
        }

        remained_partners = set(buying_offers).difference(plist[best_indx])
        offer_quantities = dict(
            zip(remained_partners, distribute(-best_diff, len(remained_partners)))
        )
        t = min(_.relative_time for _ in states.values())
        response |= {
            k: SAOResponse(
                ResponseType.REJECT_OFFER,
                (
                    q,
                    self.awi.current_step,
                    self.sell_price(
                        t,
                        self.p_min_for_selling,
                        self.awi.current_output_issues[UNIT_PRICE].max_value,
                    ),
                ),
            )
            if q > 0
            else unneeded_response
            for k, q in offer_quantities.items()
        }

        return response

    def _allowed_mismatch(self, r: float, is_selling: True):
        undermismatch_min = self.awi.n_lines * (
            self.undermismatch_min_selling
            if is_selling
            else self.undermismatch_min_buying
        )
        overmismatch_max = self.awi.n_lines * (
            self.overmismatch_max_selling
            if is_selling
            else self.overmismatch_max_buying
        )
        return undermismatch_min * ((1 - r) ** self.mismatch_exp), overmismatch_max * (
            r**self.mismatch_exp
        )

    def _overordering_fraction(self, t: float):
        mn, mx = self.overordering_min, self.overordering_max
        return mx - (mx - mn) * (t**self.overordering_exp)

    def is_supplier(self, negotiator_id):
        return negotiator_id in self.awi.my_suppliers

    def is_consumer(self, negotiator_id):
        return negotiator_id in self.awi.my_consumers

    def best_price(self, partner_id):
        """Best price for a negotiation today"""
        if partner_id == "SELLER":
            try:
                return int(
                    self.awi.current_exogenous_input_price
                    / self.awi.current_exogenous_input_quantity
                )
            except ZeroDivisionError:
                return 0
        if partner_id == "BUYER":
            try:
                return int(
                    self.awi.current_exogenous_output_price
                    / self.awi.current_exogenous_output_quantity
                )
            except ZeroDivisionError:
                return 0
        # issue = self.get_nmi(partner_id).issues[UNIT_PRICE]
        return (
            self.awi.current_output_issues[UNIT_PRICE].max_value
            if self.is_consumer(partner_id)
            else self.awi.current_input_issues[UNIT_PRICE].min_value
        )

    def good_price(self, partner_id, today: bool):
        """A good price to use"""
        nmi = self.get_nmi(partner_id)
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        if self.is_supplier(partner_id):
            return self.buy_price(nmi.state.relative_time, mn, mx, today=today)
        return self.sell_price(
            self.get_nmi(partner_id).state.relative_time, mn, mx, today=today
        )

    def buy_price(self, t: float, mn: float, mx: float, today: bool) -> float:
        """Return a good price to buy at"""
        e = self.today_exp if today else self.future_exp
        return max(mn, min(mx, int(mn + (mx - mn) * (t**e) + 0.5)))

    def sell_price(self, t: float, mn: float, mx: float, today: bool) -> float:
        """Return a good price to sell at"""
        e = self.today_exp if today else self.future_exp
        if not today:
            mn = mn + self.fmin * (mx - mn)
        return max(mn, min(mx, int(0.5 + mx - (mx - mn) * (t**e))))

    def good2buy(self, p: float, t: float, mn, mx, today: bool):
        """Is p a good price to buy at?"""
        if not today:
            mx = mx - self.fmin * (mx - mn)
        return p - 0.0001 <= self.buy_price(t, mn, mx, today)

    def good2sell(self, p: float, t: float, mn, mx, today: bool):
        """Is p a good price to sell at?"""
        return p + 0.0001 >= self.sell_price(t, mn, mx, today)


# if __name__=="__main__":
#     from scml.std import *
#     from helpers.runner import run
#     from scml_agents import get_agents
#     from helpers.agents import *

#     # world = SCML2024StdWorld(
#     #     **SCML2024StdWorld.generate(
#     #         agent_types = [SupplyBasedAgent,AgentVSC2024,GreedySyncAgent,GreedyStdAgent,SyncRandomStdAgent,RandomStdAgent], n_steps=10
#     #     ),
#     #     construct_graphs=True,
#     # )
#     # print(world.agents)
#     # world.run()
#     # print(world.scores())
#     OneShot2023Agents = get_agents(version=2023,track="oneshot",finalists_only=True,as_class=True)
#     run([Prot3_22,AgentVSC2024,GreedyStdAgent,OneShot2023Agents[0],CautiousStdAgent],"std",n_steps=100,n_configs=5)
