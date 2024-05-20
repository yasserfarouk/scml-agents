from __future__ import annotations
from negmas.sao import SAONMI
import threading
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE

# required for typing
import random

# from scipy.stats import linregress
# required for development
from scml.std import StdAgent
from scml.scml2020.components import *

# required for typing
from negmas import Contract, SAOState
from negmas import ResponseType

__all__ = ["S5s"]


class Neg_list:
    #
    def __init__(self, neg_id):
        self.neg_id = neg_id
        self.q_dif = []  # 量の差分
        self.p_dif = []  # 価格の差分
        self.t_dif = []  # 納期の差分


class S5s(StdAgent):
    def init(self):
        super().init()
        # print(self.awi.level)
        # 最大日数分のリストを作成
        self.raw_inventory = [0 for _ in range(self.awi.n_steps)]  # 原料の在庫
        self.buy_cont = [0 for _ in range(self.awi.n_steps)]  # 購入数
        self.sell_cont = [0 for _ in range(self.awi.n_steps)]  # 販売数
        self.fac_lines_cap = [
            self.awi.n_lines for _ in range(self.awi.n_steps)
        ]  # 工場の生産能力
        self.loss_prd = []  # 不足量のリスト
        self.buy_p_sum = 0  # 購入価格の合計
        self.buy_q_num = 0  # 購入数
        self.sell_p_sum = 0  # 販売価格の合計
        self.sell_q_num = 0  # 販売数
        self.buy_day = [
            0 for _ in range(self.awi.n_steps)
        ]  # その日ごとの購入価格の合計
        self.buy_num = [0 for _ in range(self.awi.n_steps)]  # その日ごとの購入数
        self.sell_day = [
            0 for _ in range(self.awi.n_steps)
        ]  # その日ごとの販売価格の合計
        self.sell_num = [0 for _ in range(self.awi.n_steps)]  # その日ごとの販売数
        self.neg_place = [[] for _ in range(self.awi.n_steps)]  # 交渉相手のリスト
        self.ip = self.awi.catalog_prices[self.awi.my_input_product]
        self.op = self.awi.catalog_prices[self.awi.my_output_product]
        self.storage_num = 0
        self.ave_storage_cost = 0
        self.buy_num_day = 0  # 当日の購入数
        self.sell_num_day = 0  # 当日の販売数
        self.buy_cont_day = 0  # 当日の購入契約数
        self.sell_cont_day = 0  # 当日の販売契約数
        self.buy_p_day = 0  # その日の購入額
        self.sell_p_day = 0  # その日の販売額
        self.buy_ave_day = 0  # その日の購入平均価格
        self.sell_ave_day = 0  # その日の販売平均価格
        self.buy_ave_cont = 0  # その日の購入契約平均個数
        self.sell_ave_cont = 0  # その日の販売契約平均個数
        self.current_finances = self.awi.current_balance  # 現在の資金
        self.ip = self.awi.catalog_prices[self.awi.my_input_product]
        self.op = self.awi.catalog_prices[self.awi.my_output_product]
        self.used = threading.Lock()

    def before_step(self):
        self.offer_counter = 0
        self.inve_cost = self.awi.current_balance - self.current_finances
        # 在庫1つあたりのコストを計算
        if self.awi.current_inventory_input > 0:
            self.ave_storage_cost = (
                self.ave_storage_cost * self.storage_num + self.inve_cost
            ) / (self.storage_num + self.awi.current_inventory_input)
        self.storage_num += self.awi.current_inventory_input
        self.buy_num_day = 0  # 当日の購入数
        self.sell_num_day = 0  # 当日の販売数
        self.buy_cont_day = 0  # 当日の購入契約数
        self.sell_cont_day = 0  # 当日の販売契約数
        self.buy_p_day = 0  # その日の購入額
        self.sell_p_day = 0  # その日の販売額

        if self.awi.is_first_level:
            Q = self.awi.current_exogenous_input_quantity
            T = self.awi.current_step
            P = self.awi.current_exogenous_input_price
            self.buy_num[T] += Q
            self.buy_day[T] += P * Q
            self.buy_q_num += Q
            self.buy_p_sum += Q * P
            self.buy_cont_day += 1
            self.buy_num_day += Q
            self.buy_p_day += P * Q
            for t in range(T, self.awi.n_steps):
                if t != self.awi.n_steps - 1:
                    self.raw_inventory[t] = sum(self.buy_num[: t + 1]) - sum(
                        self.sell_num[: t + 1]
                    )
                else:
                    self.raw_inventory[t] = sum(self.buy_num) - sum(self.sell_num)

        elif self.awi.is_last_level:
            Q = self.awi.current_exogenous_output_quantity
            T = self.awi.current_step
            P = self.awi.current_exogenous_output_price
            self.sell_num[T] += Q
            self.sell_day[T] += P * Q
            self.sell_q_num += Q
            self.sell_p_sum += Q * P
            self.sell_cont_day += 1
            self.sell_num_day += Q
            self.sell_p_day += P * Q
            for t in range(T, self.awi.n_steps):
                if t != self.awi.n_steps - 1:
                    self.raw_inventory[t] = sum(self.buy_num[: t + 1]) - sum(
                        self.sell_num[: t + 1]
                    )
                else:
                    self.raw_inventory[t] = sum(self.buy_num) - sum(self.sell_num)

        return super().before_step()

    def step(self):
        # if self.awi.current_step == self.awi.n_steps - 1:
        #     # print(self.fac_lines_cap)
        #     print(self.awi.current_inventory_input)
        self.current_finances = self.awi.current_balance
        self._update_iop()

        self.buy_ave_day = 0  # その日の購入平均価格
        self.sell_ave_day = 0  # その日の販売平均価格
        self.buy_ave_cont = 0  # その日の購入契約平均個数
        self.sell_ave_cont = 0  # その日の販売契約平均個数
        if self.buy_num_day > 0:
            # 1日あたりの購入契約合意数
            # self.buy_day = self.buy_cont_day /self.buy_num_day
            self.buy_ave_day = self.buy_p_day / self.buy_num_day  # その日の購入平均価格
            self.buy_ave_cont = (
                self.buy_num_day / self.buy_cont_day
            )  # その日の購入契約平均個数
        else:
            self.buy_ave_day = 0
            self.buy_ave_cont = 0
        if self.sell_num_day > 0:
            # 1日あたりの販売契約合意数
            self.sell_ave_day = self.sell_p_day / self.sell_num_day
            self.sell_ave_cont = self.sell_num_day / self.sell_cont_day
        else:
            self.sell_ave_day = 0
            self.sell_ave_cont = 0
        self._update_iop()
        if self.raw_inventory[self.awi.current_step] < 0:
            self.buy_num[self.awi.current_step] += -self.raw_inventory[
                self.awi.current_step
            ]
            self.buy_day[self.awi.current_step] += (
                self.ip * -self.raw_inventory[self.awi.current_step]
            )
            self.buy_q_num += -self.raw_inventory[self.awi.current_step]
            self.buy_p_sum += self.ip * -self.raw_inventory[self.awi.current_step]
        for t in range(self.awi.current_step, self.awi.n_steps):
            if t != self.awi.n_steps - 1:
                self.raw_inventory[t] = sum(self.buy_num[: t + 1]) - sum(
                    self.sell_num[: t + 1]
                )
            else:
                self.raw_inventory[t] = sum(self.buy_num) - sum(self.sell_num)
        return super().step()

    def respond(self, negotiator_id: str, state: SAOState, source="") -> ResponseType:
        if state.current_offer is None:
            return ResponseType.REJECT_OFFER
        neg_info = None
        # if self.neg_place[self.awi.current_step] is None:
        #     neg_info = Neg_list(negotiator_id)
        #     self.neg_place[self.awi.current_step].append(neg_info)
        for place in self.neg_place[self.awi.current_step]:
            if place.neg_id == negotiator_id:
                neg_info = place
                break
        if neg_info is None:
            neg_info = Neg_list(negotiator_id)
            self.neg_place[self.awi.current_step].append(neg_info)
        neg_info.q_dif.append(state.current_offer[QUANTITY])
        neg_info.p_dif.append(state.current_offer[UNIT_PRICE])
        neg_info.t_dif.append(state.current_offer[TIME])

        Q = state.current_offer[QUANTITY]
        T = state.current_offer[TIME]
        P = state.current_offer[UNIT_PRICE]
        if T > self.awi.n_steps - 1:
            return ResponseType.REJECT_OFFER
        with self.used:
            if negotiator_id in self.awi.my_consumers:
                # print("完成品の販売")
                if self.fac_lines_cap[T] < Q:
                    return ResponseType.REJECT_OFFER
                if self.awi.current_step < self.awi.n_steps * 2 / 3:
                    if T == 0:
                        if self.raw_inventory[T] > Q:
                            if P > self.op:
                                return ResponseType.ACCEPT_OFFER
                        return ResponseType.REJECT_OFFER
                    if T < self.awi.current_step + self.awi.n_steps / 10:
                        if Q < self.raw_inventory[T]:
                            if P > self.op:
                                return ResponseType.ACCEPT_OFFER
                    elif T < self.awi.current_step + self.awi.n_steps / 5:
                        if Q < self.raw_inventory[T]:
                            if P > self.op:
                                return ResponseType.ACCEPT_OFFER
                        else:
                            if P > self.op * 1.2:
                                return ResponseType.ACCEPT_OFFER
                    else:
                        if P > self.op:
                            return ResponseType.ACCEPT_OFFER

                    return ResponseType.REJECT_OFFER
                else:
                    if self.raw_inventory[T] > Q:
                        return ResponseType.ACCEPT_OFFER
                return ResponseType.REJECT_OFFER

            else:
                # 材料購入
                if (
                    sum(self.buy_num) - sum(self.sell_num) + Q
                    > sum(self.fac_lines_cap[self.awi.current_step :]) * 0.6
                ):
                    return ResponseType.REJECT_OFFER

                for t in range(
                    T, min(int(T + self.awi.n_steps / 10), self.awi.n_steps)
                ):
                    if self.raw_inventory[t] < 0:
                        if self.raw_inventory[t] + Q < self.awi.n_lines / 2:
                            if P < self.ip:
                                return ResponseType.ACCEPT_OFFER
                        else:
                            if P < self.ip * 0.9:
                                return ResponseType.ACCEPT_OFFER
                # 全日程を通した，生産ライン数の60％を超える購入は拒否
                # if sum(self.buy_num[self.awi.current_step:]) + Q  > self.awi.n_lines * (self.awi.n_steps - self.awi.current_step) * 0.4:
                if sum(self.buy_num) - sum(self.sell_num) + Q > self.awi.n_lines:
                    # print("too much")
                    return ResponseType.REJECT_OFFER
                if T < self.awi.n_steps * 2 / 3:
                    max_q = max(
                        self.raw_inventory[T : int(self.awi.n_steps * 2 / 3) + 1]
                    )
                    if max_q + Q < self.awi.n_lines:
                        if P < self.ip:
                            return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER
                return ResponseType.REJECT_OFFER

    def propose(self, negotiator_id: str, state):
        with self.used:
            neg_info = None
            for place in self.neg_place[self.awi.current_step]:
                if place.neg_id == negotiator_id:
                    neg_info = place
                    break
            if neg_info is None:
                neg_info = Neg_list(negotiator_id)
                self.neg_place[self.awi.current_step].append(neg_info)
            if negotiator_id in self.awi.my_suppliers:
                # 材料購入戦略
                need_num = 0
                if self.awi.current_step < self.awi.n_steps * 2 / 3:
                    if sum(self.buy_num) - sum(self.sell_num) > 2 * self.awi.n_lines:
                        return None
                else:
                    return None
                if state.step > 10:
                    inde = 0
                    for t_offer in neg_info.t_dif:
                        for time in range(
                            t_offer,
                            min(int(t_offer + self.awi.n_steps / 10), self.awi.n_steps),
                        ):
                            if self.raw_inventory[time] < 0:
                                need_num = neg_info.q_dif[inde]
                                first_loss = time
                                break
                        if need_num > 0:
                            break
                        inde += 1
                # current_step + (n_steps/10)以内で不足があれば率先して提案
                if (
                    self.awi.current_step + self.awi.n_steps / 10
                    > self.awi.n_steps * 0.8
                ):
                    return None
                min_ip = min(
                    self.ip * 0.8,
                    self.awi.trading_prices[self.awi.my_input_product] * 0.9,
                )
                if need_num == 0:
                    first_loss = 0
                    for t in range(
                        self.awi.current_step,
                        int(self.awi.current_step + self.awi.n_steps / 10),
                    ):
                        if self.raw_inventory[t] < 0:
                            if first_loss == 0:
                                first_loss = t
                            if need_num < -self.raw_inventory[t]:
                                need_num = -self.raw_inventory[t]
                    if need_num > 0:
                        need_num = min(need_num, self.awi.n_lines / 3)
                    else:
                        need_num = self.awi.n_lines / 3
                    if first_loss == 0:
                        first_loss = random.randint(
                            self.awi.current_step,
                            int(self.awi.current_step + self.awi.n_steps / 10),
                        )

                        need_num = min(need_num, self.awi.n_lines / 3)
                min_ip = min(
                    self.ip * 0.8,
                    self.awi.trading_prices[self.awi.my_input_product] * 0.9,
                )
                p = int(state.step / 20 * (self.ip - min_ip) + min_ip)
                offer = self.check_proposal([need_num, p, first_loss], negotiator_id)
                return tuple(offer)
            else:
                # 製品販売戦略
                if self.awi.current_step < self.awi.n_steps * 2 / 3:
                    emp_lines = [0 for _ in range(self.awi.n_steps)]
                    for time in range(
                        self.awi.current_step,
                        int(self.awi.current_step + self.awi.n_steps / 5),
                    ):
                        if self.raw_inventory[time] == 0:
                            emp_lines[time] = 0
                        else:
                            emp_lines[time] = min(
                                self.fac_lines_cap[time], self.raw_inventory[time]
                            )
                    emp_lines_index = sorted(
                        range(len(emp_lines)), key=lambda k: emp_lines[k], reverse=True
                    )
                    time = emp_lines_index[state.step // 3]
                    q = min(emp_lines[time], self.awi.n_lines / 3)
                    # q = emp_lines[time]
                    if q == 0:
                        return None
                    p = int(
                        -state.step
                        / 20
                        * (
                            (
                                (
                                    self.awi.catalog_prices[self.awi.my_output_product]
                                    + self.awi.trading_prices[
                                        self.awi.my_output_product
                                    ]
                                )
                                / 2
                            )
                            - self.op
                        )
                        + (
                            (
                                self.awi.catalog_prices[self.awi.my_output_product]
                                + self.awi.trading_prices[self.awi.my_output_product]
                            )
                            / 2
                        )
                    )
                    offer = self.check_proposal([q, p, time], negotiator_id)
                else:
                    q = 0
                    time = 0
                    for t in range(self.awi.current_step, self.awi.n_steps):
                        if self.raw_inventory[t] > 0:
                            time = t
                            q = min(self.fac_lines_cap[t], self.raw_inventory[t])
                            break
                    q = min(q, self.awi.n_lines / 3)
                    p = self.awi.current_output_issues[1].values[0]
                    offer = self.check_proposal([q, p, time], negotiator_id)
                return tuple(offer)

    def check_proposal(self, offer, nego_id):
        if nego_id in self.awi.my_suppliers:
            current_issue = self.awi.current_input_issues
            if offer[0] < current_issue[0].values[0]:
                offer[0] = current_issue[0].values[0]
            elif offer[0] > current_issue[0].values[1]:
                offer[0] = current_issue[0].values[1]

            if offer[1] < current_issue[1].values[0]:
                offer[1] = current_issue[1].values[0]
            elif offer[1] > current_issue[1].values[1]:
                offer[1] = current_issue[1].values[1]

            if offer[2] < current_issue[2].values[0]:
                offer[2] = current_issue[2].values[0]
            elif offer[2] > current_issue[2].values[1]:
                offer[2] = current_issue[2].values[1]
        else:
            current_issue = self.awi.current_output_issues
            if offer[0] < current_issue[0].values[0]:
                offer[0] = current_issue[0].values[0]
            elif offer[0] > current_issue[0].values[1]:
                offer[0] = current_issue[0].values[1]

            if offer[1] < current_issue[1].values[0]:
                offer[1] = current_issue[1].values[0]
            elif offer[1] > current_issue[1].values[1]:
                offer[1] = current_issue[1].values[1]

            if offer[2] < current_issue[2].values[0]:
                offer[2] = current_issue[2].values[0]
            elif offer[2] > current_issue[2].values[1]:
                offer[2] = current_issue[2].values[1]
        return offer

    def on_negotiation_success(self, contract: Contract, mechanism: SAONMI) -> None:
        # 交渉が成功したときに呼び出される
        # 合意に至った契約に基づいて、自身の内部状態を更新する
        # 購入契約の場合は，入荷日の原料の在庫を更新する
        with self.used:
            quantity = contract.agreement["quantity"]
            price = contract.agreement["unit_price"]
            time = contract.agreement["time"]

            if contract.annotation["buyer"] == self.id:
                self.buy_num[time] += quantity
                self.buy_day[time] += price * quantity
                self.buy_p_sum += price * quantity
                self.buy_q_num += quantity
                self.buy_cont_day += 1
                self.buy_num_day += quantity
                self.buy_p_day += price * quantity

            # 販売契約の場合は，出荷日の製品の在庫を更新する
            else:
                self.fac_lines_cap[time] -= quantity
                self.sell_num[time] += quantity
                self.sell_day[time] += price * quantity
                self.sell_p_sum += price * quantity
                self.sell_q_num += quantity
                self.sell_cont_day += 1
                self.sell_num_day += quantity
                self.sell_p_day += price * quantity
            for t in range(time, self.awi.n_steps):
                if t != self.awi.n_steps - 1:
                    self.raw_inventory[t] = sum(self.buy_num[: t + 1]) - sum(
                        self.sell_num[: t + 1]
                    )
                else:
                    self.raw_inventory[t] = sum(self.buy_num) - sum(self.sell_num)
        return super().on_negotiation_success(contract, mechanism)

    def _update_iop(self):
        if self.buy_cont_day > 0:
            if self.buy_ave_day / self.ip > 0.9:
                self.ip *= 1.05
            else:
                self.ip *= 0.95
        else:
            self.ip *= 1.1

        if self.sell_cont_day > 0:
            if self.sell_ave_day / self.op > 1.1:
                self.op *= 1.1
            else:
                self.op *= 0.9
        else:
            self.op *= 0.9
