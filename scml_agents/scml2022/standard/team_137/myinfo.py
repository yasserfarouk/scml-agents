from multiprocessing import parent_process
from re import A
from typing import List
from negmas import Breach, Contract, Issue
import copy


class myinfo:
    def __init__(self, parent):
        self.parent = parent

        n_steps = self.parent.awi.n_steps
        n_lines = self.parent.awi.n_lines

        # price
        self.min_margine = 5
        self.max_margine = 10
        self.p1: tuple = None  # 0 min 1 typ 2 max
        self.p2: tuple = None  # 0 min 1 typ 2 max

        # quantity and time

        self.use_line: dict[int, int] = [0 for _ in range(n_steps)]
        self.secure_inventoryB: dict[int, int] = [
            0 for _ in range(n_steps + 1)
        ]  # 最終日の次の日用に＋１

        # 在庫管理パラメタ
        self.TIMEMARGINE = 0  # TODO

        if self.parent.awi.is_last_level == True:
            self.last_trade_day = n_steps - 1
            self.invB_checkday = n_steps  # 売りが先に決まっているのでできる時にはいつでも買ってよい。

            self.invB_q_ulimit = 0
            self.invB_q_blimit = -30  # TODO

            self.n = n_steps  # 取引可能な日数の範囲(全て)

        elif self.parent.awi.is_middle_level == True:
            self.last_trade_day = (
                n_steps
                - 1
                - (
                    (self.parent.awi.n_products - self.parent.awi.my_input_product)
                    + self.TIMEMARGINE
                )
            )
            self.last_trade_day = max(self.last_trade_day, 0)

            self.invB_checkday = self.last_trade_day - 10  # TODO
            self.invB_checkday = max(self.invB_checkday, 0)

            self.invB_q_ulimit = 50
            self.invB_q_blimit = 5  # TODO

            self.n = 40  # 取引可能な日数の範囲

        else:  # is_first_level==True
            self.last_trade_day = (
                n_steps
                - 1
                - (
                    (self.parent.awi.n_products - self.parent.awi.my_input_product)
                    + self.TIMEMARGINE
                )
            )
            self.last_trade_day = max(self.last_trade_day, 0)

            self.invB_checkday = self.last_trade_day - 10  # TODO
            self.invB_checkday = max(self.invB_checkday, 0)

            self.invB_q_ulimit = 50
            self.invB_q_blimit = 5  # TODO

            self.n = n_steps  # 取引可能な日数の範囲(倒産しないので全て)

        self.first_day = -1
        self.last_day = -1
        # need
        self.input_needs: dict[int, int] = [0 for _ in range(n_steps)]
        self.output_needs: dict[int, int] = [0 for _ in range(n_steps)]  # マイナスにもなります。
        self.max_input = -1

        # controll
        self.negotiator_laststep = 2

        # 破産管理用
        self.offer_memoryA = list()
        self.offer_memoryB = list()
        self.offer_memoryC = list()

    def set_need(self):
        n_steps = self.parent.awi.n_steps

        current_step = self.parent.awi.current_step
        n_lines = self.parent.awi.n_lines

        self.first_day = current_step

        self.last_day = min(current_step + self.n, n_steps)

        # INPUT
        self.input_needs: dict[int, int] = [0 for _ in range(n_steps)]
        for i in range(self.first_day, min(self.last_day, self.invB_checkday)):
            self.input_needs[i] = n_lines - self.use_line[i]
        self.max_input = self.invB_q_ulimit - self.secure_inventoryB[self.invB_checkday]

        # OUTPUT
        self.output_needs: dict[int, int] = [0 for _ in range(n_steps)]

        for i in reversed(range(self.first_day, self.last_day)):
            if i >= self.invB_checkday:
                if i == self.last_day - 1:
                    self.output_needs[i] = self.secure_inventoryB[i]
                else:
                    self.output_needs[i] = min(
                        self.output_needs[i + 1], self.secure_inventoryB[i]
                    )
            else:
                if i == self.last_day - 1:
                    self.output_needs[i] = (
                        self.secure_inventoryB[i] - self.invB_q_blimit
                    )
                else:
                    self.output_needs[i] = min(
                        self.output_needs[i + 1],
                        self.secure_inventoryB[i] - self.invB_q_blimit,
                    )
        # print(self.secure_inventoryB)

    def set_contractA(self, agreement):
        n_steps = self.parent.awi.n_steps
        n_lines = self.parent.awi.n_lines
        t = agreement["time"]
        q = agreement["quantity"]
        p = agreement["unit_price"]
        self.offer_memoryA.append(agreement)
        q_t = q
        d = t
        while d < n_steps:
            if n_lines - self.use_line[d] > 0:
                m = min(n_lines - self.use_line[d], q_t)
                q_t -= m
                self.use_line[d] += m
                for t_d in range(d + 1, n_steps + 1):
                    self.secure_inventoryB[t_d] += m
            if q_t == 0:
                break
            d += 1
        if q_t > 0:
            # print("error:myinfo/toolargecontract(A)")
            pass

    def set_contractB(self, agreement):
        n_steps = self.parent.awi.n_steps
        self.offer_memoryB.append(agreement)
        t = agreement["time"]
        q = agreement["quantity"]
        p = agreement["unit_price"]
        for d in range(t, n_steps + 1):
            self.secure_inventoryB[d] -= q

    def set_contractC(self, agreement):  # spot
        n_steps = self.parent.awi.n_steps
        n_lines = self.parent.awi.n_lines
        t = agreement["time"]
        q = agreement["quantity"]
        p = agreement["unit_price"]
        self.offer_memoryC.append(agreement)

        for t_d in range(t, n_steps + 1):
            self.secure_inventoryB[t_d] += q

    def set_bankrapt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ):

        if len(contracts) == 0:
            return

        n_steps = self.parent.awi.n_steps

        temp_offer_memoryA = copy.deepcopy(self.offer_memoryA)
        self.offer_memoryA.clear()
        temp_offer_memoryB = copy.deepcopy(self.offer_memoryB)
        self.offer_memoryB.clear()
        temp_offer_memoryC = copy.deepcopy(self.offer_memoryC)
        self.offer_memoryC.clear()

        # 一旦クリアして再生する
        self.use_line: dict[int, int] = [0 for _ in range(n_steps)]
        self.secure_inventoryB: dict[int, int] = [0 for _ in range(n_steps + 1)]

        for contract, new_quantity in zip(contracts, quantities):
            c_q = contract.agreement["quantity"]
            c_t = contract.agreement["time"]
            c_p = contract.agreement["unit_price"]

            if contract.annotation["seller"] == self.parent.id:
                for i in range(len(temp_offer_memoryB)):
                    if (
                        temp_offer_memoryB[i]["quantity"] == c_q
                        and temp_offer_memoryB[i]["time"] == c_t
                        and temp_offer_memoryB[i]["unit_price"] == c_p
                    ):
                        temp_offer_memoryB[i]["quantity"] = new_quantity
                        break
            else:
                for i in range(len(temp_offer_memoryA)):
                    if (
                        temp_offer_memoryA[i]["quantity"] == c_q
                        and temp_offer_memoryA[i]["time"] == c_t
                        and temp_offer_memoryA[i]["unit_price"] == c_p
                    ):
                        temp_offer_memoryA[i]["quantity"] = new_quantity
                        break

        for agreement in temp_offer_memoryA:
            self.set_contractA(agreement)
        for agreement in temp_offer_memoryB:
            self.set_contractB(agreement)
        for agreement in temp_offer_memoryC:  # spot
            self.set_contractC(agreement)
