from __future__ import annotations

import random

# required for typing
# required for typing
from negmas import Contract, ResponseType, SAOResponse

# required for development
from scml.oneshot import *
from scml.std import *

__all__ = ["SKRI"]


def distribute(
    q: int,
    n: int,
    *,
    mx: int | None = None,
    equal=True,
    concentrated=False,
    allow_zero=False,
    concentrated_idx: list[int] | None = None,
) -> list[int]:
    from collections import Counter

    from numpy.random import choice

    q, n = int(q), int(n)

    if mx is not None and q > mx * n:
        q = mx * n

    # normalize concentrated_idx default to avoid mutable default
    if concentrated_idx is None:
        concentrated_idx = []

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n

    # concentrated 配分ロジックは q>=n を前提とするため、ここで扱う
    if concentrated:
        assert mx is not None
        # 各binに最低1個ずつ配分（q>=n を想定）
        lst = [1] * n
        remaining = max(0, q - n)

        # 残りを1個ずつ順番に配分
        idx = 0
        while remaining > 0:
            if lst[idx] < mx:
                lst[idx] += 1
                remaining -= 1
            idx = (idx + 1) % n

        # concentrated_idxの順序でlstの値を割り当て
        result = [0] * n
        for i, target_idx in enumerate(concentrated_idx):
            if i < len(lst) and 0 <= target_idx < n:
                result[target_idx] = lst[i]

        # 未割当は残りから順に割り当てる
        assigned = sum(1 for v in result if v > 0)
        if assigned < n:
            # 残りのインデックス
            remaining_indices = [i for i, v in enumerate(result) if v == 0]
            # lst の残り値を順に割り当て
            j = 0
            for val in lst[assigned:]:
                if j >= len(remaining_indices):
                    break
                result[remaining_indices[j]] = val
                j += 1

        return result

    if allow_zero:
        per = 0
    else:
        per = (q // n) if equal else 1
    q -= per * n
    # 明示的に replace=True を指定して安全にサンプリング
    r = Counter(choice(n, q, replace=True))
    return [r.get(_, 0) + per for _ in range(n)]


def powerset(iterable):
    """冪集合"""
    from itertools import chain, combinations

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class SKRI(OneShotSyncAgent):
    def __init__(
        self,
        *args,
        equal: bool = True,
        over_buying: float = 0.2,
        quantity_price_balance: float = 0.85,
        buyer_QP_score_threshold: float = 0.6,  # 買い手の意思決定閾値
        seller_QP_score_threshold: float = 0.6,  # 売り手の意思決定閾値
        concession_alpha: float = 2.0,
        **kwargs,
    ):
        self.equal_distribution = equal
        self.over_buying = over_buying
        self.quantity_price_balance = quantity_price_balance
        self.buyer_QP_score_threshold = buyer_QP_score_threshold
        self.seller_QP_score_threshold = seller_QP_score_threshold
        # 譲歩関数の形状パラメータ（f(t)=t**alpha）
        self.concession_alpha = concession_alpha
        # self.check_needs = 0
        super().__init__(*args, **kwargs)

    def init(self):
        self.total_agreed_quantity = {
            k: 0
            for k in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
        }
        self.is_seller = self.awi.my_suppliers == ["SELLER"]
        # opponent modelling: 観測されたオファー数と合意数
        self.offer_seen_count = {k: 0 for k in self.total_agreed_quantity.keys()}
        self.offer_accept_count = {k: 0 for k in self.total_agreed_quantity.keys()}
        # 各相手に対して送信したカウンター回数
        self.counter_offer_sent = {k: 0 for k in self.total_agreed_quantity.keys()}
        return super().init()

    def _counter_price_for(self, partner: str, is_seller: bool, issues) -> int:
        """Compute counter price for a partner: start from most-favorable price and
        move by 0.3 * (pmax-pmin) per counter already sent (cumulative).
        """
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        base = float(pmax) if is_seller else float(pmin)
        rng = float(pmax - pmin)
        # next counter index (1-based)
        next_cnt = self.counter_offer_sent.get(partner, 0) + 1
        delta = 0.3 * rng * next_cnt
        sign = -1.0 if is_seller else 1.0
        price_f = base + sign * delta
        price = int(round(price_f))
        price = max(int(pmin), min(int(pmax), price))
        # record that we've sent a counter to this partner
        self.counter_offer_sent[partner] = next_cnt
        return price

    def distribute_needs(
        self,
        t: float,
        mx: int | None = None,
        equal: bool | None = None,
        allow_zero: bool | None = None,
        concentrated: bool = False,
        concentrated_ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Distributes my needs equally over all my partners"""

        if equal is None:
            equal = True
        if allow_zero is None:
            allow_zero = self.awi.allow_zero_quantity
        if concentrated_ids is None:
            concentrated_ids = []

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # find suppliers and consumers still negotiating with me
            partners, n_partners = [], 0
            concentrated_idx = []
            for p in all_partners:
                if p not in self.negotiators.keys():
                    continue
                partners.append(p)
                if p in concentrated_ids:
                    concentrated_idx.append(n_partners)
                n_partners += 1

            # if I need nothing, end all negotiations
            if needs <= 0:
                dist.update(dict(zip(partners, [0] * n_partners)))
                continue

            # 買い手の場合のみ必要数量より多めに買う
            is_buyer = all_partners == self.awi.my_suppliers
            adjusted_needs = int(needs * (1 + self.over_buying) if is_buyer else needs)
            # distribute my needs over my (remaining) partners.
            dist.update(
                dict(
                    zip(
                        partners,
                        distribute(
                            adjusted_needs,
                            n_partners,
                            mx=mx,
                            equal=equal,
                            concentrated=concentrated,
                            allow_zero=allow_zero,
                            concentrated_idx=concentrated_idx,
                        ),
                    )
                )
            )
        return dist

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        partner_id = [p for p in contract.partners if p != self.id][0]
        self.total_agreed_quantity[partner_id] += contract.agreement["quantity"]
        # 合意が成立した相手のカウントを増やす
        if partner_id not in self.offer_accept_count:
            self.offer_accept_count[partner_id] = 0
        self.offer_accept_count[partner_id] += 1

    def first_proposals(self):
        # distribute my needs equally over my partners (with best price for me).
        s, p = self._step_and_price(best_price=True)
        my_negotiators = [
            p
            for p in (
                self.awi.my_consumers
                if self.awi.my_suppliers == ["SELLER"]
                else self.awi.my_suppliers
            )
            if p in self.negotiators.keys()
        ]
        if (
            self.awi.current_step > max(50, self.awi.n_steps * 0.5)
            and self.awi.level == 0
            and len(my_negotiators) > 0
        ):
            # 合意確率の高い相手を優先
            concentrated_ids = sorted(
                my_negotiators,
                key=lambda x: (
                    (self.offer_accept_count.get(x, 0) + 1)
                    / (self.offer_seen_count.get(x, 0) + 2),
                    self.total_agreed_quantity.get(x, 0),
                ),
                reverse=True,
            )
            distribution = self.distribute_needs(
                t=0,
                mx=3,
                equal=True,
                allow_zero=False,
                concentrated=True,
                concentrated_ids=concentrated_ids,
            )
        elif (
            self.awi.current_step > max(100, self.awi.n_steps * 0.75)
            and self.awi.level == 1
            and len(my_negotiators) > 0
        ):
            concentrated_ids = sorted(
                my_negotiators,
                key=lambda x: (
                    (self.offer_accept_count.get(x, 0) + 1)
                    / (self.offer_seen_count.get(x, 0) + 2),
                        self.total_agreed_quantity.get(x, 0),
                ),
                reverse=True,
            )
            distribution = self.distribute_needs(
                t=0,
                mx=3,
                equal=True,
                allow_zero=False,
                concentrated=True,
                concentrated_ids=concentrated_ids,
            )
        else:
            distribution = self.distribute_needs(
                t=0, mx=3, allow_zero=False, equal=True
            )
        d = {
            k: (q, s, p) if q > 0 or self.awi.allow_zero_quantity else None
            for k, q in distribution.items()
        }
        return d

    def counter_all(self, offers, states):
        response = dict()
        # my_initial を後で送信（現ステップで提案を出した相手を除外して先手を取る）
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
        # 観測されたオファーをカウント（opponent modelling）
        for p in offers.keys():
            self.offer_seen_count[p] = self.offer_seen_count.get(p, 0) + 1
        # 先手オファー: まず全相手に自分に最も有利な価格でオファーを送る
        my_initial = self.first_proposals()
        if my_initial:
            for k, v in my_initial.items():
                if v is None:
                    continue
                response[k] = SAOResponse(ResponseType.REJECT_OFFER, v)
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
            price = issues[UNIT_PRICE].rand()
            # find active partners in some random order
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)
            is_selling = all_partners == self.awi.my_consumers

            # 買い手判定と調整済みニーズを早期に定義
            is_buyer = all_partners == self.awi.my_suppliers
            adjusted_needs = int(needs * (1 + self.over_buying) if is_buyer else needs)

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER, (0, self.awi.current_step, price)
                )
            )

            # 交渉相手の提案の組み合わせを評価
            # 相手数が小さい場合は全探索、それ以外は0/1ナップサック近似で高速化
            if len(partners) <= 10:
                plist = list(powerset(partners))[::-1]
            else:
                # ナップサック近似：各提案を価値(＝数量×単位価値)と重み(数量)にして選択
                p_list = list(partners)
                # 価格正規化用の範囲
                pmin = issues[UNIT_PRICE].min_value
                pmax = issues[UNIT_PRICE].max_value
                price_range = pmax - pmin if pmax != pmin else 1

                # 各提案の重みと価値を計算
                items = []  # (idx, weight, value)
                for i, pid in enumerate(p_list):
                    q = offers[pid][QUANTITY]
                    up = offers[pid][UNIT_PRICE]
                    # 単位価格の好みスコア（買い手は低い方が良い、売り手は高い方が良い）
                    if is_selling:
                        unit_price_score = (up - pmin) / price_range
                    else:
                        unit_price_score = (pmax - up) / price_range
                    unit_price_score = max(0.0, min(1.0, unit_price_score))

                    # 数量の好み（needs に近い数量を高く評価）
                    avg_target = needs / max(1, len(p_list))
                    quantity_pref = 1.0 - abs(q - avg_target) / (needs if needs != 0 else 1)
                    quantity_pref = max(0.0, min(1.0, quantity_pref))

                    per_unit_value = (
                        self.quantity_price_balance * quantity_pref
                        + (1 - self.quantity_price_balance) * unit_price_score
                    )
                    # 合意確率を乗じて期待効用とする
                    accept_rate = (
                        (self.offer_accept_count.get(pid, 0) + 1)
                        / (self.offer_seen_count.get(pid, 0) + 2)
                    )
                    value = per_unit_value * q * accept_rate
                    weight = int(max(1, q))
                    items.append((i, weight, value))

                # 容量は調整済みニーズ（買い手は adjusted_needs を使う）
                cap = adjusted_needs if adjusted_needs > 0 else sum(w for _, w, _ in items)
                cap = int(max(1, cap))

                # DP テーブル（小さい cap を想定）
                dp_val = [0.0] * (cap + 1)
                dp_choice = [set() for _ in range(cap + 1)]
                for idx, w, v in items:
                    if w > cap:
                        # 重すぎるアイテムはスキップまたは単独候補として後で考慮
                        continue
                    for c in range(cap, w - 1, -1):
                        if dp_val[c - w] + v > dp_val[c]:
                            dp_val[c] = dp_val[c - w] + v
                            dp_choice[c] = dp_choice[c - w].copy()
                            dp_choice[c].add(idx)

                # best selection
                best_c = max(range(cap + 1), key=lambda x: dp_val[x])
                chosen_idx_set = dp_choice[best_c]
                chosen = tuple(p_list[i] for i in chosen_idx_set)
                # 最低でも最大価値単体は候補に入れる（DPで除外された大きいアイテム対策）
                if not chosen:
                    # 単体で最も価値の高いアイテムを選ぶ
                    if items:
                        best_item = max(items, key=lambda it: it[2])
                        chosen = (p_list[best_item[0]],)

                plist = [chosen]

            # 最適な組み合わせを選択するためのカウンター
            best_QP_score, best_quantity, best_indx = float("-inf"), 0, -1

            # 買い手の場合のみ必要数量を調整
            is_buyer = all_partners == self.awi.my_suppliers
            adjusted_needs = int(needs * (1 + self.over_buying) if is_buyer else needs)

            # 各組み合わせを評価する際に実際の必要数を使用
            for i, partner_ids in enumerate(plist):
                offered_quantity = sum(offers[p][QUANTITY] for p in partner_ids)
                quantity_diff = offered_quantity - needs

                # コスト計算
                total_price = sum(
                    offers[p][UNIT_PRICE] * offers[p][QUANTITY] for p in partner_ids
                )
                # ペナルティ
                penalty = 0
                if is_selling:  # 売り手
                    if offered_quantity > needs:  # 過剰販売
                        penalty += (
                            offered_quantity - needs
                        ) * self.awi.current_shortfall_penalty
                    if offered_quantity < needs:  # 不足
                        penalty += (
                            needs - offered_quantity
                        ) * self.awi.current_disposal_cost
                else:  # 買い手
                    if offered_quantity < needs:  # 不足
                        penalty += (
                            needs - offered_quantity
                        ) * self.awi.current_shortfall_penalty
                    if offered_quantity > needs:  # 過剰購入
                        penalty += (
                            offered_quantity - needs
                        ) * self.awi.current_disposal_cost

                # Pスコアを計算
                P_score = (
                    (total_price - penalty) if is_selling else -(total_price - penalty)
                )
                # 正規化
                normalized_P_score = (
                    0.5 + (P_score / 1000)
                    if P_score > 0
                    else 0.5 - (abs(P_score) / 1000)
                )
                normalized_P_score = max(0, min(1, normalized_P_score))

                normalized_quantity = (
                    1.0 - abs(quantity_diff) / needs if needs != 0 else 0
                )

                # QPスコアを計算
                QP_score = (
                    self.quantity_price_balance * normalized_quantity
                    + (1 - self.quantity_price_balance) * normalized_P_score
                )

                # 最良の組み合わせを更新
                if QP_score > best_QP_score:
                    best_QP_score = QP_score
                    best_quantity = quantity_diff
                    best_indx = i
                elif QP_score == best_QP_score:
                    # 同じスコアの場合、売り手は過剰を、買い手は不足を避ける
                    if (is_selling and quantity_diff < best_quantity) or (
                        not is_selling
                        and quantity_diff > best_quantity
                        and quantity_diff <= 0
                    ):
                        best_quantity = quantity_diff
                        best_indx = i

            # 選択のための閾値の決定（売り手と買い手で異なる）
            QP_score_threshold = (
                self.seller_QP_score_threshold
                if is_selling
                else self.buyer_QP_score_threshold
            )

            # 交渉時間の経過に応じて閾値を調整（時間経過で閾値を下げる）
            relative_time = min(state.relative_time for state in states.values())
            adjusted_threshold = QP_score_threshold * (1 - 0.5 * relative_time)

            # 意思決定: スコアが閾値を超えた場合、提案を受け入れる
            if best_indx >= 0 and best_QP_score >= adjusted_threshold:
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids).union(future_partners))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: unneeded_response for k in others}

                # 不足数量がある場合、まず受信済みの有利なオファーを受け入れ、残りをカウンターで埋める
                if best_quantity < 0 and len(others) > 0:
                    s, price = self._step_and_price(best_price=False)

                    # 不足数量を計算
                    shortage = -best_quantity

                    # 受信済みオファー（othersのうち現ステップのoffersにあるもの）について
                    # 小規模なら全探索で最良組合せを選ぶ
                    candidates = [p for p in others if p in offers]
                    accepted = []
                    if candidates:
                        best_subset = None
                        best_score = float("-inf")
                        # 全探索の閾値（過度な組合せ爆発を避ける）
                        if len(candidates) <= 12:
                            for subset in powerset(candidates):
                                subset = list(subset)
                                if not subset:
                                    continue
                                offered_q = sum(offers[p][QUANTITY] for p in subset)
                                # 評価指標は QP_score を用いる（needs に対する一致と価格）
                                total_price = sum(offers[p][UNIT_PRICE] * offers[p][QUANTITY] for p in subset)
                                penalty = 0
                                if is_selling:
                                    if offered_q > needs:
                                        penalty += (offered_q - needs) * self.awi.current_shortfall_penalty
                                    if offered_q < needs:
                                        penalty += (needs - offered_q) * self.awi.current_disposal_cost
                                else:
                                    if offered_q < needs:
                                        penalty += (needs - offered_q) * self.awi.current_shortfall_penalty
                                    if offered_q > needs:
                                        penalty += (offered_q - needs) * self.awi.current_disposal_cost

                                P_score = (total_price - penalty) if is_selling else -(total_price - penalty)
                                normalized_P_score = 0.5 + (P_score / 1000) if P_score > 0 else 0.5 - (abs(P_score) / 1000)
                                normalized_P_score = max(0, min(1, normalized_P_score))
                                normalized_quantity = 1.0 - abs(offered_q - needs) / needs if needs != 0 else 0
                                QP_score = self.quantity_price_balance * normalized_quantity + (1 - self.quantity_price_balance) * normalized_P_score

                                # 優先条件：不足を満たす組み合わせを高評価
                                meets_shortage = offered_q >= shortage
                                score = QP_score + (0.1 if meets_shortage else 0.0)
                                if score > best_score:
                                    best_score = score
                                    best_subset = subset
                        else:
                            # 候補が多い場合は近似的にソートして上位から選ぶ
                            scored = []
                            pmin = issues[UNIT_PRICE].min_value
                            pmax = issues[UNIT_PRICE].max_value
                            price_range = pmax - pmin if pmax != pmin else 1
                            for pid in candidates:
                                q = offers[pid][QUANTITY]
                                up = offers[pid][UNIT_PRICE]
                                if is_selling:
                                    unit_price_score = (up - pmin) / price_range
                                else:
                                    unit_price_score = (pmax - up) / price_range
                                unit_price_score = max(0.0, min(1.0, unit_price_score))
                                avg_target = needs / max(1, len(candidates))
                                quantity_pref = 1.0 - abs(q - avg_target) / (needs if needs != 0 else 1)
                                quantity_pref = max(0.0, min(1.0, quantity_pref))
                                per_unit_value = (self.quantity_price_balance * quantity_pref + (1 - self.quantity_price_balance) * unit_price_score)
                                accept_rate = ((self.offer_accept_count.get(pid, 0) + 1) / (self.offer_seen_count.get(pid, 0) + 2))
                                value = per_unit_value * q * accept_rate
                                scored.append((value, pid))
                            scored.sort(reverse=True)
                            best_subset = []
                            for _, pid in scored:
                                if shortage <= 0:
                                    break
                                best_subset.append(pid)

                        if best_subset:
                            for p in best_subset:
                                if p in offers:
                                    response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])
                                    shortage -= offers[p][QUANTITY]

                    # 残りの不足分はカウンターを提示
                    if shortage > 0:
                        remaining = [o for o in others if o not in response]
                        if remaining:
                            # 集中配分が有効な場合はそれを使う
                            if (
                                self.awi.current_step > max(50, self.awi.n_steps * 0.5)
                                and self.awi.level == 0
                            ) or (
                                self.awi.current_step > max(100, self.awi.n_steps * 0.75)
                                and self.awi.level == 1
                            ):
                                concentrated_ids = sorted(
                                    remaining,
                                    key=lambda x: (
                                        (self.offer_accept_count.get(x, 0) + 1)
                                        / (self.offer_seen_count.get(x, 0) + 2),
                                        self.total_agreed_quantity.get(x, 0),
                                    ),
                                    reverse=True,
                                )
                                concentrated_idx = []
                                for p in concentrated_ids:
                                    if p in remaining:
                                        concentrated_idx.append(remaining.index(p))
                                distribution = dict(
                                    zip(
                                        remaining,
                                        distribute(
                                            shortage,
                                            len(remaining),
                                            mx=3,
                                            equal=True,
                                            concentrated=True,
                                            concentrated_idx=concentrated_idx,
                                            allow_zero=False,
                                        ),
                                    )
                                )
                            else:
                                distribution = dict(
                                    zip(
                                        remaining,
                                        distribute(
                                            shortage,
                                            len(remaining),
                                            mx=3,
                                            equal=True,
                                            allow_zero=False,
                                        ),
                                    )
                                )
                            # 最後に残った不足分は自分にとって最も不利な価格でオファーを出す
                            worst_price = (
                                issues[UNIT_PRICE].min_value if is_selling else issues[UNIT_PRICE].max_value
                            )
                            response.update(
                                {
                                    k: (
                                        unneeded_response
                                        if q == 0
                                        else SAOResponse(
                                            ResponseType.REJECT_OFFER,
                                            (q, self.awi.current_step, worst_price),
                                        )
                                    )
                                    for k, q in distribution.items()
                                }
                            )

                continue
            partners = partners.union(future_partners)
            partners = list(partners)

            # 買い手の場合のみ必要数量を調整
            adjusted_needs = int(needs * (1 + self.over_buying) if is_buyer else needs)

            if (
                self.awi.current_step > max(50, self.awi.n_steps * 0.5)
                and self.awi.level == 0
                and len(partners) > 0
            ):
                concentrated_ids = sorted(
                    partners,
                    key=lambda x: (
                        (self.offer_accept_count.get(x, 0) + 1)
                        / (self.offer_seen_count.get(x, 0) + 2),
                        self.total_agreed_quantity.get(x, 0),
                    ),
                    reverse=True,
                )
                concentrated_idx = []
                for p in concentrated_ids:  # concentrated_idsの順序で処理
                    if p in partners:
                        idx = partners.index(p)
                        concentrated_idx.append(idx)
                distribution = dict(
                    zip(
                        partners,
                        distribute(
                            adjusted_needs,
                            len(partners),
                            mx=3,
                            equal=True,
                            concentrated=True,
                            concentrated_idx=concentrated_idx,
                            allow_zero=False,
                        ),
                    )
                )
            elif (
                self.awi.current_step > max(100, self.awi.n_steps * 0.75)
                and self.awi.level == 1
                and len(partners) > 0
            ):
                concentrated_ids = sorted(
                    partners,
                    key=lambda x: (
                        (self.offer_accept_count.get(x, 0) + 1)
                        / (self.offer_seen_count.get(x, 0) + 2),
                        self.total_agreed_quantity.get(x, 0),
                    ),
                    reverse=True,
                )
                concentrated_idx = []
                for p in concentrated_ids:  # concentrated_idsの順序で処理
                    if p in partners:
                        idx = partners.index(p)
                        concentrated_idx.append(idx)
                distribution = dict(
                    zip(
                        partners,
                        distribute(
                            adjusted_needs,
                            len(partners),
                            mx=3,
                            equal=True,
                            concentrated=True,
                            concentrated_idx=concentrated_idx,
                            allow_zero=False,
                        ),
                    )
                )
            else:
                distribution = dict(
                    zip(
                        partners,
                        distribute(
                            adjusted_needs,
                            len(partners),
                            mx=3,
                            equal=True,
                            allow_zero=False,
                        ),
                    )
                )

            response.update(
                {
                    k: (
                        unneeded_response
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, self._counter_price_for(k, is_selling, issues))
                        )
                    )
                    for k, q in distribution.items()
                }
            )
        return response

    def _step_and_price(self, best_price=False):
        s = self.awi.current_step
        # seller フラグと関連する issue を決定
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value

        # best_price は極端値（最も好ましい価格）を返す
        if best_price:
            return s, (pmax if seller else pmin)

        # 相対時間 t in [0,1]
        t = float(self.awi.current_step) / max(1, float(self.awi.n_steps))
        t = max(0.0, min(1.0, t))

        # 滑らかな譲歩関数 f(t)=t**alpha
        f_t = t ** float(getattr(self, "concession_alpha", 2.0))

        # 初期値 p0 と目標値 p_target を売り手/買い手で設定
        if seller:
            p0 = float(pmax)
            p_target = float(pmin)
        else:
            p0 = float(pmin)
            p_target = float(pmax)

        # 線形補間に譲歩ファクタを乗じた価格
        price_f = p0 + (p_target - p0) * f_t
        price = int(round(price_f))

        # clamp
        price = max(int(pmin), min(int(pmax), price))

        return s, price
