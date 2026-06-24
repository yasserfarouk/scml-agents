from __future__ import annotations

import random

from negmas import Contract, ResponseType, SAOResponse

# required for development
from scml.oneshot import *

from scml.utils import anac2024_oneshot
from scml_agents import get_agents

__all__ = ["IsobeAgent"]
def powerset(iterable):
    """べき集合"""
    from itertools import chain, combinations

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class IsobeAgent(OneShotSyncAgent):

    # ------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------
    def init(self):
        self.last_processed_step = self.awi.current_step
        self.finalized_day_steps: set[int] = set()
        # 日次コスト追跡用状態
        self.day_tracking_step = self.awi.current_step
        self.day_target_supplies = max(0, int(self.awi.needed_supplies))
        self.day_target_sales = max(0, int(self.awi.needed_sales))
        self.day_agreed_supplies = 0
        self.day_agreed_sales = 0
        # overordering適応制御（W互換）
        self.overordering_max_selling: float = 0.12
        self.overordering_max_buying: float = 0.24
        self.overordering_min: float = 0.0
        self.overordering_exp: float = 0.4
        self.overordering_max: float = (
            self.overordering_max_selling
            if self.awi.level == 0
            else self.overordering_max_buying
        )
        self._day_plus: list[int] = []
        self._day_minus: list[int] = []
        self._day_perfect: list[int] = []

        # 相手評価
        # 1) 交渉成立回数(累積成立回数を相対化)
        # 2) 価格妥協率(良い価格/交渉回数)
        # 3) 成立時平均量
        all_partners = {
            p for p in (self.awi.my_suppliers + self.awi.my_consumers) if p != "SELLER"
        }
        self.partner_activity_score = {p: 0.5 for p in all_partners}
        self.partner_price_compromise = {p: 0.5 for p in all_partners}
        self.partner_avg_quantity = {p: 1.0 for p in all_partners}
        self.partner_success_count = {p: 0 for p in all_partners}

        self.day_partner_negotiations = {p: 0 for p in all_partners}
        self.day_partner_successes = {p: 0 for p in all_partners}
        self.day_partner_good_prices = {p: 0 for p in all_partners}
        self.day_partner_quantity = {p: 0 for p in all_partners}
        # 同一日の同一提案連打を抑えるための履歴
        self.step_proposal_history: dict[int, dict[tuple[tuple[str, int], ...], int]] = {}
        return super().init()



    # ------------------------------------------------------------
    # 提案生成
    # ------------------------------------------------------------
    def first_proposals(self):
        self._roll_day_if_needed()
        s, p = self._step_and_price(best_price=True)
        distribution = self.distribute_needs(t=0.0, mx=3, round_idx=0)
        d = {
            k: (q, s, p) if q > 0 or self.awi.allow_zero_quantity else None
            for k, q in distribution.items()
        }
        return d

    def _step_and_price(self, best_price=False):
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        if best_price:
            return s, pmax if seller else pmin
        return s, random.choice([pmin, pmax])

    def _propose_price(self, issues, relative_time: float, is_selling: bool):
        """相対時間に応じて、最高値/最低値のどちらを出すかを切り替える。"""
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value

        if is_selling:
            return pmin if relative_time >= 0.85 else pmax
        else:
            return pmax if relative_time >= 0.75 else pmin

    def distribute_needs(
        self,
        t: float = 0.0,
        mx: int | None = None,
        round_idx: int = 0,
    ) -> dict[str, int]:
        """相手評価(交渉頻度/価格妥協率/平均成立量)で必要数量を配分する"""
        dist = dict()
        step = self.awi.current_step
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            partners = [p for p in all_partners if p in self.negotiators]
            if needs <= 0:
                dist.update(dict.fromkeys(partners, 0))
                continue

            if len(partners) == 1:
                total_need = self._target_quantity(int(needs), t, 1)
                dist[partners[0]] = min(mx if mx is not None else total_need, total_need)
                continue

            total_need = self._target_quantity(int(needs), t, len(partners))
            dist.update(
                self._select_diversified_distribution(
                    total_need,
                    partners,
                    mx=mx,
                    round_idx=round_idx,
                    step=step,
                )
            )
        return dist

    def _distribute_by_partner_score(
        self, total_need: int, partners: list[str], mx: int | None = None
    ) -> dict[str, int]:
        """1,2の評価値で相手を評価し、3の平均成立量で配分先を補強する"""
        if not partners:
            return {}
        if total_need <= 0:
            return {p: 0 for p in partners}

        max_avg_q = max(max(self.partner_avg_quantity.get(p, 1.0), 0.1) for p in partners)
        scored = []
        for p in partners:
            activity_score = self.partner_activity_score.get(p, 0.5)
            price_compromise = self.partner_price_compromise.get(p, 0.5)
            avg_q = self.partner_avg_quantity.get(p, 1.0)

            # 1と2でエージェント評価し、3で最終提案量の寄せ先を補強
            eval_score = 0.6 * activity_score + 0.4 * price_compromise
            quantity_score = max(0.1, avg_q) / max_avg_q
            quantity_anchor = 0.5 + 0.5 * quantity_score
            scored.append(max(0.05, eval_score * quantity_anchor))

        weight_sum = sum(scored)
        if weight_sum <= 0:
            return dict(zip(partners, self._equal_distribute(total_need, partners, mx=mx)))

        raw = [total_need * w / weight_sum for w in scored]
        alloc = [int(x) for x in raw]
        frac = [(raw[i] - alloc[i], i) for i in range(len(raw))]

        remain = total_need - sum(alloc)
        for _, i in sorted(frac, reverse=True):
            if remain <= 0:
                break
            alloc[i] += 1
            remain -= 1

        if mx is not None:
            # 上限超過分は、まだ余力のある相手へ再配分する
            overflow = 0
            for i in range(len(alloc)):
                if alloc[i] > mx:
                    overflow += alloc[i] - mx
                    alloc[i] = mx

            while overflow > 0:
                candidates = [i for i in range(len(alloc)) if alloc[i] < mx]
                if not candidates:
                    break
                candidates.sort(key=lambda i: scored[i], reverse=True)
                for i in candidates:
                    if overflow <= 0:
                        break
                    room = mx - alloc[i]
                    if room <= 0:
                        continue
                    add = min(room, overflow)
                    alloc[i] += add
                    overflow -= add

        return {p: q for p, q in zip(partners, alloc)}

    def _partner_weight(self, partner: str, peers: list[str] | None = None) -> float:
        """提案先選択に使う相手重み。既存の3指標をそのまま活用する。"""
        if not peers:
            peers = [partner]

        max_avg_q = max(max(self.partner_avg_quantity.get(p, 1.0), 0.1) for p in peers)
        activity = self.partner_activity_score.get(partner, 0.5)
        price_compromise = self.partner_price_compromise.get(partner, 0.5)
        avg_q = max(0.1, self.partner_avg_quantity.get(partner, 1.0))
        quantity_score = avg_q / max_avg_q
        return max(
            0.01,
            (0.6 * activity + 0.4 * price_compromise)
            * (0.5 + 0.5 * quantity_score),
        )

    def _sample_distribution_by_weights(
        self,
        total_need: int,
        partners: list[str],
        weights: list[float],
        mx: int | None = None,
    ) -> dict[str, int]:
        """重みに従って整数配分をサンプリングする。"""
        if not partners or total_need <= 0:
            return {p: 0 for p in partners}

        alloc = {p: 0 for p in partners}
        for _ in range(total_need):
            candidates = []
            candidate_weights = []
            for i, p in enumerate(partners):
                if mx is not None and alloc[p] >= mx:
                    continue
                candidates.append(p)
                candidate_weights.append(max(0.001, weights[i]))
            if not candidates:
                break
            chosen = random.choices(candidates, weights=candidate_weights, k=1)[0]
            alloc[chosen] += 1
        return alloc

    def _distribution_signature(
        self, distribution: dict[str, int]
    ) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(distribution.items()))

    def _remember_distribution(self, step: int, distribution: dict[str, int]) -> None:
        hist = self.step_proposal_history.setdefault(step, {})
        sig = self._distribution_signature(distribution)
        hist[sig] = hist.get(sig, 0) + 1

    def _select_diversified_distribution(
        self,
        total_need: int,
        partners: list[str],
        mx: int | None,
        round_idx: int,
        step: int,
        relative_time: float | None = None,
    ) -> dict[str, int]:
        """1回目は最適配分、2回目以降は評価ベース候補から重み付き抽選で選ぶ。"""
        if not partners:
            return {}

        base = self._distribute_by_partner_score(total_need, partners, mx=mx)

        # 探索弱化: 終盤は探索せず、評価ベースの最適配分に固定する。
        if relative_time is not None and relative_time >= 0.7:
            self._remember_distribution(step, base)
            return base

        if round_idx <= 0 or len(partners) <= 1:
            self._remember_distribution(step, base)
            return base

        candidates = [base]
        base_weights = [self._partner_weight(p, partners) for p in partners]
        n_trials = min(8, 3 + round_idx)
        noise_scale = min(0.45, 0.15 + 0.05 * round_idx)

        for _ in range(n_trials):
            noisy_weights = [
                w * random.uniform(1.0 - noise_scale, 1.0 + noise_scale)
                for w in base_weights
            ]
            sampled = self._sample_distribution_by_weights(
                total_need,
                partners,
                noisy_weights,
                mx=mx,
            )
            candidates.append(sampled)

        history = self.step_proposal_history.setdefault(step, {})
        scored_candidates: list[tuple[dict[str, int], float]] = []
        for dist in candidates:
            expected = sum(dist[p] * self._partner_weight(p, partners) for p in partners)
            repeat_count = history.get(self._distribution_signature(dist), 0)
            score = expected - 0.35 * repeat_count
            scored_candidates.append((dist, max(0.001, score)))

        scores = [s for _, s in scored_candidates]
        picked = random.choices(scored_candidates, weights=scores, k=1)[0][0]
        self._remember_distribution(step, picked)
        return picked

    def _equal_distribute(
        self, q: int, partners: list[str], mx: int | None = None
    ) -> list[int]:
        """均等配分（余りをランダムに割り当て）"""
        n = len(partners)
        if n == 0:
            return []
        if q <= 0:
            return [0] * n
        if mx is not None:
            q = min(q, mx * n)
        base, remainder = divmod(q, n)
        allocation = [base] * n
        for i in random.sample(range(n), remainder):
            allocation[i] += 1
        return allocation

    #オーバーオーダリングのための数量調整関数
    def _overordering_fraction(self, t: float) -> float:
        """水増し係数（t=0で最大、t=1で最小）。relative_time で交渉内変動にも転用"""
        return self.overordering_min + (self.overordering_max - self.overordering_min) * (
            (1.0 - t) ** self.overordering_exp
        )

    def _overordered_quantity(self, needs: int, t: float) -> int:
        """相手数や経過時間を踏まえた overordering 後の数量"""
        if needs <= 0:
            return 0
        return max(0, int(needs * (1.0 + self._overordering_fraction(t))))

    def _target_quantity(self, needs: int, t: float, n_partners: int) -> int:
        """相手が1人のときは overordering せず、複数相手のときのみ適用する"""
        if needs <= 0:
            return 0
        if n_partners <= 1:
            return int(needs)
        return self._overordered_quantity(int(needs), t)

    # ------------------------------------------------------------
    # 交渉応答
    # ------------------------------------------------------------
    def counter_all(self, offers, states):
        self._roll_day_if_needed()

        response = dict()
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
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
            # find active partners in some random order
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)
            is_selling = all_partners == self.awi.my_consumers
            relative_time = min(state.relative_time for state in states.values())
            round_idx = min(state.step for state in states.values()) if states else 0
            price = self._propose_price(issues, relative_time, is_selling)

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER, (0, self.awi.current_step, price)
                )
            )

            # 交渉相手の提案の組み合わせを評価
            plist = list(powerset(partners))[::-1]

            # 最適な組み合わせを選択するためのカウンター
            best_offer_score, best_quantity, best_indx = float("-inf"), 0, -1

            # 各組み合わせを評価する際に実際の必要数を使用
            for i, agent_ids in enumerate(plist):
                offered_quantity = sum(offers[p][QUANTITY] for p in agent_ids)
                total_price = sum(
                    offers[p][UNIT_PRICE] * offers[p][QUANTITY] for p in agent_ids
                )
                # 関数化: スコア計算と数量差を一度に取得
                offer_score, quantity_diff = self._compute_price_score(
                    offered_quantity,
                    needs,
                    total_price,
                    is_selling,
                    relative_time,
                    max_unit_price=issues[UNIT_PRICE].max_value,
                )

                # 最良の組み合わせを更新
                if offer_score > best_offer_score:
                    best_offer_score = offer_score
                    best_quantity = quantity_diff
                    best_indx = i
                elif offer_score == best_offer_score:
                    # 同じスコアの場合、売り手は過剰を、買い手は不足を避ける
                    if (is_selling and quantity_diff < best_quantity) or (
                        not is_selling
                        and quantity_diff > best_quantity
                        and quantity_diff <= 0
                    ):
                        best_quantity = quantity_diff
                        best_indx = i

            # 動的妥協率のみで受理閾値を決定
            adjusted_threshold = self._compute_acceptance_threshold(relative_time)

            # 意思決定: スコアが閾値を超えた場合、提案を受け入れる
            if best_indx >= 0 and best_offer_score >= adjusted_threshold:
                agent_ids = plist[best_indx]
                others = list(partners.difference(agent_ids).union(future_partners))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in agent_ids
                } | {k: unneeded_response for k in others}

                # 不足数量がある場合、他の相手に対して追加提案
                if best_quantity < 0 and len(others) > 0:
                    s = self.awi.current_step
                    price = self._propose_price(issues, relative_time, is_selling)

                    # 不足数量を計算
                    shortage = self._target_quantity(-best_quantity, relative_time, len(others))

                    distribution = self._select_diversified_distribution(
                        shortage,
                        list(others),
                        mx=3,
                        round_idx=round_idx,
                        step=self.awi.current_step,
                        relative_time=relative_time,
                    )
                    response.update(
                        {
                            k: (
                                unneeded_response
                                if q == 0
                                else SAOResponse(
                                    ResponseType.REJECT_OFFER,
                                    (q, self.awi.current_step, price),
                                )
                            )
                            for k, q in distribution.items()
                        }
                    )

                continue

            partners = partners.union(future_partners)
            partners = list(partners)

            adjusted_needs = self._target_quantity(needs, relative_time, len(partners))

            distribution = self._select_diversified_distribution(
                adjusted_needs,
                partners,
                mx=3,
                round_idx=round_idx,
                step=self.awi.current_step,
                relative_time=relative_time,
            )

            response.update(
                {
                    k: (
                        unneeded_response
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                        )
                    )
                    for k, q in distribution.items()
                }
            )
        return response

    def _compute_acceptance_threshold(self, relative_time: float) -> float:
        if self.awi.level == 0:  # 売り手
            threshold = 0.9 - 0.4 * (relative_time ** 0.5)
        else:  # 買い手 乗数を大きくすると妥協が加速
            threshold = 0.9 - 0.4 * (relative_time ** 1.8)
        return max(0.0, min(1.0, threshold))

    def _compute_price_score(
        self,
        offered_quantity: int,
        needs: int,
        total_price: int,
        is_selling: bool,
        relative_time: float,
        max_unit_price: float = 11.0,
    ) -> tuple[float, int]:
        """相手提案に対するスコアを計算（数量差は同点時の判定用）"""
        quantity_diff = offered_quantity - needs

        # 交渉relative_timeに応じた調整係数
        
        # ペナルティ計算
        penalty = 0
        if is_selling:  # 売り手
            if offered_quantity > needs:  # 過剰販売により、在庫不足
                penalty += (
                    offered_quantity - needs
                ) * self.awi.current_shortfall_penalty
            if offered_quantity < needs:  # 販売不足により、在庫廃棄
                penalty += (
                    needs - offered_quantity
                ) * self.awi.current_disposal_cost * (0.5 + relative_time * 0.5)
        else:  # 買い手
            if offered_quantity < needs:  # 購入不足により、在庫不足
                penalty += (
                    needs - offered_quantity
                ) * self.awi.current_shortfall_penalty * (0.2 + relative_time * 2)
            if offered_quantity > needs:  # 過剰購入により、在庫廃棄
                penalty += (
                    offered_quantity - needs
                ) * self.awi.current_disposal_cost * (1 - relative_time * 0.3)
        # 価格スコアを計算
        if is_selling:
            # 売り手: 収入 - ペナルティ（高いほど良い）
            score = total_price - penalty
            scale = max_unit_price * needs
            offer_score = max(0.0, min(1.0, 0.5 + score / (2.0 * scale))) if scale > 0 else 0.0
        else:
            if relative_time < 0.5:
                scale = needs * self.awi.current_shortfall_penalty
                P_score = max_unit_price - (total_price / max(1, offered_quantity)) if offered_quantity > 0 else 0
                score = max(0.0, scale - penalty)
                offer_score = max(0.0, min(1.0, score / (2.0 * scale) + P_score / (2.0 * max_unit_price))) if scale > 0 else 0.0
            else:
                scale = needs * self.awi.current_shortfall_penalty
                score = max(0.0, scale - penalty)
                offer_score = max(0.0, min(1.0, 0.5 + score / (2.0 * scale))) if scale > 0 else 0.0

            if relative_time > 0.9:
                if (needs * self.awi.current_shortfall_penalty) > total_price:
                    offer_score = max(offer_score * 10, 0.55)  # 最終段階で不足ペナルティが価格を上回る場合は受け入れやすくする
        return offer_score, quantity_diff

    # ------------------------------------------------------------
    # 合意処理
    # ------------------------------------------------------------
    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        agent_id = [p for p in contract.partners if p != self.id][0]
        quantity = int(contract.agreement["quantity"])

        self._touch_partner(agent_id)
        self.day_partner_negotiations[agent_id] += 1
        self.day_partner_successes[agent_id] += 1
        self.day_partner_quantity[agent_id] += quantity

        price = float(contract.agreement["unit_price"])
        if agent_id in self.awi.my_consumers:
            issues = self.awi.current_output_issues
            pmin = float(issues[UNIT_PRICE].min_value)
            pmax = float(issues[UNIT_PRICE].max_value)
            good_price = price >= (pmin + pmax) / 2
        else:
            issues = self.awi.current_input_issues
            pmin = float(issues[UNIT_PRICE].min_value)
            pmax = float(issues[UNIT_PRICE].max_value)
            good_price = price <= (pmin + pmax) / 2
        if good_price:
            self.day_partner_good_prices[agent_id] += 1

        if agent_id in self.awi.my_suppliers:
            self.day_agreed_supplies += quantity
        elif agent_id in self.awi.my_consumers:
            self.day_agreed_sales += quantity

    def on_negotiation_failure(self, partners, annotation, mechanism, state) -> None:
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        partner = next((p for p in partners if p != self.id), None)
        if partner is None:
            return
        self._touch_partner(partner)
        self.day_partner_negotiations[partner] += 1


    # ------------------------------------------------------------
    # 日次処理
    # ------------------------------------------------------------
    def _roll_day_if_needed(self) -> None:
        # 追加: step切り替わり時に前日データを確定
        current_step = self.awi.current_step
        if current_step != self.last_processed_step:
            self._finalize_previous_day(self.last_processed_step)
            self.last_processed_step = current_step
            self._start_new_day_tracking(current_step)

    def _start_new_day_tracking(self, step: int) -> None:
        self.day_tracking_step = step
        self.day_target_supplies = max(0, int(self.awi.needed_supplies))
        self.day_target_sales = max(0, int(self.awi.needed_sales))
        self.day_agreed_supplies = 0
        self.day_agreed_sales = 0
        for p in self.day_partner_negotiations:
            self.day_partner_negotiations[p] = 0
            self.day_partner_successes[p] = 0
            self.day_partner_good_prices[p] = 0
            self.day_partner_quantity[p] = 0

    def _finalize_previous_day(self, day_step: int) -> None:
        if day_step in self.finalized_day_steps:
            return
        self._update_dynamic_compromise_from_day_result()
        self.finalized_day_steps.add(day_step)

    def _update_dynamic_compromise_from_day_result(self) -> None:
        # 不足コスト/廃棄コストの差で受理閾値シフトを更新
        shortfall_units = max(0, self.day_target_supplies - self.day_agreed_supplies) + max(
            0, self.day_agreed_sales - self.day_target_sales
        )
        disposal_units = max(0, self.day_agreed_supplies - self.day_target_supplies) + max(
            0, self.day_target_sales - self.day_agreed_sales
        )

        # overordering適応更新: 直近10日ウィンドウでexp調整
        if shortfall_units == 0 and disposal_units == 0:
            self._day_perfect.append(1); self._day_plus.append(0); self._day_minus.append(0)
        elif disposal_units > 0:
            self._day_plus.append(1); self._day_perfect.append(0); self._day_minus.append(0)
        else:
            self._day_minus.append(1); self._day_perfect.append(0); self._day_plus.append(0)
        window = 10
        n_days = len(self._day_plus)
        before = max(0, n_days - window)
        plus  = sum(self._day_plus[before:])  / max(1, n_days - before)
        minus = sum(self._day_minus[before:]) / max(1, n_days - before)
        if plus > minus and plus > 0:    # 廃棄多い → 水増し抑制
            self.overordering_exp = min(self.overordering_exp * 1.3, 5.0)
            self.overordering_max = max(
                self.overordering_min,
                self.overordering_max * 0.95,
            )
        elif minus > plus and minus > 0: # 不足多い → 水増し増加
            self.overordering_exp = max(self.overordering_exp * 0.7, 0.1)
            self.overordering_max = min(
                0.5,
                self.overordering_max * 1.05,
            )
        else:
            target_max = (
                self.overordering_max_selling
                if self.awi.level == 0
                else self.overordering_max_buying
            )
            self.overordering_max = 0.7 * self.overordering_max + 0.3 * target_max

        # 相手ごとの1,2,3指標を日次更新

        # 1) 交渉成立回数(累積成立回数を相対化)
        for p, succ in self.day_partner_successes.items():
            self.partner_success_count[p] += succ
        max_success_count = max(1, max(self.partner_success_count.values(), default=1))
        for p in self.partner_activity_score:
            self.partner_activity_score[p] = self.partner_success_count[p] / max_success_count

        # 2) 価格妥協率, 3) 成立時平均量 は日次EMAで更新
        alpha = 0.35
        for p, n in self.day_partner_negotiations.items():
            if n <= 0:
                continue
            succ = self.day_partner_successes[p]
            good = self.day_partner_good_prices[p]
            qty = self.day_partner_quantity[p]

            day_price_compromise = min(good, n) / n
            self.partner_price_compromise[p] = (
                (1.0 - alpha) * self.partner_price_compromise[p]
                + alpha * day_price_compromise
            )

            if succ > 0:
                day_avg_quantity = qty / succ
                self.partner_avg_quantity[p] = (
                    (1.0 - alpha) * self.partner_avg_quantity[p]
                    + alpha * day_avg_quantity
                )

    def _touch_partner(self, partner: str) -> None:
        if partner in self.partner_activity_score:
            return
        self.partner_activity_score[partner] = 0.5
        self.partner_price_compromise[partner] = 0.5
        self.partner_avg_quantity[partner] = 1.0
        self.partner_success_count[partner] = 0
        self.day_partner_negotiations[partner] = 0
        self.day_partner_successes[partner] = 0
        self.day_partner_good_prices[partner] = 0
        self.day_partner_quantity[partner] = 0
