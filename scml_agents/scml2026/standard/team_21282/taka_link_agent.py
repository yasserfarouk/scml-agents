# TakaLinkAgent.py
#!/usr/bin/env python
from __future__ import annotations

from typing import Any
from itertools import chain, combinations


def powerset(iterable):
    """Return the powerset of the iterable as tuples (excluding the empty set if caller filters).
    Compatible with usage in this module: list(powerset(...))
    """
    s = list(iterable)
    return map(tuple, chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
import sys
from scml.std import StdAWI, StdSyncAgent 
from scml.oneshot.common import *
from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType

# 自作オリジナルモジュール群のインポート
from .market_analyst import MarketAnalyst
from .partner_evaluator import PartnerEvaluator
from .trade_allocator import TradeAllocator
from .smart_pricing import SmartPricing

class TakaLinkAgent(StdSyncAgent):
    """TakaLinkAgent: SCML 用の交渉エージェント。
    MarketAnalyst / PartnerEvaluator / TradeAllocator / SmartPricing を組み合わせて
    発注・販売・対抗応答・提案生成を行う。
    """

    # 役割: 当日安全に販売可能な最大数を計算するユーティリティ
    def max_safe_sales_today(self):
        """当日安全に販売できる最大数量を返す。

        在庫（input）と当日の入荷予測を合算し、ライン数で上限をかけて整数で返す。
        """
        inv = self.awi.current_inventory_input
        incoming = self.awi.total_supplies_at(self.awi.current_step)
        return int(min(self.awi.n_lines, inv + incoming))
    
    def inventory_policy(self):
        """在庫ターゲットと絶対上限を一元管理する。"""
        L = self.awi.n_lines
        t = self.awi.current_step
        N = self.awi.n_steps

        # 残り期間に応じて少しだけ厚め/薄めを切り替える
        remaining_ratio = (N - t) / max(1, N)

        # まずは安全寄りに設定
        target_inventory = int(L * (1.0 + 0.4 * remaining_ratio))   # 1.0L～1.4L
        max_inventory = int(L * (1.8 + 0.3 * remaining_ratio))      # 1.8L～2.1L
        return target_inventory, max_inventory

    def projected_incoming(self):
        """将来の確定入荷をAWIベースで取得する。二重計上を避ける。"""
        return sum(
            self.awi.total_supplies_at(step)
            for step in range(self.awi.current_step, self.awi.n_steps)
        )


    def projected_outgoing(self):
        """将来の確定販売を自前リストから取得する。"""
        return sum(
            c.agreement.get("quantity", 0)
            for c in getattr(self, "future_selling_contracts", [])
            if c.agreement.get("time", self.awi.current_step) >= self.awi.current_step
        )
    
    def remaining_production_capacity(self):
        """残りステップで最大何 unit 生産できるかを返す。"""
        remaining_steps = max(0, self.awi.n_steps - self.awi.current_step)
        return self.awi.n_lines * remaining_steps

    def sell_capacity_estimate(self):
        """これから確実寄りに売れる総量の推定。"""
        # 手元 input + 将来入荷予定
        total_input_like = self.awi.current_inventory_input + self.projected_incoming()

        # 残り期間で生産できる総上限
        prod_cap = self.remaining_production_capacity()

        # input は生産能力を超えても使い切れないので cap をかける
        feasible_output = min(total_input_like, prod_cap)

        return feasible_output

    def remaining_sellable_capacity(self):
        """追加で売ってよい残余キャパを返す。"""
        secured_selling = sum(
            c.agreement.get("quantity", 0)
            for c in getattr(self, "future_selling_contracts", [])
            if c.agreement.get("time", self.awi.current_step) >= self.awi.current_step
        )

        # 少し安全マージンを引く（floor改善狙い）
        safety_margin = max(1, int(0.05 * self.awi.n_lines))

        raw_capacity = self.sell_capacity_estimate() - secured_selling - safety_margin
        return max(0, raw_capacity)
    
    def remaining_sellable_capacity_effective(self):
        """step 内仮予約も考慮した実効売り余力。"""
        base = self.remaining_sellable_capacity()
        reserved = getattr(self, "_reserved_sell_qty", 0)
        return max(0, base - reserved)


    def remaining_buyable_capacity_effective(self):
        """step 内仮予約も考慮した実効買い余力（今は保険）。"""
        target_inventory, max_inventory = self.inventory_policy()
        current_total_input = self.awi.current_inventory_input + self.projected_incoming()
        reserved = getattr(self, "_reserved_buy_qty", 0)
        return max(0, target_inventory - current_total_input - reserved)
    
    def need_emergency_buy(self):
        """初期2stepだけ使う緊急調達モード。bootstrap専用。"""
        return (
            self.awi.current_step <= 2
            and self.awi.current_step < self.awi.n_steps - 1
            and self.awi.current_inventory_input <= 0
            and self.projected_incoming() <= 0
            and self.awi.needed_supplies > 0
        )



    def emergency_buy_target(self):
        """緊急時に最低限確保したい数量（少し弱め）。"""
        L = self.awi.n_lines

        # bootstrap 目的なので、まずは半ライン分くらい確保
        base = max(int(0.5 * L), min(int(self.awi.needed_supplies), L))

        return max(1, min(L, base))


    # 役割: エージェント内部状態とサブコンポーネントを初期化する
    def __init__(self, *args, threshold=None, ptoday=0.70, productivity=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        if threshold is None:
            threshold = 1
        self._threshold = threshold  
        self._productivity = productivity  
        self.production_level = 0.5
        self.future_concession = 0.1  
        
        # 各種専門コンポーネントモジュールの初期化
        self.analyst = MarketAnalyst()
        self.evaluator = PartnerEvaluator()
        
        # 内部互換用変数
        self.p_max_for_buying = None  
        # 変更: 将来契約のトラッキング用リストを明示的に保持
        self.future_selling_contracts = []  # 既存: 将来の販売契約を追跡
        self.future_buying_contracts = []   # 追加: 将来の買い契約を追跡（修正要求1）
        self.projected_sales = 0
        self.max_inventory = 0
        self.cumulative_input = 0

    # 役割: AWI 接続後に一度だけ呼ばれる初期化処理
    def init(self):
        """AWI 初期化後の内部状態をリセットする。

        - `cumulative_input` をゼロにする
        - `max_inventory` と `projected_sales` を初期化する
        - `analyst` の状態を現在残高でリセットする
        """
        self.cumulative_input = 0
        self.projected_sales = 0
        self.analyst.reset(self.awi.current_balance)

        target_inventory, max_inventory = self.inventory_policy()
        self.target_buffer = target_inventory
        self.max_capacity = max_inventory
        self.max_inventory = max_inventory
        # 保険: 将来契約リストが未定義の場合は空で初期化
        self.future_selling_contracts = getattr(self, 'future_selling_contracts', [])
        self.future_buying_contracts = getattr(self, 'future_buying_contracts', [])
        # step 内の仮予約量（未確定だがこの step で使う予定の量）
        self._reserved_sell_qty = 0
        self._reserved_buy_qty = 0


    # --- サブモジュール群へのアクセスを仲介するブリッジメソッド ---
    # 役割: パートナーの信頼度/スコア（sigma）を返すブリッジ
    def get_sigma(self, partner: str) -> float:
        """PartnerEvaluator からパートナーの sigma 値を取得する。

        交渉や分配の重み付けに使う。
        """
        return self.evaluator.get_sigma(partner)

    # 役割: 利用可能なパートナー群から交渉対象を選ぶ
    def select_partners(self, all_partners: list[str]) -> list[str]:
        """PartnerEvaluator を使って与えられたリストから選択されたパートナーを返す。

        `analyst.ptoday` と `analyst.mode` を渡して選択基準を調整する。
        """
        return self.evaluator.select_partners(all_partners, self.analyst.ptoday, self.analyst.mode)

    # 役割: 必要量をパートナーに均等分配（sigma を利用）
    def distribute_even(self, total_needed: int, partners: list[str]) -> dict[str, int]:
        """TradeAllocator を呼び出して需要をパートナーに分配するラッパー。

        パートナーの `sigma` を重みとして利用する。
        """
        return TradeAllocator.distribute_even(total_needed, partners, self.get_sigma)

    # 役割: SmartPricing モジュールを使って提案価格を決定する
    def get_smart_price(self, partner: str, is_seller: bool) -> int:
        """SmartPricing に委譲しつつ、在庫超過時だけ seller price を追加で緩める。"""
        price = SmartPricing.get_price(
            self.get_nmi(partner),
            self.awi,
            is_seller,
            self.get_sigma(partner),
            self.analyst.mode
        )

        if is_seller:
            _, max_inventory = self.inventory_policy()
            inv = self.awi.current_inventory_input

            # 在庫超過なら、さらに 5〜10% ほど下げて売りやすくする
            if inv > max_inventory:
                # 超過率に応じて少しだけ強く下げる
                over_ratio = min(0.10, 0.03 + 0.02 * ((inv - max_inventory) / max(1, self.awi.n_lines)))
                discounted = int(price * (1.0 - over_ratio))
                return max(1, discounted)

        return price

    # 役割: 提案数量が当該パートナー・納期で必要かを判定する
    def is_needed(self, partner: str, offer: Any) -> bool:
        """与えられた `offer` の数量が `_needs(partner, time)` を満たすか判定する。

        Offer が None の場合は False を返す。
        """
        if offer is None:
            return False
        return offer[QUANTITY] <= self._needs(partner, offer[TIME])

    # 役割: 各ステップ開始時に呼ばれ、市場分析と販売予測を更新する
    def before_step(self):
        """ステップ開始時に呼ばれるフック。

        - `analyst` を更新して意思決定モード・交渉枠を同期する
        - 過去数ステップの売上から `projected_sales` を算出する
        """
        t = self.awi.current_step
        N = self.awi.n_steps
        L = self.awi.n_lines

        # ---- step reservation reset ----
        self._reserved_sell_qty = 0
        self._reserved_buy_qty = 0

        # 変更: ステップ開始時に将来契約リストをクリーンアップする（修正要求1）
        # agreement['time'] が現在ステップより小さい（到来済）の契約は削除する
        def _cleanup_contracts(contract_list: list):
            kept = []
            for c in contract_list:
                try:
                    ctime = c.agreement.get('time', None)
                except Exception:
                    ctime = None
                # time が不明（None）の場合は安全のため保持
                if ctime is None or ctime >= self.awi.current_step:
                    kept.append(c)
            return kept

        self.future_buying_contracts = _cleanup_contracts(getattr(self, 'future_buying_contracts', []))
        self.future_selling_contracts = _cleanup_contracts(getattr(self, 'future_selling_contracts', []))

        # 市場トレンドと意思決定モード、交渉枠上限の同期更新
        self.analyst.update_mode_and_cap(
            current_balance=self.awi.current_balance,
            current_step=t,
            n_steps=N,
            n_lines=L,
            current_inventory=self.awi.current_inventory
        )

        if self.p_max_for_buying is None:
            try:
                self.p_max_for_buying = self.awi.current_input_issues[UNIT_PRICE].max_value
            except:
                self.p_max_for_buying = 100
        
        history_steps = 3
        sales_data = [self.awi.total_sales_at(t - i - 1) for i in range(min(history_steps, t))]
        avg_sales = sum(sales_data) / len(sales_data) if sales_data else L * 0.5
        inventory_factor = max(0.0, 1.0 - (self.awi.current_inventory_input / (L * 3.0)))
        decay = min(1.0, max(1, N - t) / N)
        self.projected_sales = max(1, int((avg_sales + L * 0.3) * inventory_factor * decay))

        target_inventory, max_inventory = self.inventory_policy()
        self.target_buffer = target_inventory
        self.max_capacity = max_inventory
        self.max_inventory = max_inventory

        target_inventory, max_inventory = self.inventory_policy()

    """
        print("=" * 60)
        print(f"[STEP {self.awi.current_step}]")
        print(f"mode={self.analyst.mode}")

        print(
            f"inventory_input={self.awi.current_inventory_input}, "
            f"inventory_output={self.awi.current_inventory_output}"
        )

        print(
            f"projected_incoming={self.projected_incoming()}, "
            f"projected_outgoing={self.projected_outgoing()}"
        )

        print(
            f"target_inventory={target_inventory}, "
            f"max_inventory={max_inventory}"
        )

        print(
            f"needed_supplies={self.awi.needed_supplies}, "
            f"needed_sales={self.awi.needed_sales}"
        )

        print(
            f"future_buying={sum(c.agreement.get('quantity',0) for c in self.future_buying_contracts)}, "
            f"future_selling={sum(c.agreement.get('quantity',0) for c in self.future_selling_contracts)}"
        )

        print("=" * 60)
    """

    # 役割: ステップ開始時に行うファーストオファーの生成
    def first_proposals(self):
        """初回提案（先方への最初のオファー）を構築して返す。

        - 仕入れ側（suppliers）向けの量・価格設定
        - フォワード予約（将来ステップ）の割当
        - 販売側（consumers）向けの分配と価格設定
        """
        offers = {}
        unneeded = None if not self.awi.allow_zero_quantity else (0, self.awi.current_step, 0)
        t = self.awi.current_step
        L = self.awi.n_lines

        # emergency buy 時は supplier selection を絞らず全員に広げる
        selected_suppliers = (
            list(self.awi.my_suppliers)
            if self.need_emergency_buy()
            else self.select_partners(self.awi.my_suppliers)
        )
        selected_consumers = self.select_partners(self.awi.my_consumers)

        # --- サプライヤー仕入れ交渉 ---
        target_inventory, max_inventory = self.inventory_policy()
        incoming = self.projected_incoming()
        available_input = self.awi.current_inventory_input + incoming

        if self.awi.my_consumers == ["BUYER"]:
            # BUYER モードは exogenous output を優先
            todays_input_needed = max(
                0,
                self.awi.current_exogenous_output_quantity
                - (min(self.awi.current_inventory_input, L) + self.awi.current_inventory_output)
            )
        else:
            # 通常時は target_inventory を目安に買う
            todays_input_needed = max(0, target_inventory - available_input)
            # ただし絶対上限を超えるなら今日は買わない
            if self.awi.current_inventory_input >= max_inventory:
                todays_input_needed = 0

        # emergency buy: 通常ロジックより優先して最低数量を取りに行く
        if self.need_emergency_buy():
            todays_input_needed = max(todays_input_needed, self.emergency_buy_target())

        distribution_suppliers = self.distribute_even(todays_input_needed, selected_suppliers)
        for k in self.awi.my_suppliers:
            if k in distribution_suppliers and distribution_suppliers[k] > 0:
                offers[k] = (distribution_suppliers[k], t, self.get_smart_price(k, is_seller=False))
            else:
                offers[k] = unneeded

        # フォワード先行予約の展開（今日提案を上書きしない）
        sorted_suppliers = sorted(selected_suppliers, key=lambda p: self.get_sigma(p), reverse=True)

        today_used_suppliers = {
            k for k, q in distribution_suppliers.items() if q > 0
        }

        # future は「今日使っていない supplier」だけに限定
        forward_supplier_pool = [p for p in sorted_suppliers if p not in today_used_suppliers]

        # future buy は，現在在庫が target よりかなり薄いときだけ限定的に行う
        target_inventory, max_inventory = self.inventory_policy()
        future_gap = max(0, target_inventory - (self.awi.current_inventory_input + self.projected_incoming()))

        if forward_supplier_pool and future_gap > int(0.3 * L):
            n_sup = len(forward_supplier_pool)
            p_prod = min(int(L * 0.4), future_gap)  # 予約量を縮小
            idx_50 = max(1, int(n_sup * 0.5))
            idx_30 = max(1, int(n_sup * 0.3))

            forward_assignments = [
                (t + 1, forward_supplier_pool[:idx_50]),
                (t + 2, forward_supplier_pool[idx_50:idx_50 + idx_30]),
                (t + 3, forward_supplier_pool[idx_50 + idx_30:])
            ]

            for f_step, sups in forward_assignments:
                if f_step < self.awi.n_steps and sups:
                    f_distribution = self.distribute_even(p_prod, sups)
                    for k, q in f_distribution.items():
                        if q > 0 and k not in offers:
                            offers[k] = (q, f_step, self.get_smart_price(k, is_seller=False))

        # --- 消費者販売交渉 ---
        remained_consumers = selected_consumers.copy()
        secured_output = sum(
            contract.agreement["quantity"]
            for contract in self.future_selling_contracts
            if contract.agreement.get("time", t) >= t
        )

        # 追加で売ってよい総量
        remaining_sell_cap = self.remaining_sellable_capacity_effective()

        """
        print(
            f"[SELL-FIRST] step={t} "
            f"remaining_sell_cap={remaining_sell_cap} "
            f"secured_output={secured_output} "
            f"supply_est={self.awi.current_inventory_input + self.projected_incoming()} "
            f"remaining_prod={self.remaining_production_capacity()}"
        )
        """

        for future_t in range(t, self.awi.n_steps):
            if self.awi.my_consumers == ["BUYER"]:
                break

            if remaining_sell_cap <= 0:
                break

            # 元の needed_sales ベースを残しつつ、sell cap で強制的に上限
            todays_output_needed = max(self.awi.needed_sales - secured_output, 0)
            todays_output_needed = min(todays_output_needed, remaining_sell_cap)

            if todays_output_needed <= 0:
                break

            if todays_output_needed <= L or future_t == self.awi.n_steps - 1:
                distribution = dict(
                    zip(remained_consumers, distribute(todays_output_needed, len(remained_consumers)))
                )
            else:
                capped_needed = min(todays_output_needed, L)
                concentrated_ids = sorted(
                    remained_consumers,
                    key=lambda x: self.evaluator.negotiation_history[x]["total_quantity"],
                    reverse=True
                )
                distribution = dict(
                    zip(
                        remained_consumers,
                        distribute(
                            capped_needed,
                            len(remained_consumers),
                            mx=L,
                            concentrated=True,
                            concentrated_idx=[
                                i for i, p in enumerate(remained_consumers) if p in concentrated_ids
                            ],
                            allow_zero=True
                        )
                    )
                )

            
            offered_now = 0

            for k, q in distribution.items():
                if q > 0:
                    offers[k] = (
                        q,
                        future_t,
                        self.get_smart_price(k, is_seller=True)
                    )
                    offered_now += q

            remained_consumers = [
                k
                for k in remained_consumers
                if k not in distribution.keys()
                or distribution[k] <= 0
            ]

            # ここでは reservation しない（提案はまだ未成立）
            secured_output += offered_now
            remaining_sell_cap -= offered_now

            if len(remained_consumers) == 0:
                break

                
        for k in self.awi.my_consumers:
            if k not in offers:
                offers[k] = unneeded
                
        return offers

    # 役割: 受け取ったオファー群に対して一括で対抗応答を作成する
    def counter_all(self, offers, states):
        """複数パートナーからのオファーに対して ACCEPT/REJECT を一括決定する。

        - tau による対象パートナー絞り込み
        - BUYER モード（特殊ルール）と通常の買い処理、販売処理を行う
        """
        active_suppliers = self.select_partners(self.awi.my_suppliers)
        active_consumers = self.select_partners(self.awi.my_consumers)

        # サプライ（買い）とコンシューマ（売り）で別々に上限を適用
        allowed_suppliers = [p for p in offers.keys() if p in active_suppliers]
        allowed_consumers = [p for p in offers.keys() if p in active_consumers]

        tau_buy = getattr(self.analyst, 'tau_buy', self.analyst.tau)
        tau_sell = getattr(self.analyst, 'tau_sell', self.analyst.tau)

        if len(allowed_suppliers) > tau_buy:
            allowed_suppliers = sorted(allowed_suppliers, key=lambda p: self.get_sigma(p), reverse=True)[:tau_buy]
        if len(allowed_consumers) > tau_sell:
            allowed_consumers = sorted(allowed_consumers, key=lambda p: self.get_sigma(p), reverse=True)[:tau_sell]

        allowed_set = set(allowed_suppliers + allowed_consumers)

        unneeded_response = (
            SAOResponse(ResponseType.END_NEGOTIATION, None)
            if not self.awi.allow_zero_quantity
            else SAOResponse(ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0))
        )

        response = {}
        for p in offers.keys():
            if p not in allowed_set:
                response[p] = unneeded_response

        filtered_offers = {k: v for k, v in offers.items() if k in allowed_set}
        today_offers = {k: v for k, v in filtered_offers.items() if v[TIME] == self.awi.current_step}

        target_inventory, max_inventory = self.inventory_policy()

        current_total_input = self.awi.current_inventory_input + self.projected_incoming()

        # 二重計上を避けるため committed_incoming は使わない
        committed_incoming = 0

        buy_quota = max(0, target_inventory - current_total_input)

        # panic buy は「今日在庫」だけでなく「短期見込み」も含めて決める
        short_term_available = self.awi.current_inventory_input + self.awi.total_supplies_at(self.awi.current_step)
        is_panic_buy = short_term_available < max(1, int(0.8 * self.awi.n_lines))
        # ループ内で受諾した数量を即時反映するためのローカル変数（アトミックガード用）
        committed_in_loop = 0

        # BUYER 向けのサプライヤー組合せ選定処理をローカル関数化
        def _handle_buyer_case():
            """BUYER モード時の複合オファー評価ロジック（内部関数）。

            buy_quota を逐次減らして、ループ内での重複受諾を防ぐ。
            """
            # 変更: committed_in_loop を非同期受諾のガードに使用する
            nonlocal response, buy_quota, committed_in_loop

            valid_suppliers = [_ for _ in self.awi.my_suppliers if _ in filtered_offers.keys()]
            today_partners = set([_ for _ in self.awi.my_suppliers if _ in today_offers.keys()])
            plist = list(powerset(today_partners))[::-1]
            price_good_plist = [ps for ps in plist if len(ps) > 0 and max([offers[p][UNIT_PRICE] for p in ps]) * self.awi.current_exogenous_output_quantity < self.awi.current_exogenous_output_price - self.awi.profile.cost * self.awi.current_exogenous_output_quantity]
            if len(price_good_plist) > 0:
                plist = price_good_plist

            plus_best_diff, plus_best_indx = float("inf"), -1
            minus_best_diff, minus_best_indx = -float("inf"), -1
            todays_input_needed = self.awi.current_exogenous_output_quantity - (min(self.awi.current_inventory_input, self.awi.n_lines) + self.awi.current_inventory_output)

            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = offered - todays_input_needed
                if diff >= 0:
                    if diff < plus_best_diff:
                        plus_best_diff, plus_best_indx = diff, i
                    elif diff == plus_best_diff:
                        if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(offers[p][UNIT_PRICE] for p in plist[plus_best_indx]):
                            plus_best_diff, plus_best_indx = diff, i
                if diff <= 0:
                    if diff > minus_best_diff:
                        minus_best_diff, minus_best_indx = diff, i
                    elif diff == minus_best_diff:
                        if diff < 0 and len(partner_ids) < len(plist[minus_best_indx]):
                            minus_best_diff, minus_best_indx = diff, i
                        elif diff == 0 or (diff < 0 and len(partner_ids) == len(plist[minus_best_indx])):
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(offers[p][UNIT_PRICE] for p in plist[minus_best_indx]):
                                minus_best_diff, minus_best_indx = diff, i

            min_rel_time = min(s.relative_time for s in states.values()) if states else 0
            mismatch_margin = int(self.awi.n_lines * (1 - min_rel_time))
            th_min, th_max = -mismatch_margin, mismatch_margin
            if th_min <= minus_best_diff or plus_best_diff <= th_max:
                best_indx = plus_best_indx if plus_best_diff <= th_max else minus_best_indx
                if best_indx != -1:
                    # 個別にクォータをチェックしつつ受諾を出す
                    for p in plist[best_indx]:
                        q = offers[p][QUANTITY]
                        # 在庫枠とのシミュレーションガードを実行
                        allowed_space = max_inventory - (self.awi.current_inventory_input + committed_in_loop)
                        if allowed_space <= 0:
                            # 物理容量が無い場合は受諾しない
                            response[p] = unneeded_response
                            continue

                        # 非パニック時は buy_quota も考慮する
                        if not is_panic_buy and buy_quota <= 0:
                            response[p] = SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, self.get_smart_price(p, is_seller=False)))
                            continue

                        # 取得可能数量を計算（容量・枠で制限）
                        if is_panic_buy:
                            takeable = min(q, allowed_space)
                        else:
                            takeable = min(q, buy_quota, allowed_space)

                        if takeable <= 0:
                            response[p] = unneeded_response
                            continue

                        if takeable < q:
                            # 枠が足りないため縮小したカウンターを出す
                            response[p] = SAOResponse(ResponseType.REJECT_OFFER, (takeable, self.awi.current_step, offers[p][UNIT_PRICE]))
                            if not is_panic_buy:
                                buy_quota = max(0, buy_quota - takeable)
                            committed_in_loop += takeable
                            continue

                        # 十分な枠があるため完全受諾
                        response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])
                        if not is_panic_buy:
                            buy_quota -= q
                        committed_in_loop += q
                    remained_suppliers = set(valid_suppliers).difference(plist[best_indx])
                    for k in remained_suppliers:
                        response[k] = SAOResponse(ResponseType.REJECT_OFFER, (offers[k][QUANTITY], self.awi.current_step, self.get_smart_price(k, is_seller=False)))
            else:
                response |= {partner_id: unneeded_response for partner_id in filtered_offers.keys() if partner_id in self.awi.my_suppliers}

        # 通常の買い（supplier からのオファー処理）ロジックをローカル関数化
        def _handle_normal_buying():
            """通常時の仕入れオファー評価（価格比較に基づく受諾/再提案）。"""
            nonlocal response, buy_quota, committed_in_loop
            buying_offers = {partner_id: offer for partner_id, offer in filtered_offers.items() if partner_id in self.awi.my_suppliers}
            if sum(self.awi.current_inventory) > self.max_inventory:
                response |= {k: SAOResponse(ResponseType.END_NEGOTIATION, None) for k in buying_offers.keys()}
                return
            input_secured = 0
            for partner_id, offer in sorted(buying_offers.items(), key=lambda x: x[1][UNIT_PRICE]):
                smart_buy_p = self.get_smart_price(partner_id, is_seller=False)
                q = offer[QUANTITY]
                if offer[UNIT_PRICE] <= smart_buy_p:
                    # 在庫容量のシミュレーションガード
                    allowed_space = max_inventory - (self.awi.current_inventory_input + committed_in_loop)
                    if allowed_space <= 0:
                        response[partner_id] = unneeded_response
                        continue

                    # 価格が許容内で、枠がある場合の処理
                    if not is_panic_buy and buy_quota <= 0:
                        response[partner_id] = SAOResponse(ResponseType.REJECT_OFFER, (q, self.awi.current_step, smart_buy_p))
                    else:
                        if not is_panic_buy and buy_quota < q:
                            # 枠が足りないので縮小を提案（ただし容量も考慮）
                            takeable = min(buy_quota, allowed_space)
                            if takeable <= 0:
                                response[partner_id] = unneeded_response
                            else:
                                response[partner_id] = SAOResponse(ResponseType.REJECT_OFFER, (takeable, self.awi.current_step, offer[UNIT_PRICE]))
                                buy_quota = max(0, buy_quota - takeable)
                                committed_in_loop += takeable
                        else:
                            # 十分な枠がある場合の受諾判定（納期が今日のものは即受諾）
                            takeable = min(q, allowed_space) if is_panic_buy else min(q, buy_quota, allowed_space)
                            if takeable < q:
                                response[partner_id] = SAOResponse(ResponseType.REJECT_OFFER, (takeable, self.awi.current_step, offer[UNIT_PRICE]))
                                if not is_panic_buy:
                                    buy_quota = max(0, buy_quota - takeable)
                                committed_in_loop += takeable
                            else:
                                if offer[TIME] == self.awi.current_step:
                                    response[partner_id] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                                else:
                                    response[partner_id] = SAOResponse(ResponseType.REJECT_OFFER, (offer[QUANTITY], self.awi.current_step, offer[UNIT_PRICE]))
                                if not is_panic_buy:
                                    buy_quota -= q
                                committed_in_loop += q
                else:
                    response[partner_id] = SAOResponse(ResponseType.REJECT_OFFER, (offer[QUANTITY], self.awi.current_step, smart_buy_p)) if offer[QUANTITY] > 0 else unneeded_response
                input_secured += q
                if sum(self.awi.current_inventory) + input_secured > self.max_inventory or (not is_panic_buy and buy_quota <= 0):
                    break

        # route buying logic: BUYER モード or 最終日 or 通常買い を振り分け
        if self.awi.my_consumers == ["BUYER"]:
            _handle_buyer_case()
        elif self.awi.current_step == self.awi.n_steps - 1:
            response |= {partner_id: unneeded_response for partner_id in filtered_offers.keys() if partner_id in self.awi.my_suppliers}
        else:
            _handle_normal_buying()

        # consumers（販売）処理: 在庫を売り切る組合せを優先して受諾/再提案を作成
        selling_offers = {
            partner_id: offer
            for partner_id, offer in filtered_offers.items()
            if partner_id in self.awi.my_consumers
        }

        # 予約を考慮した実効売り余力
        sell_cap_remaining = self.remaining_sellable_capacity_effective()

        plist = list(powerset(selling_offers.keys()))[::-1]

        # cap を超える bundle は最初から候補外
        plist = [
            ps for ps in plist
            if sum(offers[p][QUANTITY] for p in ps) <= sell_cap_remaining
        ]

        best_diff, best_indx = float("inf"), -1

        secured_output = sum(
            contract.agreement["quantity"]
            for contract in self.future_selling_contracts
            if contract.agreement.get("time", self.awi.current_step) >= self.awi.current_step
        )

        todays_output_needed = max(self.awi.needed_sales - secured_output, 0)
        todays_output_needed = min(todays_output_needed, self.awi.n_lines, sell_cap_remaining)

        for i, partner_ids in enumerate(plist):
            offered = sum(offers[p][QUANTITY] for p in partner_ids)
            diff = offered - todays_output_needed

            if (
                (-best_diff < diff <= 0)
                or (
                    -diff == best_diff
                    and best_indx != -1
                    and sum(offers[p][UNIT_PRICE] for p in plist[best_indx])
                    < sum(offers[p][UNIT_PRICE] for p in partner_ids)
                )
            ):
                best_diff, best_indx = -diff, i

        if best_indx != -1:
            accepted_qty = sum(offers[p][QUANTITY] for p in plist[best_indx])

            # ---- reservation をここで追加 ----
            self._reserved_sell_qty += accepted_qty

            """
            print(
                f"[SELL-COUNTER] step={self.awi.current_step} "
                f"accepted_qty={accepted_qty} "
                f"sell_cap_remaining={sell_cap_remaining} "
                f"reserved_sell={self._reserved_sell_qty} "
                f"future_selling={sum(c.agreement.get('quantity',0) for c in self.future_selling_contracts)} "
                f"supply_est={self.awi.current_inventory_input + self.projected_incoming()} "
                f"remaining_prod={self.remaining_production_capacity()}"
            )
            """

            response |= {
                p: SAOResponse(ResponseType.ACCEPT_OFFER, offers[p])
                for p in plist[best_indx]
            }

            remained_consumers = list(set(selling_offers).difference(plist[best_indx]))

        else:
            remained_consumers = list(selling_offers.keys())
            best_diff = todays_output_needed

        remained_output_needs = min(best_diff, self.remaining_sellable_capacity_effective())

        for future_t in range(self.awi.current_step, self.awi.n_steps):

            if len(remained_consumers) == 0:
                break

            if remained_output_needs == 0:
                response |= {k: unneeded_response for k in remained_consumers}
                break

            tmp_output_needs = (
                min(remained_output_needs, self.awi.n_lines - (todays_output_needed - best_diff))
                if future_t == self.awi.current_step
                else min(remained_output_needs, self.awi.n_lines)
            )

            if tmp_output_needs == 0:
                continue

            distribution = self.distribute_even(tmp_output_needs, remained_consumers)

            offered_now = 0
            for k, q in distribution.items():
                if q > 0:
                    response[k] = SAOResponse(
                        ResponseType.REJECT_OFFER,
                        (q, future_t, self.get_smart_price(k, is_seller=True))
                    )
                    offered_now += q
                else:
                    response[k] = unneeded_response

            # ここでは reservation しない
            # （まだ ACCEPT していない future counter proposal だから）
            remained_output_needs -= offered_now
            remained_consumers = list(
                set(remained_consumers).difference({k for k, q in distribution.items() if q > 0})
            )

        return response

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        if offer and offer[TIME] - self.awi.current_step > 7:
            return ResponseType.REJECT_OFFER

        active_sups = self.select_partners(self.awi.my_suppliers)
        active_cons = self.select_partners(self.awi.my_consumers)
        if negotiator_id not in active_sups and negotiator_id not in active_cons:
            return ResponseType.REJECT_OFFER

        # ---- BUY side ----
        if self.is_supplier(negotiator_id):
            target_inventory, max_inventory = self.inventory_policy()
            secured = self.awi.current_inventory_input + self.projected_incoming()

            if secured >= max_inventory:
                return ResponseType.REJECT_OFFER

            # emergency buy: 近い納期・適正価格なら積極受諾
            if self.need_emergency_buy():
                if offer is None:
                    return ResponseType.REJECT_OFFER

                if (
                    offer[TIME] <= self.awi.current_step + 1
                    and offer[QUANTITY] <= max(self.awi.n_lines, self.emergency_buy_target())
                    and self.is_good_price(negotiator_id, offer)
                ):
                    return ResponseType.ACCEPT_OFFER

            return (
                ResponseType.ACCEPT_OFFER
                if self.is_needed(negotiator_id, offer) and self.is_good_price(negotiator_id, offer)
                else ResponseType.REJECT_OFFER
            )

        # ---- SELL side ----
        if self.is_consumer(negotiator_id):
            if offer is None:
                return ResponseType.REJECT_OFFER

            q = offer[QUANTITY]

            # いまこの step で本当に追加で売ってよい量
            cap = self.remaining_sellable_capacity_effective()

            if (
                q <= cap
                and self.is_needed(negotiator_id, offer)
                and self.is_good_price(negotiator_id, offer)
            ):
                # ACCEPT する瞬間だけ reservation
                self._reserved_sell_qty += q
                return ResponseType.ACCEPT_OFFER

            return ResponseType.REJECT_OFFER

        return ResponseType.REJECT_OFFER

    def is_good_price(self, partner, offer):
        if offer is None: return False
        smart_p = self.get_smart_price(partner, is_seller=self.is_consumer(partner))
        if self.is_consumer(partner):
            return offer[UNIT_PRICE] >= smart_p
        return offer[UNIT_PRICE] <= smart_p

    def propose(self, negotiator_id: str, state):
        if self.is_supplier(negotiator_id):

            # emergency buy 中は target 判定を無視して提案を出す
            if self.need_emergency_buy():
                return self.good_offer(negotiator_id, state)

            target_inventory, max_inventory = self.inventory_policy()
            secured = self.awi.current_inventory_input + self.projected_incoming()

            if secured >= int(1.2 * target_inventory):
                return None

        return self.good_offer(negotiator_id, state)
    
    def good_offer(self, partner, state):
        nmi = self.get_nmi(partner)
        if not nmi: return None
        issues = nmi.issues
        qissue = issues[QUANTITY]
        t_list = sorted(list(issues[TIME].all))
        for t in t_list:
            if abs(t - self.awi.current_step) > 7: continue
            needed = self._needs(partner, t)
            if needed <= 0: continue
            
            quantity = max(min(needed, qissue.max_value), qissue.min_value)
            price = self.get_smart_price(partner, is_seller=self.is_consumer(partner))
            return (quantity, t, price)
        return None
    
    def _needs(self, partner, t):
        if self.awi.is_first_level:
            total_needs = self.awi.needed_sales
        elif self.awi.is_last_level:
            total_needs = self.awi.needed_supplies
        else:
            total_needs = self.production_level * self.awi.n_lines

        if self.is_consumer(partner):
            future_steps = max(0, t - self.awi.current_step)
            total_needs += self.production_level * self.awi.n_lines * future_steps
            total_needs -= self.awi.total_sales_until(t)

            # 今日だけ安全販売量で制限
            if t == self.awi.current_step:
                total_needs = min(total_needs, self.max_safe_sales_today())

            total_needs = min(total_needs, self.remaining_sellable_capacity_effective())
        else:
            target_inventory, max_inventory = self.inventory_policy()

            secured_inventory = (
                self.awi.current_inventory_input
                + self.projected_incoming()
            )

            future_steps = max(0, t - self.awi.current_step)

            desired_inventory = (
                target_inventory
                + int(
                    0.5 * self.awi.n_lines * future_steps / max(1, self.awi.n_steps)
                )
            )

            total_needs = max(0, desired_inventory - secured_inventory)

            # emergency buy: 今日〜次ステップは最低 need を保証
            if self.need_emergency_buy() and t <= self.awi.current_step + 1:
                total_needs = max(total_needs, self.emergency_buy_target())
        return int(total_needs)
    
    def is_consumer(self, partner):
        return partner in self.awi.my_consumers
    
    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers

    def step(self):
        pass

    def on_negotiation_failure(self, partners: list[str], annotation: dict[str, Any], mechanism: StdAWI, state: SAOState) -> None:
        partner = [p for p in partners if p != self.id][0]
        self.evaluator.record_failure(partner)

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:
        role = "SELLER" if contract.annotation.get("seller") == self.id else "BUYER"
        partner = [p for p in contract.partners if p != self.id][0]
        
        incoming_quantity = contract.agreement.get("quantity", 0)
        self.evaluator.record_success(partner, incoming_quantity)

        # 変更: 成立した契約を将来契約リストに登録して追跡する（修正要求1）
        if role == "BUYER":
            # 買い契約: 将来入荷のトラッキングに追加
            try:
                self.future_buying_contracts.append(contract)
            except Exception:
                self.future_buying_contracts = [contract]
            self.cumulative_input += incoming_quantity

            self._reserved_buy_qty = max(
                0,
                getattr(self, "_reserved_buy_qty", 0) - incoming_quantity
            )
        else:
            # 売り契約: 将来販売契約リストに追加
            try:
                self.future_selling_contracts.append(contract)
            except Exception:
                self.future_selling_contracts = [contract]

            self._reserved_sell_qty = max(
                0,
                getattr(self, "_reserved_sell_qty", 0) - incoming_quantity
            )

        if role == "SELLER":
            """
            print(
                f"[SELL-SUCCESS] step={self.awi.current_step} "
                f"time={contract.agreement.get('time')} "
                f"q={contract.agreement.get('quantity',0)} "
                f"future_selling={sum(c.agreement.get('quantity',0) for c in self.future_selling_contracts)} "
                f"supply_est={self.awi.current_inventory_input + self.projected_incoming()}"
            )
            """

if __name__ == "__main__":
    import sys
    from .helpers.runner import run
    run([TakaLinkAgent], sys.argv[1] if len(sys.argv) > 1 else "std")