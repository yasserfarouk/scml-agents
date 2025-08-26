from scml.std import StdAWI
from typing import Any
from negmas import SAOResponse, ResponseType
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE

from .record_manager import RecordManager
from .agent_utils import AgentUtils as au
from .utility_evaluator import UtilityEvaluator
from .partner_scorer import PartnerScorer
from .supply_demand_allocator import SupplyDemandAllocator
from .offer_generator import OfferGenerator
from .utility_evaluator import UtilityEvaluator

class ProposalStrategy:
    """
    提案戦略を管理するクラス．
    交渉相手ごとの交渉変数（issues）と過去の履歴を元に
    適切な提案内容（数量・納期・価格）を計算する．
    """

    def __init__(self, awi: StdAWI, partners, record: RecordManager, productivity, ptoday):
        """提案戦略の初期化"""
        self.awi = awi
        self.partners = partners
        self.record = record
        self.productivity = productivity
        self.ptoday = ptoday
        self.round_number = 0
        self.nmis = {}

        self.scorer = PartnerScorer(self.record)
        self.sda = SupplyDemandAllocator(self.awi, self.scorer)
        

    def before_step(self, active_negotiators):
        # special rules
        # シミュレーション初期は分配量が多いので，交渉人数を増やす
        if self.awi.level == 0:
            self.ptoday = 0.6
        elif self.awi.current_step < 2:
            self.ptoday = 1.0
        else:
            self.ptoday = 0.6
        
        if self.awi.current_step == 1 and self.awi.level > 0:
            self.productivity = 0.4
        else:
            self.productivity = 0.7
        
        self.todays_distribution = self.sda.distribute_todays_needs(
            self.productivity,
            self.ptoday,
            active_negotiators
        )
        self.ue = UtilityEvaluator(self.awi, self.todays_distribution, self.record)
        self.scorer.set_todays_utility_evaluator(self.ue)
        self.offer_generator = OfferGenerator(self.awi, self.todays_distribution, self.ue)
    
    def step(self):
        self.scorer.update_all_scores()
        self.round_number = 0

    def set_nmis(self, nmis):
        self.nmis = nmis
        self.ue.set_nmis(nmis)
        self.offer_generator.set_nmis(nmis)

    def should_accept_offer(self, partner_id: str, offer, threshold: float) -> bool:
        is_producible = True
        # special rule: step < 2 & 自分が購入する & (lv.1 or lv.2) -> 原材料を確保したいのでis_producible = True
        if self.awi.current_step < 2 and au.is_seller(self.awi, partner_id) and self.awi.level > 0:
            # 原材料は確保したいが，あまりにも提案量が多いofferや納期が遅いofferはREJECT
            if offer[QUANTITY] < self.awi.n_lines * 0.7:
                return (self.ue.time_utility(partner_id, offer[TIME]) + self.ue.price_utility(partner_id, offer[UNIT_PRICE])) / 2 > threshold
            else:
                is_producible = False
        # special rule: step == 0 & 自分が販売する & 納期が0 -> 在庫が安定しない可能性大のため，is_producible = False
        elif self.awi.current_step == 0 and offer[TIME] == 0:
            if au.is_buyer(self.awi, partner_id):
                # is_producible = False
                return False
        # special rule: 一人のエージェントからの提案があまりにも大きい場合，is_producible = False
        elif offer[QUANTITY] > self.awi.n_lines * self.productivity * 0.7:
            is_producible = False
        # 自分が販売する & 納期が今日 -> 本当に作れるか確認して無理そうならis_producible = False
        # elif au.is_buyer(self.awi, partner_id) and offer[TIME] == self.awi.current_step:
        if au.is_buyer(self.awi, partner_id):
            contracted = self.record.get_contracted_sales_at_step(offer[TIME])
            if contracted + offer[QUANTITY] > self.awi.n_lines * self.productivity:
                is_producible = False
        return (self.ue.ufun(partner_id, offer) >= threshold) and is_producible
    
    def should_end_negotiation_offer(self, partner_id):
        # step == 0 & lv.0 & 自分が販売する -> 原材料を確保できていないケースが多いので売らない
        # if self.awi.current_step == 0 and self.awi.level == 0 and au.is_buyer(self.awi, partner_id):
        #     return True
        if self.sda.has_supplier_distribution(self.todays_distribution) and au.is_seller(self.awi, partner_id):
            return True
        elif self.record.get_contracted_supplies_current_step() > self.awi.n_lines * 1.4:
            return True
        return False
    
    def best_offer_combination(self, offers, max_quantity):
        best_combo = []
        max_total = 0

        for combo in au.powerset(offers):
            total = sum(offer[QUANTITY] for _, offer in combo)
            if total <= max_quantity and total > max_total:
                best_combo = combo
                max_total = total
        
        return best_combo
    
    def accept_if_few_today_offers(self, responses: dict[str, SAOResponse], offers: dict[str, tuple[int, int, int]]) -> dict[str, SAOResponse]:
        """
        counter_offerが5つ以下の場合，
        今日納期かつcapacityを超えない範囲のオファーのみACCEPTし，それ以外はREJECTする
        """

        today = self.awi.current_step
        max_quantity = self.awi.n_lines
        current_quantity = self.record.get_contracted_sales_current_step()

        today_offers = [
            (pid, offer) for pid, offer in offers.items()
            if offer[TIME] == today and au.is_buyer(self.awi, pid)
        ]

        if len(today_offers) > 6 :
            return responses  # 条件を満たさない場合は何もしない

        # 既にACCEPTしているものの数量を加味する
        for resp in responses.values():
            if resp.response == ResponseType.ACCEPT_OFFER and resp.outcome[TIME] == today:
                current_quantity += resp.outcome[QUANTITY]

        best_combo = self.best_offer_combination(today_offers, max_quantity - current_quantity)

        for partner_id, offer in today_offers:
            if (partner_id, offer) in best_combo:
                responses[partner_id] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                current_quantity += offer[QUANTITY]
            else:
                responses[partner_id] = SAOResponse(ResponseType.REJECT_OFFER)

        return responses
    
    def final_round_accept_today_offers(self, responses: dict[str, SAOResponse], offers: dict[str, tuple[int, int, int]]) -> dict[str, SAOResponse]:
        """
        最終ラウンドで，納期が今日のオファーの中から
        QUANTITYの小さいものから順に，生産可能量に収まる範囲でACCEPTする
        """
        today = self.awi.current_step
        max_quantity = self.awi.n_lines * self.productivity
        current_quantity = self.record.get_contracted_sales_current_step()

        if self.awi.level == 0:
            max_quantity *= 2.0

        # すでにresponsesにACCEPTされているofferを加算
        for resp in responses.values():
            if resp.response == ResponseType.ACCEPT_OFFER and resp.outcome[TIME] == today:
                current_quantity += resp.outcome[QUANTITY]

        # 今日納期かつ販売用のオファーを抽出
        today_offers = []
        for partner_id, offer in offers.items():
            if (
                au.is_buyer(self.awi, partner_id) and
                offer[TIME] == today and
                self.ue.ufun(partner_id, offer) > 0.1  # 念のため効用0以下は避ける
            ):
                today_offers.append((partner_id, offer))

        # QUANTITY昇順でソート
        today_offers.sort(key=lambda x: x[1][QUANTITY])

        for partner_id, offer in today_offers:
            # すでにACCEPTする予定のofferは変更しない
            if partner_id in responses and responses[partner_id].response == ResponseType.ACCEPT_OFFER:
                continue

            if current_quantity >= max_quantity:
                break  # 最大量に到達した場合，打ち切り

            q = offer[QUANTITY]
            if current_quantity + q <= max_quantity:
                responses[partner_id] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                current_quantity += q
            else:
                responses[partner_id] = SAOResponse(ResponseType.REJECT_OFFER)

        return responses

    def generate_initial_offer(self, partners):
        """最初の提案内容を生成する"""
        responses = {}
        for partner_id in partners:
            # partner_idのエージェントに割り当てする量を取得
            if partner_id in self.todays_distribution:
                pid_distribution = self.todays_distribution[partner_id]
            else:
                pid_distribution = 0
            
            if self.should_end_negotiation_offer(partner_id):
                # END_NEGOTIATIONの対象になるようなエージェントにはofferを送らない
                continue
            
            # 自分にとって有利なofferの内容を作成
            quantity = self.offer_generator.get_best_quantity(partner_id, pid_distribution)
            time = self.offer_generator.get_best_time(partner_id)
            price = self.offer_generator.get_best_price(partner_id)
            responses[partner_id] = (quantity, time, price)

        return responses
    
    def generate_counter_offer(self, offers, threshold: float):
        """カウンターオファーの提案内容を生成する"""
        self.round_number += 1
        step_discount = 1.0
        responses = {}
        for partner_id, offer in offers.items():
            # special rules: step < 2 & 自分が購入する & (lv.1 or lv.2) -> 閾値を下げる
            if self.awi.current_step < 2 and au.is_seller(self.awi, partner_id) and self.awi.level > 0:
                step_discount = 0.75
            else:
                step_discount = 1.0

            round_discount = 0.95 ** self.round_number

            # offerを記録
            self.record.set_offer_history(partner_id, self.awi.current_step, offer)
            if self.should_end_negotiation_offer(partner_id):
                responses[partner_id] = SAOResponse(
                    ResponseType.END_NEGOTIATION
                )
                continue
            elif self.should_accept_offer(partner_id, offer, threshold * step_discount * round_discount) == True:
                responses[partner_id] = SAOResponse(
                    ResponseType.ACCEPT_OFFER, offer
                )
                continue
            base_offer = self.offer_generator.generate_base_offer(partner_id, threshold * step_discount * round_discount)
            concession_offer = self.offer_generator.find_least_concession_offer(partner_id, base_offer)
            responses[partner_id] = SAOResponse(
                ResponseType.REJECT_OFFER, concession_offer
            )
        
        if self.round_number == 20 and self.awi.current_step > 1:
            self.final_round_accept_today_offers(responses, offers)
        elif self.awi.current_step > 1:
            self.accept_if_few_today_offers(responses, offers)

        return responses