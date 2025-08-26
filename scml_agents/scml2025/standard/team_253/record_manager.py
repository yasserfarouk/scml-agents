from typing import Dict

from .agent_utils import AgentUtils as au
from .agreements_manager import AgreementsManager
from .offer_history import OfferHistory

class RecordManager:
    """
    過去の交渉における契約情報を保持するクラス
    """
    def __init__(self, awi):
        self.awi = awi
        self._data: Dict[str, dict] = {}
    
    def ensure_partner(self, partner_id: str):
        """任意のパートナーが_dataに存在することを確認するメソッド. ない場合は新たに追加する"""
        if partner_id not in self._data:
            self._data[partner_id] = {
                "success": 0,
                "failure": 0,
                "agreements": AgreementsManager(),
                "offer_history": OfferHistory()
            }

    def set_agreements(self, partner_id, contract, opponent_is_buyer):
        """交渉結果を_dataに追加するメソッド"""
        self.ensure_partner(partner_id)
        self._data[partner_id]["agreements"].add(contract, opponent_is_buyer)
    
    def set_offer_history(self, parnter_id, sim_step: int, offer: tuple):
        """交渉履歴を_dataに追加するメソッド"""
        self.ensure_partner(parnter_id)
        self._data[parnter_id]["offer_history"].add(sim_step, offer)
    
    def set_success(self, my_id, contract):
        """交渉成功時に成立した契約内容を_dataに追加するメソッド"""
        # 交渉相手のIDを取得し，_dataに存在しない場合は新たに追加する
        partner_id = self.get_partner_id(my_id, contract.partners)
        self.ensure_partner(partner_id)
        # _dataに成功情報を追加する
        self.set_agreements(partner_id, contract, au.is_buyer(self.awi, partner_id))

    def set_failure(self, my_id, partners: list[str]):
        """交渉失敗時に_dataに失敗情報を追加するメソッド"""
        # 交渉相手のIDを取得し，_dataに存在しない場合は新たに追加する
        partner_id = self.get_partner_id(my_id, partners)
        self.ensure_partner(partner_id)
        # _dataに失敗情報を追加する
        self._data[partner_id]["failure"] += 1
    
    def get_partner_id(self, my_id, partners: list[str]) -> str:
        """交渉相手のIDを取得するメソッド"""
        return [p for p in partners if p != my_id][0]
    
    def get_success_count(self, partner_id: str) -> int:
        """指定したパートナーの成功回数を取得するメソッド"""
        self.ensure_partner(partner_id)
        return self._data[partner_id]["success"]
    
    def get_failure_count(self, partner_id: str) -> int:
        """指定したパートナーの失敗回数を取得するメソッド"""
        self.ensure_partner(partner_id)
        return self._data[partner_id]["failure"]
    
    def get_negotiation_count(self, partner_id: str) -> int:
        """指定したパートナーとの交渉回数を取得するメソッド"""
        self.ensure_partner(partner_id)
        return self._data[partner_id]["success"] + self._data[partner_id]["failure"]
    
    def get_contracted_sales_at_step(self, time: int) -> int:
        """
        指定されたステップで納品する必要がある，buyerとの契約量の総和を返す
        """
        total_quantity = 0
        for record in self._data.values():
            for agreement in record["agreements"].agreements:
                if agreement.time == time and agreement.opponent_is_buyer:
                    total_quantity += agreement.quantity
        return total_quantity
    
    def get_contracted_sales_current_step(self) -> int:
        return self.get_contracted_sales_at_step(self.awi.current_step)
    
    def get_contracted_supplies_at_step(self, sim_step: int) -> int:
        """
        指定されたステップで納品を受ける予定の契約（相手がseller）量の総和を返す
        """
        total_quantity = 0
        for record in self._data.values():
            for agreement in record["agreements"].agreements:
                if agreement.sim_step == sim_step and not agreement.opponent_is_buyer:
                    total_quantity += agreement.quantity
        return total_quantity

    def get_contracted_supplies_current_step(self) -> int:
        """
        現在のステップにおける仕入れ契約（相手がseller）の量の合計を返す
        """
        return self.get_contracted_supplies_at_step(self.awi.current_step)