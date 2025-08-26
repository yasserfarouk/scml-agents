from typing import List
from dataclasses import dataclass

@dataclass
class AgreementEntry:
    """
    合意契約のプロパティを保持するデータクラス
    """
    sim_step: int  # 合意時のステップ
    quantity: int  # 合意時の取引量
    time: int      # 合意時の納期
    price: int     # 合意時の単価
    lead_time: int # 発注から納品までの時間差
    opponent_is_buyer: bool # 交渉相手がbuyerか

    def __repr__(self):
        return f"(sim_step={self.sim_step}, quantity={self.quantity}, time={self.time}, price={self.price}, lead_time={self.lead_time}"

from .agent_utils import AgentUtils as au

class AgreementsManager:
    """
    合意契約のエントリAgreementEntryのリストを保持するクラス
    """
    def __init__(self):
        self.agreements: List[AgreementEntry] = []
    
    def __repr__(self):
        return f"AgreementsManager(n={len(self.agreements)}, last={self.agreements[-1] if self.agreements else 'None'})"

    def add(self, contract, is_buyer):
        entry = AgreementEntry(
            sim_step=contract.annotation["sim_step"],
            quantity=contract.agreement["quantity"],
            time=contract.agreement["time"],
            price=contract.agreement["unit_price"],
            lead_time=contract.agreement["time"] - contract.annotation["sim_step"],
            opponent_is_buyer=is_buyer
        )
        self.agreements.append(entry)
    

    def get_weighted_avg_agreements_quantity(self) -> float:
        """
        合意契約の量の重み付き平均を計算するメソッド
        """
        if not self.agreements:
            return 0.0
        return au.weighted_average([entry.quantity for entry in self.agreements])

    def get_weighted_avg_agreements_price(self) -> float:
        """
        合意契約の単価の重み付き平均を計算するメソッド
        """
        if not self.agreements:
            return 0.0
        return au.weighted_average([entry.price for entry in self.agreements])
    
    def get_weighted_avg_agreements_lead_time(self) -> float:
        """
        合意契約の発注から納品までの時間差を計算するメソッド
        """
        if not self.agreements:
            return 0.0
        return au.weighted_average([entry.lead_time for entry in self.agreements])