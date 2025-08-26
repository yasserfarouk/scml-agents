from dataclasses import dataclass

@dataclass
class OfferEntry:
    """
    過去の提案のプロパティを保持するデータクラス
    """
    sim_step: int # 提案時のステップ
    offer: tuple # 提案内容（数量, 納期, 単価）

    def __repr__(self):
        return f"(sim_step={self.sim_step}, offer={self.offer})"
    
class OfferHistory:
    """
    提案履歴のエントリOfferEntryのリストを保持するクラス
    """
    def __init__(self):
        self.offers: list[OfferEntry] = []
    
    def __repr__(self):
        return f"OfferHistory(n={len(self.offers)}, last={self.offers[-1] if self.offers else 'None'})"
    
    def add(self, sim_step: int, offer: tuple):
        entry = OfferEntry(
            sim_step=sim_step,
            offer=offer
        )
        self.offers.append(entry)