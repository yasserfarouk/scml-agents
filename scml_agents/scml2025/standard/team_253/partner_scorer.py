from .record_manager import RecordManager
from .agreements_manager import AgreementsManager
from .agent_utils import AgentUtils as au

class PartnerScorer:
    def __init__(self, record: RecordManager):
        self.record = record
        self.scores: dict[str, list[float]] = {}
        self.weighted_scores: dict[str, float] = {}
    
    def set_todays_utility_evaluator(self, ufun):
        self.ue = ufun

    def update_all_scores(self) -> None:
        """
        すべてのパートナーのスコアを更新するメソッド
        """
        for partner_id, data in self.record._data.items():
            # 交渉相手のIDが存在しない場合は新たに追加する
            if partner_id not in self.scores:
                self.scores[partner_id] = []

            # 成功率スコア
            successes = self.record.get_success_count(partner_id)
            failures = self.record.get_failure_count(partner_id)
            reliability_score = successes / (successes + failures) if (successes + failures) > 0 else 0.0

            # 合意情報から効用スコアを計算
            agreements: AgreementsManager = data["agreements"]
            avg_quantity = agreements.get_weighted_avg_agreements_quantity()
            avg_price = agreements.get_weighted_avg_agreements_price()
            avg_lead_time = agreements.get_weighted_avg_agreements_lead_time()

            u_score = self.ue.ufun(partner_id, (avg_quantity, avg_lead_time, avg_price))

            score = (reliability_score + u_score) / 2
            # 本ステップのスコアを記録する
            self.scores[partner_id].append(score)

            # スコアの履歴から重み付き平均を計算し，weighted_scoresを上書きする
            self.weighted_scores[partner_id] = self.get_weighted_score(partner_id)
    
    def get_weighted_score(self, partner_id: str) -> float:
        """
        指定したパートナーの過去のスコアから重み付き平均を計算し，返すメソッド
        """
        if partner_id not in self.scores:
            return 0.0
        
        return au.weighted_average(self.scores[partner_id])

    def sorted_partners(self, reverse: bool = True) -> list[str]:
        """
        今までのスコアの重み月平均スコアに基づいてパートナーIDを並べ替えたリストを返すメソッド
        デフォルトではこのスコアが高い順に並べる．
        """
        return sorted(
            self.weighted_scores.keys(),
            key=lambda pid: self.weighted_scores.get(pid, 0.0),
            reverse=reverse,
        )