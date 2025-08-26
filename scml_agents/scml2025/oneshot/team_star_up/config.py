from dataclasses import dataclass

@dataclass
class AgentConfig_Balanced:
    # ==============================
    # PARAMETERS
    # pは必要量にかけるパラメータ
    # 現段階では1/2のneedsを最善価格で受け入れるようにしている
    ACCEPT_COEFFICIENT: float = 1
    # 廃棄ペナルティの許容割合（needs*penalty_accept）
    PENALTY_ACCEPT: float = 0.05
    # 多めにofferをかけるパラメータ
    # 自分の側（売り手・買い手）が少ない時、強気に行くために1に近づける
    # n_competetiors + 1 > ...って感じで場合わけ
    # 自分の側が多い時は、契約をできる限り勝ち取るために1より大きくする
    # OFFERの数の係数
    OFFERING_FACTOR: float = 1.2
    # OFFERの金額の係数
    PRICE_FACTOR: float = 0.9

    INIT_BUY_PRICE_RATIO: float = 0.2
    INIT_SALE_PRICE_RATIO: float = 0.8

    # 交渉相手エージェントのスコアを算出する関数に使うパラメータ(売り手、買い手によって変わってくる)
    # score = PRICE_SCORE * (price_mean - price_min) + ROUNED_SCORE * (20 - round_mean) 
    #                                   + ACCEPT_SCORE * (abs(accepted_offer_quantity -first_offer_quantity))
    PRICE_SCORE: float = 1
    ROUND_SCORE: float = 0.1
    ACCEPT_SCORE: float = 0.5
    # 

    # ==============================

@dataclass
class AgentConfig_Strong:
    #競合が少ない時

    # ==============================
    # PARAMETERS
    # pは必要量にかけるパラメータ
    # 現段階では1/2のneedsを最善価格で受け入れるようにしている
    ACCEPT_COEFFICIENT: float = 0.5
    # 廃棄ペナルティの許容割合（needs*penalty_accept）
    PENALTY_ACCEPT: float = 0.05
    # 多めにofferをかけるパラメータ
    OFFERING_FACTOR: float = 1.2
    PRICE_FACTOR: float = 0.9

    INIT_BUY_PRICE_RATIO: float = 0.2
    INIT_SALE_PRICE_RATIO: float = 0.8

    PRICE_SCORE: float = 1
    ROUND_SCORE: float = 0.1
    ACCEPT_SCORE: float = 0.5
    # ==============================

@dataclass
class AgentConfig_Weak:
    #競合が多い時
    # ==============================
    # PARAMETERS
    # pは必要量にかけるパラメータ
    # 現段階では1/2のneedsを最善価格で受け入れるようにしている
    ACCEPT_COEFFICIENT: float = 0.5
    # 廃棄ペナルティの許容割合（needs*penalty_accept）
    PENALTY_ACCEPT: float = 0.05
    # 多めにofferをかけるパラメータ
    OFFERING_FACTOR: float = 1.2
    PRICE_FACTOR: float = 0.9

    INIT_BUY_PRICE_RATIO: float = 0.2
    INIT_SALE_PRICE_RATIO: float = 0.8

    PRICE_SCORE: float = 1
    ROUND_SCORE: float = 0.1
    ACCEPT_SCORE: float = 0.5
    # ==============================