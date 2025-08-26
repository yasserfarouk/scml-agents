# exports the name of the training algorithm
from pathlib import Path

import numpy as np
from gymnasium import spaces
from negmas.outcomes import Outcome
from scml.oneshot.rl.observation import FlexibleObservationManager
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.context import GeneralContext#, SupplierContext, ConsumerContext
from .mycontext import MySupplierContext, MyConsumerContext
from stable_baselines3 import A2C, HerReplayBuffer, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from typing import Iterable
import itertools
from scml.oneshot.rl.helpers import (
    discretize_and_clip,
    # read_offers,
    clip,
    recover_offers,
)

from collections import deque
from typing import Mapping, TypeVar
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE, OneShotState
from scml.oneshot.rl.common import group_partners

TrainingAlgorithm: type[BaseAlgorithm] = PPO
"""The algorithm used for training. You can use any stable_baselines3 algorithm or develop your own"""

MODEL_PATH = Path(__file__).parent / "models" / "mymodel"
"""The path in which train.py saves the trained model and from which myagent.py loads it"""


def make_context(as_supplier: bool) -> GeneralContext:
    """Generates a context as a supplier or as a consumer"""
    if as_supplier:
        return MySupplierContext()

    return MyConsumerContext()


class MyObservationManager(FlexibleObservationManager):
    def __attrs_post_init__(self):
        p = self.context.extract_context_params(not self.reduce_space_size)
        if p.nlines:
            object.__setattr__(self, "n_suppliers", p.nsuppliers)
            object.__setattr__(self, "n_consumers", p.nconsumers)
            object.__setattr__(
                self, "max_quantity", p.nlines * self.capacity_multiplier
            )
            if not self.exogenous_multiplier:
                object.__setattr__(self, "exogenous_multiplier", p.nlines)
            object.__setattr__(self, "n_partners", p.nsuppliers + p.nconsumers)
        n = (3 * self.n_partners) * self.n_past_received_offers
        self._previous_offers = deque([0] * n, maxlen=n) if n else deque()


    """This is my observation manager implementing encoding and decoding the state used by the RL algorithm"""
    def get_dims(self) -> list[int]:
        """Get the sizes of all dimensions in the observation space. Used if not continuous."""
        return (
            list(
                itertools.chain(
                    [self.max_group_size * self.max_quantity + 1, self.n_prices, 2]
                    * self.n_partners
                )
            )
            # + list(
            #     itertools.chain(
            #         [self.max_group_size * self.max_quantity + 1, self.n_prices, 2]
            #         * self.n_partners
            #         * self.n_past_received_offers
            #     )
            # )
            + [self.max_quantity + 1] * 2  # needed sales and supplies
            + [self.n_bins] * 1  # level
            + [self.n_bins] * 1  # relative_simulation
            + [self.n_bins * 2]  # neg_relative
            + [self.n_bins] * 3  # production cost, penalties and other costs
            + [self.n_bins] * 2  # exogenous_contract quantity summary
            + []
        )

    def make_space(self) -> spaces.MultiDiscrete | spaces.Box:
        """Creates the action space"""
        dims = self.get_dims() # 観測空間のすべての次元のサイズを取得(継承元で定義されているフィールド値の次元数によって定義される)
        if self._dims is None:
            self._dims = dims
        elif self.extra_checks:
            assert all(
                a == b for a, b in zip(dims, self._dims, strict=True)
            ), f"Surprising dims while making space\n{self._dims=}\n{dims=}"
        if self.continuous:
            return spaces.Box(0.0, 1.0, shape=(len(dims),)) # 連続空間, 各次元が0から1の連続実数値範囲を持つ
        return spaces.MultiDiscrete(np.asarray(dims)) # 離散空間, 各次元の値域を指定する

    def encode(self, awi: OneShotAWI) -> np.ndarray:
        """Encodes the awi as an array"""
        # print(self.n_suppliers)
        # print(self.n_consumers)
        offers = read_offers(
            awi, # awiを引数として渡してofferを取得する
            self.n_suppliers,
            self.n_consumers,
            self.max_group_size,
            self.continuous,
        )

        current_offers = np.asarray(offers).flatten().tolist() # read_offers関数で得られた現在のofferのリストをフラットに
        # この段階でもofferはループしていない
        

        if self.extra_checks: # 整合性チェック
            assert (
                len(current_offers) == self.n_partners * 3
            ), f"{len(current_offers)=} but {self.n_partners=}"
            # assert (
            #     len(self._previous_offers)
            #     == self.n_past_received_offers * self.n_partners * 3
            # ), f"{self._previous_offers=} but {self.n_partners=}"

        extra = self.extra_obs(awi) # 追加の観測情報について取得
        v = np.asarray(
            current_offers
            # + list(self._previous_offers)
            + (
                [min(1, max(0, v[0] if isinstance(v, Iterable) else v)) for v in extra]
                if self.continuous
                else [
                    discretize_and_clip(
                        clip(v[0]) if isinstance(v, Iterable) else clip(v),
                        clip(v[1]) if isinstance(v, Iterable) else self.n_bins,
                    )
                    for v in extra
                ]
            ),
            dtype=np.float32 if self.continuous else np.int32,
        )
        if self.continuous:
            v = np.minimum(np.maximum(v, 0.0), 1.0)

        # if self._previous_offers:
        #     for _ in current_offers:
        #         self._previous_offers.append(_) # 過去のオファーリストの更新
        if self.extra_checks:
            space = self.make_space()
            assert self.continuous or isinstance(space, spaces.MultiDiscrete)
            assert not self.continuous or isinstance(space, spaces.Box)
            assert space is not None and space.shape is not None
            exp = space.shape[0]
            assert (
                len(v) == exp
            ), f"{len(v)=}, {len(extra)=}, {len(offers)=}, {exp=}, {self.n_partners=}\n{awi.current_negotiation_details=}"
            if self._dims is None:
                self._dims = self.get_dims()
            assert self.continuous or all(
                a <= b for a, b in zip(v, self._dims, strict=True)
            ), f"Surprising dims\n{v=}\n{self._dims=}"
            assert not self.continuous or all(
                [0 <= x <= 1 for x in v]
            ), f"Surprising dims (continuous)\n{v=}"
            if isinstance(space, spaces.MultiDiscrete):
                if not all(0 <= a < b for a, b in zip(v, space.nvec)):
                    print(
                        f"{v=}\n{space.nvec=}\n{space.nvec - v =}\n{ (awi.current_exogenous_input_quantity , awi.total_supplies , awi.total_sales , awi.current_exogenous_output_quantity) }"
                    )
                assert all(
                    0 <= a < b for a, b in zip(v, space.nvec)
                ), f"{offers=}\n{extra=}\n{v=}\n{space.nvec=}\n{space.nvec - v =}\n{ (awi.current_exogenous_input_quantity , awi.total_supplies , awi.total_sales , awi.current_exogenous_output_quantity) }"  # type: ignore
        # print(v)
        # input()
        return v

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi)

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, Outcome | None]:
        """
        Gets offers from an encoded awi.
        """
        return recover_offers(
            encoded,
            awi,
            self.n_suppliers,
            self.n_consumers,
            self.max_group_size,
            self.continuous,
            n_prices=self.n_prices,
        )
    
    def extra_obs(self, awi):
        """
        The observation values other than offers and previous offers.

        Returns:
            A list of tuples. Each is some observation variable as a
            real number between zero and one and a number of bins to
            use for discrediting this variable. If a single value, the
            number of bins will be self.n_bin

        """
        # adding extra components to the observation
        if awi.current_states:
            neg_relative_time = min(
                awi.current_states.values(), key=lambda x: x.relative_time
            ).relative_time
        else:
            neg_relative_time = 0  # デフォルト値
        exogenous = awi.exogenous_contract_summary
        incost = (
            awi.current_disposal_cost if awi.is_perishable else awi.current_storage_cost
        )

        return [
            (awi.needed_sales / self.max_quantity, self.max_quantity + 1),
            (awi.needed_supplies / self.max_quantity, self.max_quantity + 1),
            awi.level / (awi.n_processes - 1),
            awi.relative_time, # シミュレーション終了時を1としたときの現在の相対時間
            (neg_relative_time, 2 * self.n_bins),
            awi.profile.cost / self.max_production_cost,
            incost / (incost + awi.current_shortfall_penalty),
            (
                awi.trading_prices[awi.my_output_product]
                - awi.trading_prices[awi.my_input_product]
            )
            / awi.trading_prices[awi.my_output_product],
            exogenous[0][0]
            / (self.exogenous_multiplier * awi.production_capacities[0]),
            exogenous[-1][0]
            / (self.exogenous_multiplier * awi.production_capacities[-1]),
        ]
"""
観測情報
extra_obs
    awiから
        あと必要な販売量
        あと必要な供給量
        工場レベル
        相対時間
        製造コスト
        廃棄コスト
        不足コスト
        現在の取引価格
        外生契約の数量や価格
encode

"""

# ensure that the folder containing models is created
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)




def read_offers(
    state: OneShotAWI | OneShotState,
    n_suppliers: int,
    n_consumers: int,
    max_group_size: int,
    continuous: bool,
) -> list[tuple[int, int, int]] | list[tuple[float, float, float]]:
    return encode_given_offers(
        offers=state.current_offers,  # type: ignore
        state=state,
        n_suppliers=n_suppliers,
        n_consumers=n_consumers,
        max_group_size=max_group_size,
        continuous=continuous,
    )

def encode_given_offers(
    offers: dict[str, Outcome | None],
    state: OneShotAWI | OneShotState,
    n_suppliers: int,
    n_consumers: int,
    max_group_size: int,
    continuous: bool,
) -> list[tuple[int, int]] | list[tuple[float, float]]:
    encoder = encode_offers_no_time
    normalizer = normalize_offers_no_time
    suppliers = group_partners(state.my_suppliers, n_suppliers, max_group_size)
    consumers = group_partners(state.my_consumers, n_consumers, max_group_size)
    # print(f"suppliers {suppliers}")
    # print(f"consumers {consumers}")
    # input()

    min_iprice = state.current_input_outcome_space.issues[UNIT_PRICE].min_value
    max_iprice = state.current_input_outcome_space.issues[UNIT_PRICE].max_value
    max_iquantity = state.current_input_outcome_space.issues[QUANTITY].max_value
    ioffers = encoder(offers, suppliers, min_iprice, max_iprice, negotiation_details(state))
    if continuous:
        ioffers = normalizer(
            ioffers, min_iprice, max_iprice, 0, max_iquantity, subtract_min_price=False
        )
    min_oprice = state.current_output_outcome_space.issues[UNIT_PRICE].min_value
    max_oprice = state.current_output_outcome_space.issues[UNIT_PRICE].max_value
    max_oquantity = state.current_output_outcome_space.issues[QUANTITY].max_value
    ooffers = encoder(offers, consumers, min_oprice, max_iprice, negotiation_details(state))
    if continuous:
        ooffers = normalizer(
            ooffers, min_oprice, max_oprice, 0, max_oquantity, subtract_min_price=False
        )
    return ioffers + ooffers

def encode_offers_no_time(
    offers: Mapping[str, Outcome | None],
    partner_groups: list[list[str]],
    min_price: int,
    max_price: int,
    negotiation_details: tuple[list[str], list[str], list[tuple[int, int]], list[tuple[int, int]]],
) -> list[tuple[int, int, int]]:
    """
    Encodes offers from the given partner groups into `n_partners` tuples of quantity, unit-price, and role values.

    Args:
        offers: All received offers. Keys are sources. Sources not in the `partner_groups` will be ignored.
        partner_groups: A list of lists of partner IDs each defining a group to be considered together.
        min_price: Minimum allowed price.
        max_price: Maximum allowed price.
        negotiation_details: A tuple containing details about the negotiation (current_proposer, new_offerer_agents, offered_list, offer_list).

    Return:
        A list of quantity, unit-price, and role tuples of length `len(partner_groups)`.
        Role: 0 if the agent is the proposer, 1 if the agent is the receiver.
    """
    n_partners = len(partner_groups)
    offer_list: list[tuple[int, int, int]] = [(0, 0, 0) for _ in range(n_partners)]
    current_proposer, new_offerer_agents, offered_list, offer_list_details = negotiation_details

    for i, partners in enumerate(partner_groups):
        n_read = 0
        curr_offer = (0, 0, 0)  # (量, 単価, フラグ)
        for partner in partners:
            outcome = offers.get(partner, None)
            if outcome is None:
                continue

            # 自分が提案者か受け手かを判定
            if partner in current_proposer:
                # 自分が提案を受けた場合
                curr_offer = (
                    curr_offer[0] + outcome[QUANTITY],
                    curr_offer[1] + outcome[UNIT_PRICE] * outcome[QUANTITY],
                    1,  # フラグ 1: 提案を受けた
                )
            elif partner in new_offerer_agents:
                # 自分が提案者の場合
                curr_offer = (
                    curr_offer[0] + outcome[QUANTITY],
                    curr_offer[1] + outcome[UNIT_PRICE] * outcome[QUANTITY],
                    0,  # フラグ 0: 提案者
                )
            n_read += 1

        if n_read:
            if curr_offer[0]:
                curr_offer = (
                    curr_offer[0],
                    curr_offer[1] // curr_offer[0] - min_price,
                    curr_offer[2],
                )
            else:
                curr_offer = (0, max_price - min_price, curr_offer[2])

        offer_list[i] = curr_offer

    # print(f"encode_offers_no_time_with_role: offer_list: {offer_list}")
    # この時点では提案はループしていない
    return offer_list

def normalize_offers_no_time(
    offers: list[tuple[int, int]],
    min_price: int,
    max_price: int,
    min_quantity: int,
    max_quantity: int,
    subtract_min_price: int = False,
) -> list[tuple[float, float]]:
    """
    Normalize the offers to values between 0 and 1 for both quantity and unit price
    """
    d = max_price - min_price
    if not d:
        d = 1
    dq = max_quantity - min_quantity
    if not dq:
        dq = 1
    if not subtract_min_price:
        min_price = 0
    return [
        (float(offer[0] - min_quantity) / dq, float(offer[1] - min_price) / d)
        for offer in offers
    ]

def negotiation_details(awi: OneShotAWI):
    
    # awi.current_negotiation_detailsの中身を確認
    for key, value in awi.current_negotiation_details.items():
        current_proposer = [] # 誰からの提案か
        new_offerer_agents = [] # 誰に提案したか
        offered_list = []  # 提案された内容のリストを初期化
        offer_list = [] # 提案した内容のリストを初期化
        is_current_proposer_me = False # 自分が提案者か？
        # print(f"Key: {key}, Value: {value}")

        # "buy"または"sell"キーの中に交渉の詳細がある場合
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                # print(f"  Sub-Key: {sub_key}, Sub-Value: {sub_value}")

                # NegotiationDetailsオブジェクトの中にnmiがある場合
                if hasattr(sub_value, "nmi"):
                    nmi = sub_value.nmi
                    if hasattr(nmi, "_mechanism"):
                        mechanism = nmi._mechanism
                        # SAONMIオブジェクトのcurrent_stateを取得
                        if hasattr(mechanism, "_current_state"):
                            current_state = mechanism._current_state
                            # print(f"    Current State for {sub_key}:")                              
                            # print(f"      Running: {current_state.running}")
                            # print(f"      Step: {current_state.step}")
                            # print(f"      Current Offer: {current_state.current_offer}")
                            
                            # if current_state.current_offer is not None:
                            #     offer_list.append((current_state.current_offer[0], current_state.current_offer[2]))
                            
                            
                            # print(f"      Current Proposer: {current_state.current_proposer}")
                            
                            if current_state.current_proposer is not None:
                                if str(current_state.current_proposer) is not str(awi.agent): # offerが自分から出ない場合
                                    current_proposer.append(current_state.current_proposer) # 誰からの提案か
                                    if current_state.current_offer is not None:
                                        offered_list.append((current_state.current_offer[0], current_state.current_offer[2])) # このoffered_listに追加するのは受け取ったofferのみ
                                else: # 自分が提案者の場合
                                    new_offerer_agents.append(current_state.new_offerer_agents[0]) # 誰への提案か
                                    if current_state.current_offer is not None:
                                        offer_list.append((current_state.current_offer[0], current_state.current_offer[2])) # offer_listに追加するのは自分が提案したofferのみ
                                        
                            # print(f"      New Offers: {current_state.new_offers}")
                            # print(f"      Agreement: {current_state.agreement}")
                            # print(f"      Timed Out: {current_state.timedout}")
                            # print(f"      Broken: {current_state.broken}")
                            # print("-" * 50)
                        
        # print(f"Total Offer Quantity: {total_offer_quantity}") # 交渉中の合計数量(offerもcounterも含む)これが自分のrequireとかけ離れていると損
        # print(f"Total Estimated Price: {total_estimated_price}") # 取引量に価格もかけて、どれくらいの販売益が見込めるか(ただしペナルティは考慮していない)
    return current_proposer, new_offerer_agents, offered_list, offer_list