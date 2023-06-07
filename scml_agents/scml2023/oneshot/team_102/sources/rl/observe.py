from abc import abstractmethod
from typing import Dict, List

import numpy as np
import torch
from negmas import MechanismState
from negmas.sao.common import SAONMI, SAOState
from scml.oneshot import *

__all__ = [
    "SyncObserveManager",
    "SyncBoxOM",
    "SyncMultiDiscreteOM",
    "IndObserveManager",
    "IndBoxOM",
    "IndMultiDiscreteOM",
]


class SyncObserveManager:
    def __str__(self):
        return "S-OM"

    @property
    def state_dim(self) -> int:
        ...

    @abstractmethod
    def encode(
        self,
        offers: Dict[str, tuple],
        states: Dict[str, SAOState],
        needs: int,
        nmi: SAONMI,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def decode(self, state: torch.Tensor, opp_ids: List[str]) -> Dict[str, tuple]:
        ...


class SyncBoxOM(SyncObserveManager):
    def __init__(
        self,
        n_common=2,
        n_per_offer=3,
        n_opp_agents=10,
    ):
        self.n_common = n_common
        self.n_per_offer = n_per_offer
        self.n_opp_agents = n_opp_agents

    def __str__(self):
        return "S-B-OM"

    @property
    def state_dim(self) -> int:
        n_state = self.n_common + self.n_per_offer * self.n_opp_agents
        return n_state

    def encode(
        self,
        offers: Dict[str, tuple],
        states: Dict[str, SAOState],
        needs: int,
    ) -> torch.Tensor:
        state = torch.zeros(self.state_dim)

        # common state
        for i, s in enumerate(
            [
                list(states.values())[0].step,
                needs,
            ]
        ):
            state[i] = s

        # offers
        sorted_offers = dict((k, v) for k, v in sorted(offers.items()))
        idx = self.n_common
        for i, o in enumerate(sorted_offers.items()):
            opp_id, offer = o
            for j, v in enumerate(offer):
                state[idx + i * self.n_per_offer + j] = v

        return state

    def decode(self, state: torch.Tensor, opp_ids: List[str]) -> Dict[str, tuple]:
        pass


class SyncMultiDiscreteOM(SyncObserveManager):
    def __init__(
        self,
        n_rounds=21,
        n_needs=11,
        n_prices=2,
        n_quantity=11,
        n_opp_agents=10,
    ):
        self.n_rounds = n_rounds
        self.n_needs = n_needs
        self.n_prices = n_prices
        self.n_quantity = n_quantity
        self.n_opp_agents = n_opp_agents

    def __str__(self):
        return "S-MD-OM"

    @property
    def state_dim(self) -> int:
        n_state = sum(
            [
                self.n_rounds,
                self.n_needs,
            ]
            + [
                self.n_quantity,
                self.n_rounds,
                self.n_prices,
            ]
            * self.n_opp_agents
        )
        return n_state

    def encode(
        self,
        offers: Dict[str, tuple],
        states: Dict[str, SAOState],
        needs: int,
        nmi: SAONMI,
    ) -> torch.Tensor:
        state = []

        def int_to_array(s: int, n: int):
            a = torch.zeros(n)
            s = np.clip(s, a_min=0, a_max=n - 1)
            a[s] = 1
            return a

        # common state
        state += [
            int_to_array(list(states.values())[0].step, self.n_rounds),
            int_to_array(needs, self.n_needs),
        ]

        # offers
        sorted_offers = dict((k, v) for k, v in sorted(offers.items()))
        for opp_id, offer in sorted_offers.items():
            price = 0 if offer[UNIT_PRICE] == nmi.issues[UNIT_PRICE].max_value else 1
            state += [
                int_to_array(offer[QUANTITY], self.n_quantity),
                int_to_array(offer[TIME], self.n_rounds),
                int_to_array(price, self.n_prices),
            ]
        pad = torch.full([self.state_dim - sum([len(_) for _ in state])], 0)
        state += [pad]
        state = torch.cat(state, dim=0)

        return state

    def decode(self, state: torch.Tensor, opp_ids: List[str]) -> Dict[str, tuple]:
        pass


class IndObserveManager:
    def __str__(self):
        return "I-OM"

    @property
    def state_dim(self) -> int:
        ...

    @abstractmethod
    def encode(
        self,
        offer: tuple,
        state: MechanismState,
        needs: int,
        nmi: SAONMI,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def decode(
        self,
        state: torch.Tensor,
    ) -> tuple:
        ...


class IndBoxOM(IndObserveManager):
    def __init__(
        self,
        n_common=2,
        n_per_offer=3,
    ):
        self.n_common = n_common
        self.n_per_offer = n_per_offer

    def __str__(self):
        return "I-B-OM"

    @property
    def state_dim(self) -> int:
        n_state = self.n_common + self.n_per_offer
        return n_state

    def encode(
        self,
        offer: tuple,
        state: MechanismState,
        needs: int,
        nmi: SAONMI,
    ) -> torch.Tensor:
        obs = torch.zeros(self.state_dim)

        # common state
        for i, s in enumerate(
            [
                state.step,
                needs,
            ]
        ):
            obs[i] = s

        # offers
        idx = self.n_common
        obs[idx : idx + self.n_per_offer] = torch.tensor(offer)

        return obs

    def decode(
        self,
        state: torch.Tensor,
    ) -> tuple:
        pass


class IndMultiDiscreteOM(IndObserveManager):
    def __init__(
        self,
        n_rounds=21,
        n_needs=11,
        n_prices=2,
        n_quantity=11,
    ):
        self.n_rounds = n_rounds
        self.n_needs = n_needs
        self.n_prices = n_prices
        self.n_quantity = n_quantity

    def __str__(self):
        return "I-MD-OM"

    @property
    def state_dim(self) -> int:
        n_state = sum(
            [
                self.n_rounds,
                self.n_needs,
            ]
            + [
                self.n_quantity,
                self.n_rounds,
                self.n_prices,
            ]
        )
        return n_state

    def encode(
        self,
        offer: tuple,
        state: MechanismState,
        needs: int,
        nmi: SAONMI,
    ) -> torch.Tensor:
        obs = []

        def int_to_array(s: int, n: int):
            a = torch.zeros(n)
            s = np.clip(s, a_min=0, a_max=n - 1)
            a[s] = 1
            return a

        # common state
        obs += [
            int_to_array(state.step, self.n_rounds),
            int_to_array(needs, self.n_needs),
        ]

        # offers
        price = 0 if offer[UNIT_PRICE] == nmi.issues[UNIT_PRICE].max_value else 1
        obs += [
            int_to_array(offer[QUANTITY], self.n_quantity),
            int_to_array(offer[TIME], self.n_rounds),
            int_to_array(price, self.n_prices),
        ]

        obs = torch.cat(obs, dim=0)

        return obs

    def decode(
        self,
        state: torch.Tensor,
    ) -> tuple:
        pass
