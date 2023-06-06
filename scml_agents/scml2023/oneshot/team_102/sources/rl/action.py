from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse, SAONMI
from scml.oneshot import *

__all__ = [
    'SyncActionManager',
    'IndActionManager',
    'SyncMultiDiscreteAM',
    'IndMultiDiscreteAM',
]


class SyncActionManager(ABC):
    def __str__(self):
        return 'S-AM'

    @property
    def action_space(self) -> int | List[int]:
        return 1

    @abstractmethod
    def decode(
            self,
            action: np.ndarray,
            responses: Dict[str, SAOResponse],
            nmi: SAONMI,
            step: int,
    ) -> Dict[str, SAOResponse]:
        ...

    @abstractmethod
    def encode(
            self,
            responses: Dict[str, SAOResponse]
    ) -> np.ndarray:
        ...


class SyncMultiDiscreteAM(SyncActionManager):
    def __init__(
            self,
            n_prices=2,
            n_quantity=10,
            n_responses=2,
            n_opp_agents=10,
    ):
        self.n_prices = n_prices
        self.n_quantity = n_quantity
        self.n_responses = n_responses
        self.n_opp_agents = n_opp_agents

    def __str__(self):
        return 'S-MD-AM'

    @property
    def action_space(self) -> int | List[int]:
        n_actions = [
                        self.n_prices,
                        self.n_quantity,
                        self.n_responses,
                    ] * self.n_opp_agents
        return n_actions

    @property
    def n_action_kind(self) -> int:
        return int(len(self.action_space) / self.n_opp_agents)

    def decode(
            self,
            action: np.ndarray,
            responses: Dict[str, SAOResponse],
            nmi: SAONMI,
            step: int,
    ) -> Dict[str, SAOResponse]:
        for i, opp_id in enumerate(responses.keys()):
            offer = [-1, step, -1]

            idx = i * self.n_action_kind
            price, quantity, response = action[idx: idx + self.n_action_kind]

            # price
            if price == 0:
                offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].max_value
            else:
                offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].min_value

            # quantity
            offer[QUANTITY] = quantity

            # responses
            if response == 0:
                response = ResponseType.ACCEPT_OFFER
            else:
                response = ResponseType.REJECT_OFFER

            responses[opp_id] = SAOResponse(response, offer)

        return responses

    def encode(
            self,
            responses: Dict[str, SAOResponse]
    ) -> np.ndarray:
        pass


class IndActionManager(ABC):
    def __str__(self):
        return 'I-AM'

    @property
    def action_space(self) -> int | List[int]:
        return 1

    @abstractmethod
    def decode(
            self,
            action: np.ndarray,
            nmi: SAONMI,
            step: int,
    ) -> SAOResponse:
        ...

    @abstractmethod
    def encode(
            self,
            response: SAOResponse,
    ) -> np.ndarray:
        ...


class IndMultiDiscreteAM(IndActionManager):
    def __init__(
            self,
            n_prices=2,
            n_quantity=10,
            n_responses=2,
    ):
        self.n_prices = n_prices
        self.n_quantity = n_quantity
        self.n_responses = n_responses

    def __str__(self):
        return 'I-MD-AM'

    @property
    def action_space(self) -> int | List[int]:
        n_actions = [
            self.n_prices,
            self.n_quantity,
            self.n_responses,
        ]
        return n_actions

    def decode(
            self,
            action: np.ndarray,
            nmi: SAONMI,
            step: int,
    ) -> SAOResponse:
        offer = [-1, step, -1]

        price, quantity, response = action

        # price
        if price == 0:
            offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].max_value
        else:
            offer[UNIT_PRICE] = nmi.issues[UNIT_PRICE].min_value

        # quantity
        offer[QUANTITY] = quantity

        # responses
        if response == 0:
            response = ResponseType.ACCEPT_OFFER
        else:
            response = ResponseType.REJECT_OFFER

        response = SAOResponse(response, tuple(offer))

        return response

    def encode(
            self,
            responses: Dict[str, SAOResponse]
    ) -> np.ndarray:
        pass
