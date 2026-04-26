from .bilat_ufun import BilatUFun
from .negotiation_history import BilateralHistory


class OpponentModel:
    """Negotiation histories -> opponent ufun"""

    def __init__(self, neg_id: str):
        self._neg_id = neg_id

    def __call__(
        self, my_ufun: BilatUFun, histories: List[BilateralHistory]
    ) -> BilatUFun:
        raise NotImplementedError


class OpponentModelStatic:
    """Assumes opponent ufun is an average"""

    def __call__(
        self, my_ufun: BilatUFun, histories: List[BilateralHistory]
    ) -> BilatUFun:
        raise NotImplementedError


class OpponentModelPareto:
    """Assumes opponent offers are on the pareto frontier"""

    def __call__(
        self, my_ufun: BilatUFun, histories: List[BilateralHistory]
    ) -> BilatUFun:
        raise NotImplementedError
