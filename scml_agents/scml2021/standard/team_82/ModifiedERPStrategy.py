from typing import List, Optional

from negmas import Breach, Contract
from scml import ExecutionRatePredictionStrategy


class ModifiedERPStrategy(ExecutionRatePredictionStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._execution_fractions = dict()
        self._default_execution_fraction = 0.5
        self._total_quantity = 0

    def predict_quantity(self, contract: Contract):
        return (
            contract.agreement["quantity"]
            * self._execution_fractions[contract.partners[1]]
        )

    def on_contract_executed(self, contract: Contract) -> None:
        super().on_contract_executed(contract)
        q = contract.agreement["quantity"]
        self._total_quantity += q
        my_partner = contract.partners[1]
        if my_partner not in self._execution_fractions:
            self._execution_fractions[my_partner] = 0.5
            self._execution_fractions[my_partner] += 0.05
        else:
            self._execution_fractions[my_partner] += 0.05

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        super().on_contract_breached(contract, breaches, resolution)
        self._total_quantity += contract.agreement["quantity"]
        my_partner = contract.partners[1]
        if my_partner not in self._execution_fractions:
            self._execution_fractions[my_partner] = 0.5
            self._execution_fractions[my_partner] -= 0.05
        else:
            self._execution_fractions[my_partner] -= 0.05
