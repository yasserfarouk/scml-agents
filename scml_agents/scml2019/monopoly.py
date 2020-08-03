from negmas import Contract, Negotiator, NiceNegotiator
from scml.scml2019 import DoNothingFactoryManager, CFP
import random

from typing import Optional

__all__ = ["Monopoly"]


class Monopoly(DoNothingFactoryManager):
    def step(self):
        if self.awi.current_step + 10 > self.awi.n_steps:
            return
        for output in [process.outputs[0] for process in self.awi.processes]:
            for _ in range(1):
                cfp = CFP(
                    is_buy=True,
                    publisher=self.id,
                    product=output.product,
                    time=(self.awi.current_step + 6, self.awi.current_step + 9),
                    unit_price=(
                        self.awi.products[output.product].catalog_price * 0.5,
                        self.awi.products[output.product].catalog_price * 1.2,
                    ),
                    quantity=5,
                    penalty=random.randint(100, 1000),
                )
                self.awi.register_cfp(cfp)

    def confirm_contract_execution(self, contract: Contract) -> bool:
        return True

    def respond_to_negotiation_request(
        self, cfp: "CFP", partner: str
    ) -> Optional[Negotiator]:
        if self.awi.is_bankrupt(partner):
            return None
        return NiceNegotiator()
