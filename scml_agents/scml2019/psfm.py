from typing import (
    Dict,
    Iterable,
    Any,
    Callable,
    Collection,
    Type,
    List,
    Optional,
    Union,
)
import negmas
from negmas import (
    Contract,
    Negotiator,
)
from scml.scml2019 import CFP
from scml.scml2019 import DoNothingFactoryManager
from scml.scml2019.factory_managers.builtins import PessimisticNegotiatorUtility


class PenaltySabotageFactoryManager(DoNothingFactoryManager):
    def _hideMoney(self, amount):
        if amount < 0:
            amount = self.awi.state.wallet + amount
        if amount > self.awi.state.wallet:
            return False
        self._hiddenModey += amount
        self.awi.hide_funds(amount)
        return True

    def _unhideMoney(self, amount):
        if amount < 0:
            amount = self._hiddenModey + amount
        if amount > self._hiddenModey:
            return False
        self._hiddenModey -= amount
        self.awi.unhide_funds(amount)
        return True

    def _hideAllMoney(self):
        self._hideMoney(self.awi.state.wallet)

    def _unhideAllMoney(self):
        self._unhideMoney(self._hiddenModey)

    def _adjustMoney(self, amount):
        self._unhideAllMoney()
        self._hideMoney(-amount)

    def init(self):
        """Called once after the agent-world interface is initialized"""
        super().init()
        self._hiddenModey = 0
        self._hideAllMoney()
        pass

    def step(self):
        """Called at every production step by the world"""
        self.awi.bb_remove(section="cfps", query={"publisher": self})
        super().step()
        # storage = sum([_n_stocks for _p_id, _n_stocks in self.awi.state.storage.items()])

        time = min(self.awi.current_step + 15, self.awi.n_steps - 2)
        for product_id in self.consuming.keys():
            product = self.products[product_id]
            if product.catalog_price is None:
                price_range = (0.0, 100.0)
            else:
                price_range = 1000
            cfp = CFP(
                is_buy=True,
                publisher=self.id,
                product=product_id,
                time=time,
                unit_price=price_range,
                quantity=(1, 10),
                penalty=2000,
            )
            for i in range(1):
                self.awi.register_cfp(cfp)
        for product_id in self.producing.keys():
            product = self.products[product_id]
            if product.catalog_price is None:
                price_range = (0.0, 100.0)
            else:
                price_range = 1000
            cfp = CFP(
                is_buy=True,
                publisher=self.id,
                product=product_id,
                time=time,
                unit_price=price_range,
                quantity=(1, 10),
                penalty=2000,
            )
            for i in range(1):
                self.awi.register_cfp(cfp)

        self._hideAllMoney()
        if self.awi.current_step == self.awi.n_steps - 1:
            self._unhideAllMoney()

    # ==========================
    # Important Events Callbacks
    # ==========================

    def on_new_cfp(self, cfp: CFP) -> None:
        """Called when a new CFP for a product for which the agent registered interest is published"""
        if cfp.penalty != None and cfp.penalty >= 100:
            return None

    def respond_to_negotiation_request(
        self, cfp: "CFP", partner: str
    ) -> Optional[Negotiator]:
        """Called whenever someone (partner) is requesting a negotiation with the agent about a Call-For-Proposals
        (cfp) that was earlier published by this agent to the bulletin-board

        Returning `None` means rejecting to enter this negotiation

        """
        if cfp.publisher == self.id:
            pass

        if self.awi.is_bankrupt(partner):
            return None

        ufun_ = PessimisticNegotiatorUtility(
            self, self._create_annotation(cfp=cfp, partner=partner)
        )
        neg = negmas.sao.AspirationNegotiator(
            name=self.name + "*" + partner[:4], ufun=ufun_
        )
        return neg

    def sign_contract(self, contract: Contract) -> Optional[str]:
        """Called after the signing delay from contract conclusion to sign the contract. Contracts become binding
        only after they are signed.

        Remarks:

            - Return `None` if you decided not to sign the contract. Return your ID (self.id) otherwise.

        """
        signature = self.id
        # self.awi.buy_insurance(contract=contract)
        (
            _cfp,
            _seller_id,
            _buyer_id,
            _time,
            _quantity,
            _unit_price,
            _product_id,
        ) = self._split_contract(contract=contract)
        if self.id == _buyer_id:
            _is_buy = True
        elif self.id == _seller_id:
            _is_buy = False
        else:
            _is_buy = None
            print("\033[31mBUY/SELL ERROR\033[0m")
            pass

        # super().sign_contract(contract)
        if not _is_buy:
            return None
        return signature

    def total_utility(self, contracts: Collection[Contract] = ()) -> float:
        """Calculates the total utility for the agent of a collection of contracts"""
        return 100

    def _split_contract(self, contract: Contract) -> Any:
        _cfp = contract.annotation["cfp"]
        _seller_id = contract.annotation["seller"]
        _buyer_id = contract.annotation["buyer"]
        _time = contract.agreement["time"]
        _quantity = contract.agreement["quantity"]
        _unit_price = contract.agreement["unit_price"]
        _product_id = _cfp.product
        return (_cfp, _seller_id, _buyer_id, _time, _quantity, _unit_price, _product_id)
