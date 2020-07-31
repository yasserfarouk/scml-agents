from negmas import (
    Issue,
    AgentMechanismInterface,
    Negotiator,
    MechanismState,
    Contract,
    Breach,
    SAOController,
    AspirationNegotiator,
)
from scml.scml2020 import SCML2020Agent, AWI, FactoryState, Failure
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

__all__ = ["PrintingAgent", "PrintingSAOController"]

logger = logging.getLogger("Logging")
target_agent: str = ""
last_log: str = ""
printing_target: int = 1


class PrintingAgent(SCML2020Agent):
    def __init__(self, *args, is_debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_debug = is_debug

    def print(self, logs, force=False):
        global target_agent
        global last_log
        if target_agent == "":
            # if target_agent.split("@")[-1] == str(printing_target):
            target_agent = self.id
            logger.setLevel(50 - 30 * self.is_debug)
            logger.log(30, "Printing target: {}".format(self.id))
        if target_agent == self.id or force:
            if logs != last_log:
                logger.log(30, logs)

    def request_negotiations(
        self,
        is_buy: bool,
        product: int,
        quantity: Union[int, Tuple[int, int]],
        unit_price: Union[int, Tuple[int, int]],
        time: Union[int, Tuple[int, int]],
        controller: Optional[SAOController],
        negotiators: List[Negotiator] = None,
        partners: List[str] = None,
        extra: Dict[str, Any] = None,
    ) -> bool:
        self.print(
            "<<<Outgoing {} Negotiation from {}: issue[quantity:{} unit_price:{}, time:{}]".format(
                "Buy" if is_buy else "sell", self.id, quantity, unit_price, time
            )
        )
        awi: AWI = self.awi
        result: bool = awi.request_negotiations(
            is_buy=is_buy,
            product=product,
            quantity=quantity,
            unit_price=unit_price,
            time=time,
            controller=controller,
            negotiators=negotiators,
            partners=partners,
            extra=extra,
        )
        return result

    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        super().on_neg_request_accepted(req_id, mechanism)
        self.print(
            "~>>Outgoing Negotiation Accepted by {}: ID({}) issue{}".format(
                mechanism.annotation["seller"]
                if mechanism.annotation["buyer"] == self.id
                else mechanism.annotation["buyer"],
                mechanism["id"][:4],
                str(mechanism["issues"]),
            )
        )

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        super().on_neg_request_rejected(req_id, by)
        for rejecter in by:
            self.print(
                "~>|Outgoing Negotiation Rejected by {}: ID({})".format(
                    rejecter, req_id[:4]
                )
            )

    def init(self, *agrs, **kwargs):
        """Called once after the agent-world interface is initialized"""
        super().init(*agrs, **kwargs)
        awi = self.awi  # type: AWI
        self.i_step_step_base: int = -1
        self.print("I am {}".format(self.name), force=True)
        self.print("My lines: {}".format(awi.state.n_lines), force=True)
        self.print("Production cost: {}".format(min(awi.profile.costs[0])), force=True)
        self.print("Catalog price: {}".format(awi.catalog_prices))
        self.print(
            "Class inheritance:\n{}".format(self.__class__.__mro__).replace(",", "\n")
        )
        pass

    def step(self, *args, **kwargs):
        """Called at every production step by the world"""
        awi = self.awi  # type: AWI
        factory_state = awi.state  # type: FactoryState
        self.i_step_step_base += 1
        self.print("-------------")
        self.print("I am {}".format(self.name))
        self.print("Step:{}/{}".format(self.awi.current_step, self.awi.n_steps))
        self.print("My inventory: {}".format(factory_state.inventory))
        self.print("My balance: {}".format(factory_state.balance))
        self.print("My lines are to work:")
        line_usage_list: List[int] = []
        for i in range(awi.n_steps):
            i_line_working = factory_state.commands[i]
            line_usage_list.append(-sum(i_line_working == -1) + factory_state.n_lines)
        self.print("line usage: {}".format(line_usage_list))
        if self.awi.current_step >= 1:
            self.print("Financial Report:")
            fr_list = awi.reports_at_step(
                self.awi.current_step - 1
            )  # type: Dict[str, FinancialReport]
            if fr_list is not None:
                for key, x in fr_list.items():
                    self.print(
                        "Agent: {},\tcash: {:>7},\tassets: {:>5},\tbreach prob: {:.5f},\tbreach level:{:.5f},"
                        "\tbankrupt:{},".format(
                            x.agent_id,
                            x.cash,
                            x.assets,
                            x.breach_prob,
                            x.breach_level,
                            x.is_bankrupt,
                        )
                    )
        super().step(*args, **kwargs)

    # ================================
    # Negotiation Control and Feedback
    # ================================
    def on_accepting_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> None:
        if self.awi.current_step == self.i_step_step_base:
            self.print(
                "<<~Accept Incoming Negotiation from {}: ID({}) issue{}".format(
                    initiator, mechanism["id"][:4], str(issues)
                )
            )
        else:
            self.print(
                "<<<<<~Accept Incoming Negotiation for next from {}: ID({}) issue{}".format(
                    initiator, mechanism["id"][:4], str(issues)
                )
            )

    def on_rejecting_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> None:
        if self.awi.current_step == self.i_step_step_base:
            self.print(
                "|<~Reject Incoming Negotiation from {}: ID({}) issue{}".format(
                    initiator, mechanism["id"][:4], str(issues)
                )
            )
        else:
            self.print(
                "|<<<<~Reject Incoming Negotiation for next from {}: ID({}) issue{}".format(
                    initiator, mechanism["id"][:4], str(issues)
                )
            )

    def respond_to_negotiation_request(
        self,
        initiator: str,
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
    ) -> Optional[Negotiator]:
        """Called whenever an agent requests a negotiation with you.
        Return either a negotiator to accept or None (default) to reject it"""
        if self.awi.current_step == self.i_step_step_base:
            self.print(
                ">>>Incoming Negotiation from {}: ID({}) issue{}".format(
                    initiator, mechanism["id"][:4], str(issues)
                )
            )
        else:
            self.print(
                ">>>>>>Incoming Negotiation for next from {}: ID({}) issue{}".format(
                    initiator, mechanism["id"][:4], str(issues)
                )
            )
        return super().respond_to_negotiation_request(
            initiator, issues, annotation, mechanism
        )

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without
        agreement"""
        self.print(
            "<|>{} Negotiation failure w/ {}: ID({}) issue{}".format(
                "Sell" if annotation["seller"] == self.id else "Buy",
                list(set(partners) - set([self.id]))[0],
                mechanism["id"][:4],
                str(mechanism["issues"]),
            )
        )
        super().on_negotiation_failure(partners, annotation, mechanism, state)

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called when a negotiation the agent is a party of ends with
        agreement"""
        self.print(
            "|||{} Negotiation success w/ {}: ID({}) Agreement{} issue{}".format(
                "Sell" if contract.annotation["seller"] == self.id else "Buy",
                list(set(contract["partners"]) - set([self.id]))[0],
                mechanism["id"][:4],
                contract["agreement"],
                str(mechanism["issues"]),
            )
        )
        super().on_negotiation_success(contract, mechanism)

    # =============================
    # Contract Control and Feedbackt
    # =============================

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        """Called to ask you to sign all contracts that were concluded in
        one step (day)"""
        return super().sign_all_contracts(contracts)

    def on_contracts_finalized(
        self,
        signed: List[Contract],
        cancelled: List[Contract],
        rejectors: List[List[str]],
    ) -> None:
        """Called to inform you about the final status of all contracts in
        a step (day)"""
        super().on_contracts_finalized(signed, cancelled, rejectors)

    def on_contract_executed(self, contract: Contract) -> None:
        """Called when a contract executes successfully and fully"""
        super().on_contract_executed(contract)

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        """Called when a breach occur. In 2020, there will be no resolution
        (i.e. resoluion is None)"""
        super().on_contract_breached(contract, breaches, resolution)

    # ====================
    # Production Callbacks
    # ====================

    def confirm_production(
        self, commands: np.ndarray, balance: int, inventory: np.ndarray
    ) -> np.ndarray:
        """
        Called just before production starts at every step allowing the
        agent to change what is to be produced in its factory on that step.
        """
        return super().confirm_production(commands, balance, inventory)

    def on_failures(self, failures: List[Failure]) -> None:
        """Called when production fails. If you are careful in
        what you order in `confirm_production`, you should never see that."""
        super().on_failures(failures)

    # ==========================
    # Callback about Bankruptcy
    # ==========================

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: int,
        compensation_money: int,
    ) -> None:
        """Called whenever any agent goes bankrupt. It informs you about changes
        in future contracts you have with you (if any)."""
        self.print(")))Agent {} bankrupt".format(agent))
        super().on_agent_bankrupt(agent, contracts, quantities, compensation_money)


class PrintingSAOController(SAOController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_join(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
        **kwargs
    ) -> bool:
        """
        Called by children negotiators to get permission to join negotiations

        Args:
            negotiator_id: The negotiator ID
            ami  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.

        Returns:
            True if the negotiator is allowed to join the negotiation otherwise
            False

        """
        return super().before_join(
            negotiator_id=negotiator_id, ami=ami, state=state, ufun=ufun, role=role
        )

    def after_join(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent"
    ) -> None:
        """
        Called by children negotiators after joining a negotiation to inform
        the controller

        Args:
            negotiator_id: The negotiator ID
            ami  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.
        """
        super().after_join(
            negotiator_id=negotiator_id, ami=ami, state=state, ufun=ufun, role=role
        )

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:
        return super().propose(negotiator_id=negotiator_id, state=state)

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        return super().respond(negotiator_id=negotiator_id, state=state, offer=offer)

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        super().on_negotiation_end(negotiator_id=negotiator_id, state=state)

    def on_negotiation_start(self, negotiator_id: str, state: MechanismState) -> None:
        super().on_negotiation_start(negotiator_id=negotiator_id, state=state)
