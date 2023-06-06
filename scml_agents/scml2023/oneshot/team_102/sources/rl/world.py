from __future__ import annotations

import copy
import sys
import time
import traceback
from collections import defaultdict
import random

from negmas.events import Event
from negmas.helpers import (
    exception2str,
)
from negmas.mechanisms import Mechanism
from negmas.situated.agent import Agent
from negmas.situated.common import Operations
from negmas.situated.entity import Entity
from negmas.situated.contract import Contract
from scml.oneshot.world import SCML2023OneShotWorld


class SCML2023RLOneShotWorld(SCML2023OneShotWorld):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # initialize stats
        # ----------------
        self.n_new_contract_executions = 0
        self.n_new_breaches = 0
        self.n_new_contract_errors = 0
        self.n_new_contract_nullifications = 0
        self.activity_level = 0
        self.n_steps_broken, self.n_steps_success = 0, 0
        self.n_broken, self.n_success = 0, 0
        self.stage = 0
        self.stats_stage = 0
        self.blevel = 0.0

        self._n_registered_negotiations_before = len(self._negotiations)
        self.results = tuple()

        # operations
        self.pre_step_operation = (
            Operations.StatsUpdate,
            Operations.SimulationStep,
            Operations.Negotiations
        )
        self.post_step_operations = (
            Operations.ContractSigning,
            Operations.ContractExecution,
            Operations.AgentSteps,
            Operations.SimulationStep,
            Operations.StatsUpdate,
        )

    def _step_a_mechanism(
        self, mechanism, force_immediate_signing
    ) -> tuple[Contract | None, bool]:
        """Steps a mechanism one step.


        Returns:

            The agreement or None and whether the negotiation is still running
        """
        contract = None
        try:
            result = mechanism.step()
        except Exception as e:
            result = mechanism.abort()
            if not self.ignore_negotiation_exceptions:
                raise e
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logerror(
                f"Mechanism exception: " f"{traceback.format_tb(exc_traceback)}",
                Event("entity-exception", dict(exception=e)),
            )
        finally:
            namap = dict()
            for neg in mechanism.negotiators:
                namap[neg.id] = neg.owner

            if mechanism.stats["times"]:
                for source, t in mechanism.stats["times"].items():
                    self.times[namap[source].id if namap[source] else "Unknown"] += t

            if mechanism.stats["exceptions"]:
                for source, exceptions in mechanism.stats["exceptions"].items():
                    self.negotiator_exceptions[
                        namap[source].id if namap[source] else "Unknown"
                    ].append(
                        list(zip(itertools.repeat(self._current_step), exceptions))
                    )

        agreement, is_running = result.agreement, result.running
        if agreement is not None or not is_running:
            negotiation = self._negotiations.get(mechanism.id, None)
            if agreement is None:
                self._register_failed_negotiation(mechanism.nmi, negotiation)
            else:
                contract = self._register_contract(
                    mechanism.nmi,
                    negotiation,
                    self._tobe_signed_at(agreement, force_immediate_signing),
                )
            self._log_negotiation(negotiation)
            if negotiation:
                self._negotiations.pop(mechanism.uuid, None)
        return contract, is_running

    def _step_negotiations(
        self,
        mechanisms: list[Mechanism],
        n_steps: int | float | None,
        force_immediate_signing,
        partners: list[list[Agent]],
    ) -> tuple[list[Contract | None], list[bool], int, int, int, int]:
        """Runs all bending negotiations"""
        running = [_ is not None for _ in mechanisms]
        contracts: list[Contract | None] = [None] * len(mechanisms)
        indices = list(range(len(mechanisms)))
        n_steps_broken_, n_steps_success_ = 0, 0
        n_broken_, n_success_ = 0, 0
        current_step = 0
        if n_steps is None:
            n_steps = float("inf")

        while any(running):
            if self.shuffle_negotiations:
                random.shuffle(indices)
            for i in indices:
                if not running[i]:
                    continue
                if self.time >= self.time_limit:
                    break
                mechanism = mechanisms[i]
                contract, r = self._step_a_mechanism(mechanism, force_immediate_signing)
                contracts[i] = contract
                running[i] = r
                if not running[i]:
                    if contract is None:
                        n_broken_ += 1
                        n_steps_broken_ += mechanism.state.step + 1
                    else:
                        n_success_ += 1
                        n_steps_success_ += mechanism.state.step + 1
                    for _p in partners:
                        self._add_edges(
                            _p[0],
                            _p,
                            self._edges_negotiations_succeeded
                            if contract is not None
                            else self._edges_negotiations_failed,
                            issues=mechanism.issues,
                            bi=True,
                        )
            current_step += 1
            if current_step >= n_steps:
                break
            if self.time >= self.time_limit:
                break
        return (
            contracts,
            running,
            n_steps_broken_,
            n_steps_success_,
            n_broken_,
            n_success_,
        )

    def pre_step(self):
        """Process until RLAgent takes action"""
        if self._start_time is None or self._start_time < 0:
            self._start_time = time.perf_counter()
        if self.time >= self.time_limit:
            return False
        self._n_negs_per_agent_per_step = defaultdict(int)
        if self.current_step >= self.n_steps:
            return False
        self._started = True
        if self.current_step == 0:
            self._sim_start = time.perf_counter()
            self._step_start = self._sim_start
            for priority in sorted(self._entities.keys()):
                for agent in self._entities[priority]:
                    self.call(agent, agent.init_)
                    if self.time >= self.time_limit:
                        return False
            # update monitors
            for monitor in self.stats_monitors:
                if self.safe_stats_monitoring:
                    __stats = copy.deepcopy(self.stats)
                else:
                    __stats = self.stats
                monitor.init(__stats, world_name=self.name)
            for monitor in self.world_monitors:
                monitor.init(self)
        else:
            self._step_start = time.perf_counter()
        # do checkpoint processing
        self.checkpoint_on_step_started()

        for agent in self.agents.values():
            self.call(agent, agent.on_simulation_step_started)
            if self.time >= self.time_limit:
                return False

        self.loginfo(
            f"{len(self._negotiations)} Negotiations/{len(self.agents)} Agents"
        )

        # initialize stats
        # ----------------
        self.n_new_contract_executions = 0
        self.n_new_breaches = 0
        self.n_new_contract_errors = 0
        self.n_new_contract_nullifications = 0
        self.activity_level = 0
        self.n_steps_broken, self.n_steps_success = 0, 0
        self.n_broken, self.n_success = 0, 0
        self.stage = 0
        self.stats_stage = 0
        self.blevel = 0.0

        self._n_registered_negotiations_before = len(self._negotiations)

        operation_map = {
            # Operations.AgentSteps: _step_agents,
            # Operations.ContractExecution: _execute_contracts,
            # Operations.ContractSigning: _sign_contracts,
            Operations.Negotiations: self._pre_run_negotiations,
            Operations.SimulationStep: self._simulation_step,
            Operations.StatsUpdate: self._stats_update,
        }

        for operation in self.pre_step_operation:
            operation_map[operation]()
            if self.time >= self.time_limit:
                return False

    def post_step(self):
        """Process after RLAgent takes action"""

        operation_map = {
            Operations.AgentSteps: self._step_agents,
            Operations.ContractExecution: self._execute_contracts,
            Operations.ContractSigning: self._sign_contracts,
            Operations.Negotiations: self._post_run_negotiations,
            Operations.SimulationStep: self._simulation_step,
            Operations.StatsUpdate: self._stats_update,
        }

        for operation in self.post_step_operations:
            operation_map[operation]()
            if self.time >= self.time_limit:
                return False

        # remove all negotiations that are completed
        # ------------------------------------------
        completed = list(
            k
            for k, _ in self._negotiations.items()
            if _ is not None and _.mechanism.completed
        )
        for key in completed:
            self._negotiations.pop(key, None)

        # update stats
        # ------------
        self._stats["n_registered_negotiations_before"].append(
            self._n_registered_negotiations_before
        )
        self._stats["n_contracts_executed"].append(self.n_new_contract_executions)
        self._stats["n_contracts_erred"].append(self.n_new_contract_errors)
        self._stats["n_contracts_nullified"].append(self.n_new_contract_nullifications)
        self._stats["n_contracts_cancelled"].append(self.__n_contracts_cancelled)
        self._stats["n_contracts_dropped"].append(self.__n_contracts_dropped)
        self._stats["n_breaches"].append(self.n_new_breaches)
        self._stats["breach_level"].append(self.blevel)
        self._stats["n_contracts_signed"].append(self.__n_contracts_signed)
        self._stats["n_contracts_concluded"].append(self.__n_contracts_concluded)
        self._stats["n_negotiations"].append(self.__n_negotiations)
        self._stats["n_negotiation_rounds_successful"].append(self.n_steps_success)
        self._stats["n_negotiation_rounds_failed"].append(self.n_steps_broken)
        self._stats["n_negotiation_successful"].append(self.n_success)
        self._stats["n_negotiation_failed"].append(self.n_broken)
        self._stats["n_registered_negotiations_after"].append(len(self._negotiations))
        self._stats["activity_level"].append(self.activity_level)
        current_time = time.perf_counter() - self._step_start
        self._stats["step_time"].append(current_time)
        total = self._stats.get("total_time", [0.0])[-1]
        self._stats["total_time"].append(total + current_time)
        self.__n_negotiations = 0
        self.__n_contracts_signed = 0
        self.__n_contracts_concluded = 0
        self.__n_contracts_cancelled = 0
        self.__n_contracts_dropped = 0

        self.append_stats()
        for agent in self.agents.values():
            self.call(agent, agent.on_simulation_step_ended)
            if self.time >= self.time_limit:
                return False

        for monitor in self.stats_monitors:
            if self.safe_stats_monitoring:
                __stats = copy.deepcopy(self.stats)
            else:
                __stats = self.stats
            monitor.step(__stats, world_name=self.name)
        for monitor in self.world_monitors:
            monitor.step(self)

        self._current_step += 1
        self.frozen_time = self.time
        # always indicate that the simulation is to continue
        return True

    def _pre_run_negotiations(self, n_steps: int | None = None):
        """Runs all bending negotiations"""
        mechanisms = list(
            (_.mechanism, _.partners)
            for _ in self._negotiations.values()
            if _ is not None
        )
        self.results = self._step_negotiations(
            [_[0] for _ in mechanisms], n_steps, False, [_[1] for _ in mechanisms]
        )

    def _post_run_negotiations(self):
        (
            _,
            _,
            n_steps_broken_,
            n_steps_success_,
            n_broken_,
            n_success_,
        ) = self.results
        if self.time >= self.time_limit:
            return
        n_total_broken = self.n_broken + n_broken_
        if n_total_broken > 0:
            self.n_steps_broken = (
                                     self.n_steps_broken * self.n_broken + n_steps_broken_ * n_broken_
                             ) / n_total_broken
            self.n_broken = n_total_broken
        n_total_success = self.n_success + n_success_
        if n_total_success > 0:
            self.n_steps_success = (
                                      self.n_steps_success * self.n_success + n_steps_success_ * n_success_
                              ) / n_total_success
            self.n_success = n_total_success

    def _step_agents(self):
        # Step all entities in the world once:
        # ------------------------------------
        # note that entities are simulated in the partial-order specified by their priority value
        tasks: list[Entity] = []
        for priority in sorted(self._entities.keys()):
            tasks += [_ for _ in self._entities[priority]]

        for task in tasks:
            self.call(task, task.step_)
            if self.time >= self.time_limit:
                break

    def _sign_contracts(self):
        self._process_unsigned()

    def _simulation_step(self):
        try:
            self.simulation_step(self.stage)
            if self.time >= self.time_limit:
                return
        except Exception as e:
            self.simulation_exceptions[self._current_step].append(exception2str())
            if not self.ignore_simulation_exceptions:
                raise (e)
        self.stage += 1

    def _execute_contracts(self):
        # execute contracts that are executable at this step
        # --------------------------------------------------
        current_contracts = [
            _ for _ in self.executable_contracts() if _.nullified_at < 0
        ]
        if len(current_contracts) > 0:
            # remove expired contracts
            executed = set()
            current_contracts = self.order_contracts_for_execution(
                current_contracts
            )

            for contract in current_contracts:
                if self.time >= self.time_limit:
                    break
                if contract.signed_at < 0:
                    continue
                try:
                    contract_breaches = self.start_contract_execution(contract)
                except Exception as e:
                    for p in contract.partners:
                        self.contracts_erred[p] += 1
                    self.contract_exceptions[self._current_step].append(
                        exception2str()
                    )
                    contract.executed_at = self.current_step
                    self._saved_contracts[contract.id]["breaches"] = ""
                    self._saved_contracts[contract.id]["executed_at"] = -1
                    self._saved_contracts[contract.id]["dropped_at"] = -1
                    self._saved_contracts[contract.id]["nullified_at"] = -1
                    self._saved_contracts[contract.id][
                        "erred_at"
                    ] = self._current_step
                    self._add_edges(
                        contract.partners[0],
                        contract.partners,
                        self._edges_contracts_erred,
                        bi=True,
                    )
                    self.n_new_contract_errors += 1
                    if not self.ignore_contract_execution_exceptions:
                        raise e
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    self.logerror(
                        f"Contract exception @{str(contract)}: "
                        f"{traceback.format_tb(exc_traceback)}",
                        Event(
                            "contract-exception",
                            dict(contract=contract, exception=e),
                        ),
                    )
                    continue
                if contract_breaches is None:
                    for p in contract.partners:
                        self.contracts_nullified[p] += 1
                    self._saved_contracts[contract.id]["breaches"] = ""
                    self._saved_contracts[contract.id]["executed_at"] = -1
                    self._saved_contracts[contract.id]["dropped_at"] = -1
                    self._saved_contracts[contract.id][
                        "nullified_at"
                    ] = self._current_step
                    self._add_edges(
                        contract.partners[0],
                        contract.partners,
                        self._edges_contracts_nullified,
                        bi=True,
                    )
                    self._saved_contracts[contract.id]["erred_at"] = -1
                    self.n_new_contract_nullifications += 1
                    self.loginfo(
                        f"Contract nullified: {str(contract)}",
                        Event("contract-nullified", dict(contract=contract)),
                    )
                elif len(contract_breaches) < 1:
                    for p in contract.partners:
                        self.contracts_executed[p] += 1
                    self._saved_contracts[contract.id]["breaches"] = ""
                    self._saved_contracts[contract.id]["dropped_at"] = -1
                    self._saved_contracts[contract.id][
                        "executed_at"
                    ] = self._current_step
                    self._add_edges(
                        contract.partners[0],
                        contract.partners,
                        self._edges_contracts_executed,
                        bi=True,
                    )
                    self._saved_contracts[contract.id]["nullified_at"] = -1
                    self._saved_contracts[contract.id]["erred_at"] = -1
                    executed.add(contract)
                    self.n_new_contract_executions += 1
                    _size = self.contract_size(contract)
                    if _size is not None:
                        self.activity_level += _size
                    for partner in contract.partners:
                        self.call(
                            self.agents[partner],
                            self.agents[partner].on_contract_executed,
                            contract,
                        )
                        if self.time >= self.time_limit:
                            break
                else:
                    for p in contract.partners:
                        self.contracts_breached[p] += 1
                    self._saved_contracts[contract.id]["executed_at"] = -1
                    self._saved_contracts[contract.id]["nullified_at"] = -1
                    self._saved_contracts[contract.id]["dropped_at"] = -1
                    self._saved_contracts[contract.id]["erred_at"] = -1
                    self._saved_contracts[contract.id]["breaches"] = "; ".join(
                        f"{_.perpetrator}:{_.type}({_.level})"
                        for _ in contract_breaches
                    )
                    breachers = {
                        (_.perpetrator, tuple(_.victims)) for _ in contract_breaches
                    }
                    for breacher, victims in breachers:
                        if isinstance(victims, str) or isinstance(victims, Agent):
                            victims = [victims]
                        self._add_edges(
                            breacher,
                            victims,
                            self._edges_contracts_breached,
                            bi=False,
                        )
                    for b in contract_breaches:
                        self._saved_breaches[b.id] = b.as_dict()
                        self.loginfo(
                            f"Breach of {str(contract)}: {str(b)} ",
                            Event(
                                "contract-breached",
                                dict(contract=contract, breach=b),
                            ),
                        )
                    resolution = self._process_breach(
                        contract, list(contract_breaches)
                    )
                    if resolution is None:
                        self.n_new_breaches += 1
                        self.blevel += sum(_.level for _ in contract_breaches)
                    else:
                        self.n_new_contract_executions += 1
                        self.loginfo(
                            f"Breach resolution cor {str(contract)}: {str(resolution)} ",
                            Event(
                                "breach-resolved",
                                dict(
                                    contract=contract,
                                    breaches=list(contract_breaches),
                                    resolution=resolution,
                                ),
                            ),
                        )
                    self.complete_contract_execution(
                        contract, list(contract_breaches), resolution
                    )
                    self.loginfo(
                        f"Executed {str(contract)}",
                        Event("contract-executed", dict(contract=contract)),
                    )
                    for partner in contract.partners:
                        self.call(
                            self.agents[partner],
                            self.agents[partner].on_contract_breached,
                            contract,
                            list(contract_breaches),
                            resolution,
                        )
                        if self.time >= self.time_limit:
                            break
                contract.executed_at = self.current_step
        dropped = self.get_dropped_contracts()
        self.delete_executed_contracts()  # note that all contracts even breached ones are to be deleted
        for c in dropped:
            self.loginfo(
                f"Dropped {str(c)}",
                Event("dropped-contract", dict(contract=c)),
            )
            self._saved_contracts[c.id]["dropped_at"] = self._current_step
            for p in c.partners:
                self.contracts_dropped[p] += 1
        self.__n_contracts_dropped += len(dropped)

    def _stats_update(self):
        self.update_stats(self.stats_stage)
        self.stats_stage += 1

    def step_env(self):
        """step RL environment"""
        pass
