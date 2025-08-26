from scml.oneshot.context import LimitedPartnerNumbersOneShotContext

# from scml.oneshot.agents.greedy import GreedyOneShotAgent
# from scml.oneshot.agent import OneShotAgent
# from scml_agents.scml2024.oneshot.team_miyajima_oneshot.cautious import CautiousOneShotAgent
# from scml_agents.scml2024.oneshot.team_193.matching_pennies2 import MatchingPennies
# from scml_agents.scml2024.oneshot.team_171.dist_redist_agent import DistRedistAgent
# from scml_agents.scml2024.oneshot.teamyuzuru.epsilon_greedy_agent import EpsilonGreedyAgent
# from scml_agents.scml2024.oneshot.team_abc.suzuka_agent import SuzukaAgent
# from scml.oneshot.agents.rand import (
#     EqualDistOneShotAgent,
#     NiceAgent,
#     RandDistOneShotAgent,
# )

N_SUPPLIERS = (4, 8)
"""Numbers of suppliers supported"""
N_CONSUMERS = (4, 8)
"""Numbers of consumers supported"""
# NTESTS = 20
# DEFAULT_PLACEHOLDER_AGENT_TYPES = (Placeholder,)

# WARN_ON_FAILURE = True
# RAISE_ON_FAILURE = False

DefaultAgentsOneShot = (
    # CautiousOneShotAgent,
    # MatchingPennies,
    # DistRedistAgent,
    # EpsilonGreedyAgent,
    # SuzukaAgent,
    # RandDistOneShotAgent,
    # EqualDistOneShotAgent,
)

class MySupplierContext(LimitedPartnerNumbersOneShotContext):
    def __init__(self, *args, **kwargs):
        n_agents_per_process = (
            min(N_SUPPLIERS[0], N_CONSUMERS[0]),  # type: ignore
            max(N_SUPPLIERS[1], N_CONSUMERS[1]),  # type: ignore
        )
        kwargs |= dict(
            n_suppliers=(0, 0),  # suppliers have no suppliers
            n_consumers=N_CONSUMERS,
            n_competitors=(N_SUPPLIERS[0] - 1, N_SUPPLIERS[1] - 1),
            n_agents_per_process=n_agents_per_process,
            level=0,  # suppliers are always in the first level
            non_competitors = DefaultAgentsOneShot,
        )
        super().__init__(*args, **kwargs)

    

class MyConsumerContext(LimitedPartnerNumbersOneShotContext):
    """A world context that can generate any world compatible with the observation manager"""

    def __init__(self, *args, **kwargs):
        n_agents_per_process = (
            min(N_SUPPLIERS[0], N_CONSUMERS[0]),  # type: ignore
            max(N_SUPPLIERS[1], N_CONSUMERS[1]),  # type: ignore
        )
        kwargs |= dict(
            n_suppliers=N_SUPPLIERS,
            n_consumers=(0, 0),  # consumers have no consumers
            n_competitors=(N_CONSUMERS[0] - 1, N_CONSUMERS[1] - 1),
            n_agents_per_process=n_agents_per_process,
            level=-1,  # consumers are always in the last level
            non_competitors = DefaultAgentsOneShot,
        )
        super().__init__(*args, **kwargs)