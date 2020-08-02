import sys
from typing import Union, Tuple, Optional
import scml_agents.scml2019 as scml2019
import scml_agents.scml2020 as scml2020

# import scml_agents.contrib as contrib
from negmas.helpers import get_full_type_name
from negmas import Agent

__all__ = ["get_agents"]


def get_agents(
    version: Union[str, int],
    *,
    qualified_only: bool = True,
    finalists_only: bool = True,
    winners_only: bool = True,
    top_only: Optional[Union[int, float]] = None,
    as_class: bool = True,
) -> Tuple[Union[Agent, str]]:
    """
    Gets agent classes/full class names for a version which can either be a competition year (int) or "contrib".

    Args:
        version: Either a competition year (2019, 2020) or the value "contrib" for all other agents
        finalists_only: If true, only agents that were submitted to SCML and passed qualifications will be 
                        returned
        winners_only: If true, only winners of SCML (the given version) will be returned.
        top_only: Either a fraction of finalists or the top n finalists with highest scores in the finals of 
                  SCML
        as_class: If true, the agent classes will be returned otherwise their full class names.
    """
    if isinstance(version, int) and version == 2019:
        classes = (
            scml2019.FJ2FactoryManager,
            scml2019.RaptFactoryManager,
            scml2019.InsuranceFraudFactoryManager,
            scml2019.SAHAFactoryManager,
            scml2019.CheapBuyerFactoryManager,
            scml2019.NVMFactoryManager,
        )
    elif isinstance(version, int) and version == 2020:
        classes = tuple(
            sum(
                [
                    [f"{_.__name__}.{a}" for a in _.__all__]
                    for _ in (
                        scml2020.team_may,
                        scml2020.team_22,
                        scml2020.team_25,
                        scml2020.team_15,
                        scml2020.bargent,
                        scml2020.agent0x111,
                        scml2020.a_sengupta,
                        scml2020.past_frauds,
                        scml2020.monty_hall,
                        scml2020.team_19,
                        scml2020.team_17,
                        scml2020.team_10,
                        scml2020.threadfield,
                        scml2020.team_29,
                        scml2020.team_20,
                        scml2020.team_27,
                        scml2020.team_18,
                        scml2020.biu_th,
                        scml2020.team_32,
                    )
                ],
                start=[],
            )
        )
    elif isinstance(version, str) and version == "contrib":
        classes = tuple()
    else:
        raise ValueError(
            f"The version {version} is unknown. Valid versions are 2019, 2020 (as ints), 'contrib' as a string"
        )
    if not as_class:
        return tuple(get_full_type_name(_) for _ in classes)
    return classes
