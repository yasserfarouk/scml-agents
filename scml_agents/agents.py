from __future__ import annotations

from inspect import ismodule
from typing import Literal, overload

from negmas.helpers import get_class, get_full_type_name
from negmas.situated import Agent

import scml_agents.scml2019 as scml2019
import scml_agents.scml2020 as scml2020
import scml_agents.scml2021 as scml2021
import scml_agents.scml2022 as scml2022

__all__ = ["get_agents"]


@overload
def get_agents(
    version: str | int,
    *,
    track: str = "any",
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
    bird_only: bool = False,
    top_only: int | float | None = None,
    as_class: Literal[False],
) -> tuple[str, ...]:
    ...


@overload
def get_agents(
    version: str | int,
    *,
    track: str = "any",
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
    bird_only: bool = False,
    top_only: int | float | None = None,
    as_class: Literal[True],
) -> tuple[Agent, ...]:
    ...


def get_agents(
    version: str | int,
    *,
    track: str = "any",
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
    bird_only: bool = False,
    top_only: int | float | None = None,
    as_class: bool = True,
) -> tuple[Agent | str, ...]:
    """
    Gets agent classes/full class names for a version which can either be a competition year (int) or "contrib".

    Args:
        version: Either a competition year (2019, 2020, 2021, ....) or the following special values:

                 - "contrib" for agents contributed directly to the repository not through ANAC's SCML Competition
                 - "all"/"any" for all agents

        track: The track (all, any, collusion, std, sabotage[only for 2019], oneshot [starting 2021]).
        qualified_only: If true, only agents that were submitted to SCML and ran in the qualifications round will be
                        returned
        finalists_only: If true, only agents that were submitted to SCML and passed qualifications will be
                        returned
        winners_only: If true, only winners of SCML (the given version) will be returned.
        bird_only: If true, only winners of the BIRD Innovation Award (the given version) will be returned.
        top_only: Either a fraction of finalists or the top n finalists with highest scores in the finals of
                  SCML
        as_class: If true, the agent classes will be returned otherwise their full class names.
    """
    if version in ("all", "any"):
        results = []
        for v in (2019, 2020, 2021, 2022, "contrib"):
            results += get_agents(
                v,
                track=track,
                qualified_only=qualified_only,
                finalists_only=finalists_only,
                winners_only=winners_only,
                bird_only=bird_only,
                top_only=top_only,
                as_class=as_class,
            )
        return tuple(results)
    classes = tuple()
    track = track.lower()
    if isinstance(version, int) and version == 2019:
        if track in ("any", "all") and not winners_only:
            classes = (
                scml2019.FJ2FactoryManager,
                scml2019.RaptFactoryManager,
                scml2019.InsuranceFraudFactoryManager,
                scml2019.SAHAFactoryManager,
                scml2019.CheapBuyerFactoryManager,
                scml2019.NVMFactoryManager,
                scml2019.Monopoly,
                scml2019.PenaltySabotageFactoryManager,
            )
        if track in ("std", "standard", "collusion") and not winners_only:
            classes = (
                scml2019.FJ2FactoryManager,
                scml2019.RaptFactoryManager,
                scml2019.InsuranceFraudFactoryManager,
                scml2019.SAHAFactoryManager,
                scml2019.CheapBuyerFactoryManager,
                scml2019.NVMFactoryManager,
            )
        if track == "sabotage" and not winners_only:
            # track is sabotage. Monopoly and PSFM (to be added)
            classes = (
                scml2019.Monopoly,
                scml2019.PenaltySabotageFactoryManager,
            )
        elif track in ("std", "standard") and winners_only:
            classes = (
                scml2019.InsuranceFraudFactoryManager,
                scml2019.NVMFactoryManager,
                scml2019.SAHAFactoryManager,
            )
        elif track in ("any", "all") and winners_only:
            classes = (
                scml2019.InsuranceFraudFactoryManager,
                scml2019.NVMFactoryManager,
                scml2019.SAHAFactoryManager,
                scml2019.FJ2FactoryManager,
            )
        elif track in ("coll", "collusion") and winners_only:
            classes = (
                scml2019.InsuranceFraudFactoryManager,
                scml2019.NVMFactoryManager,
                scml2019.FJ2FactoryManager,
            )
        elif track in ("sabotage",) and winners_only:
            classes = tuple()
    elif isinstance(version, int) and version == 2020:
        if track in ("std", "standard") and finalists_only:
            classes = tuple(
                sum(
                    (
                        [f"{_.__name__}.{a}" for a in _.__all__]
                        for _ in (
                            scml2020.team_may,
                            scml2020.team_22,
                            scml2020.team_25,
                            scml2020.team_15,
                            scml2020.a_sengupta,
                            scml2020.monty_hall,
                            scml2020.team_17,
                            scml2020.team_10,
                            scml2020.threadfield,
                            scml2020.team_20,
                            scml2020.biu_th,
                            scml2020.team_32,
                        )
                    ),
                    [],
                )
            )
        elif track in ("coll", "collusion") and finalists_only:
            classes = tuple(
                sum(
                    (
                        [f"{_.__name__}.{a}" for a in _.__all__]
                        for _ in (
                            scml2020.team_17,
                            scml2020.team_may,
                            scml2020.team_25,
                            scml2020.team_15,
                            scml2020.a_sengupta,
                            scml2020.team_20,
                        )
                    ),
                    [],
                )
            )
        elif (
            track in ("any", "all", "std", "standard", "collusion") and not winners_only
        ):
            classes = tuple(
                sum(
                    (
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
                    ),
                    [],
                )
            )
        elif track in ("std", "standard") and winners_only:
            classes = tuple(
                sum(
                    (
                        [f"{_.__name__}.{a}" for a in _.__all__]
                        for _ in (
                            scml2020.team_15,
                            scml2020.team_25,
                        )
                    ),
                    [],
                )
            )
        elif track in ("any", "all") and winners_only:
            classes = tuple(
                sum(
                    (
                        [f"{_.__name__}.{a}" for a in _.__all__]
                        for _ in (
                            scml2020.team_15,
                            scml2020.team_may,
                            scml2020.team_25,
                            scml2020.a_sengupta,
                        )
                    ),
                    [],
                )
            )
        elif track in ("coll", "collusion") and winners_only:
            classes = tuple(
                sum(
                    (
                        [f"{_.__name__}.{a}" for a in _.__all__]
                        for _ in (
                            scml2020.team_may,
                            scml2020.a_sengupta,
                        )
                    ),
                    [],
                )
            )
    elif isinstance(version, int) and version == 2021:
        if bird_only:
            classes = (scml2021.oneshot.team_corleone.MAIN_AGENT,)
        elif track in ("std", "standard") and winners_only:
            classes = (
                (scml2021.standard.team_may.MAIN_AGENT,),
                (scml2021.standard.bossagent.MAIN_AGENT,),
                (scml2021.standard.wabisabikoalas.MAIN_AGENT,),
            )
        elif track in ("coll", "collusion") and winners_only:
            classes = (
                (scml2021.standard.team_may.MAIN_AGENT,),
                (scml2021.standard.bossagent.MAIN_AGENT,),
            )
        elif track in ("one", "oneshot") and winners_only:
            classes = (
                (scml2021.oneshot.team_86.MAIN_AGENT,),
                (scml2021.oneshot.team_73.MAIN_AGENT,),
                (
                    scml2021.oneshot.team_50.MAIN_AGENT,
                    scml2021.oneshot.team_62.MAIN_AGENT,
                ),
            )
        elif track in ("any", "all") and winners_only:
            classes = (
                scml2021.standard.team_may.MAIN_AGENT,
                scml2021.standard.bossagent.MAIN_AGENT,
                scml2021.standard.wabisabikoalas.MAIN_AGENT,
                scml2021.oneshot.team_86.MAIN_AGENT,
                scml2021.oneshot.team_73.MAIN_AGENT,
                scml2021.oneshot.team_50.MAIN_AGENT,
                scml2021.oneshot.team_62.MAIN_AGENT,
            )
        elif track in ("std", "standard") and finalists_only:
            classes = (
                scml2021.standard.team_may.MAIN_AGENT,
                scml2021.standard.bossagent.MAIN_AGENT,
                scml2021.standard.wabisabikoalas.MAIN_AGENT,
                scml2021.standard.team_mediocre.MAIN_AGENT,
                scml2021.standard.team_53.MAIN_AGENT,
            )
        elif track in ("coll", "collusion") and finalists_only:
            classes = (
                scml2021.standard.team_may.MAIN_AGENT,
                scml2021.standard.bossagent.MAIN_AGENT,
                scml2021.standard.wabisabikoalas.MAIN_AGENT,
                scml2021.standard.team_mediocre.MAIN_AGENT,
                scml2021.standard.team_53.MAIN_AGENT,
            )
        elif track in ("oneshot", "one") and finalists_only:
            classes = (
                scml2021.oneshot.team_86.MAIN_AGENT,
                scml2021.oneshot.team_50.MAIN_AGENT,
                scml2021.oneshot.team_73.MAIN_AGENT,
                scml2021.oneshot.team_62.MAIN_AGENT,
                scml2021.oneshot.team_54.MAIN_AGENT,
                scml2021.oneshot.staghunter.MAIN_AGENT,
                scml2021.oneshot.team_corleone.MAIN_AGENT,
                scml2021.oneshot.team_55.MAIN_AGENT,
            )
        elif track in ("all", "any") and finalists_only:
            classes = (
                scml2021.standard.team_may.MAIN_AGENT,
                scml2021.standard.bossagent.MAIN_AGENT,
                scml2021.standard.wabisabikoalas.MAIN_AGENT,
                scml2021.standard.team_mediocre.MAIN_AGENT,
                scml2021.standard.team_53.MAIN_AGENT,
                scml2021.oneshot.team_86.MAIN_AGENT,
                scml2021.oneshot.team_50.MAIN_AGENT,
                scml2021.oneshot.team_73.MAIN_AGENT,
                scml2021.oneshot.team_62.MAIN_AGENT,
                scml2021.oneshot.team_54.MAIN_AGENT,
                scml2021.oneshot.staghunter.MAIN_AGENT,
                scml2021.oneshot.team_corleone.MAIN_AGENT,
                scml2021.oneshot.team_55.MAIN_AGENT,
            )
        elif track in ("std", "standard") and qualified_only:
            classes = (
                scml2021.standard.bossagent.MAIN_AGENT,
                scml2021.standard.iyibiteam.MAIN_AGENT,
                scml2021.standard.team_41.MAIN_AGENT,
                scml2021.standard.team_44.MAIN_AGENT,
                scml2021.standard.team_45.MAIN_AGENT,
                scml2021.standard.team_46.MAIN_AGENT,
                scml2021.standard.team_49.MAIN_AGENT,
                scml2021.standard.team_53.MAIN_AGENT,
                scml2021.standard.team_67.MAIN_AGENT,
                scml2021.standard.team_78.MAIN_AGENT,
                scml2021.standard.team_82.MAIN_AGENT,
                scml2021.standard.team_91.MAIN_AGENT,
                scml2021.standard.team_may.MAIN_AGENT,
                scml2021.standard.team_mediocre.MAIN_AGENT,
                scml2021.standard.wabisabikoalas.MAIN_AGENT,
            )
        elif track in ("coll", "collusion") and qualified_only:
            classes = (
                scml2021.standard.bossagent.MAIN_AGENT,
                scml2021.standard.iyibiteam.MAIN_AGENT,
                scml2021.standard.team_41.MAIN_AGENT,
                scml2021.standard.team_44.MAIN_AGENT,
                scml2021.standard.team_45.MAIN_AGENT,
                scml2021.standard.team_46.MAIN_AGENT,
                scml2021.standard.team_49.MAIN_AGENT,
                scml2021.standard.team_53.MAIN_AGENT,
                scml2021.standard.team_67.MAIN_AGENT,
                scml2021.standard.team_78.MAIN_AGENT,
                scml2021.standard.team_82.MAIN_AGENT,
                scml2021.standard.team_91.MAIN_AGENT,
                scml2021.standard.team_may.MAIN_AGENT,
                scml2021.standard.team_mediocre.MAIN_AGENT,
                scml2021.standard.wabisabikoalas.MAIN_AGENT,
            )
        elif track in ("oneshot", "one") and qualified_only:
            classes = (
                scml2021.oneshot.staghunter.MAIN_AGENT,
                scml2021.oneshot.team_50.MAIN_AGENT,
                scml2021.oneshot.team_51.MAIN_AGENT,
                scml2021.oneshot.team_54.MAIN_AGENT,
                scml2021.oneshot.team_55.MAIN_AGENT,
                scml2021.oneshot.team_62.MAIN_AGENT,
                scml2021.oneshot.team_72.MAIN_AGENT,
                scml2021.oneshot.team_73.MAIN_AGENT,
                scml2021.oneshot.team_86.MAIN_AGENT,
                scml2021.oneshot.team_90.MAIN_AGENT,
                scml2021.oneshot.team_corleone.MAIN_AGENT,
            )
        elif track in ("all", "any") and qualified_only:
            classes = (
                scml2021.standard.bossagent.MAIN_AGENT,
                scml2021.standard.iyibiteam.MAIN_AGENT,
                scml2021.standard.team_41.MAIN_AGENT,
                scml2021.standard.team_44.MAIN_AGENT,
                scml2021.standard.team_45.MAIN_AGENT,
                scml2021.standard.team_46.MAIN_AGENT,
                scml2021.standard.team_49.MAIN_AGENT,
                scml2021.standard.team_53.MAIN_AGENT,
                scml2021.standard.team_67.MAIN_AGENT,
                scml2021.standard.team_78.MAIN_AGENT,
                scml2021.standard.team_82.MAIN_AGENT,
                scml2021.standard.team_91.MAIN_AGENT,
                scml2021.standard.team_may.MAIN_AGENT,
                scml2021.standard.team_mediocre.MAIN_AGENT,
                scml2021.standard.wabisabikoalas.MAIN_AGENT,
                scml2021.oneshot.staghunter.MAIN_AGENT,
                scml2021.oneshot.team_50.MAIN_AGENT,
                scml2021.oneshot.team_51.MAIN_AGENT,
                scml2021.oneshot.team_54.MAIN_AGENT,
                scml2021.oneshot.team_55.MAIN_AGENT,
                scml2021.oneshot.team_62.MAIN_AGENT,
                scml2021.oneshot.team_72.MAIN_AGENT,
                scml2021.oneshot.team_73.MAIN_AGENT,
                scml2021.oneshot.team_86.MAIN_AGENT,
                scml2021.oneshot.team_90.MAIN_AGENT,
                scml2021.oneshot.team_corleone.MAIN_AGENT,
            )
        elif track in ("std", "coll", "standard", "collusion"):
            classes = tuple(
                sum(
                    (
                        [
                            eval(f"scml2021.standard.{_}.{a}")
                            for a in eval(f"scml2021.standard.{_}").__all__
                        ]
                        for _ in dir(scml2021.standard)
                        if ismodule(eval(f"scml2021.standard.{_}"))
                    ),
                    [],
                )
            )
        elif track in ("one", "oneshot"):
            classes = tuple(
                sum(
                    (
                        [
                            eval(f"scml2021.oneshot.{_}.{a}")
                            for a in eval(f"scml2021.oneshot.{_}").__all__
                        ]
                        for _ in dir(scml2021.oneshot)
                        if ismodule(eval(f"scml2021.oneshot.{_}"))
                    ),
                    [],
                )
            )
        elif track in ("any", "all"):
            classes = tuple(
                sum(
                    [
                        [
                            eval(f"scml2021.standard.{_}.{a}")
                            for a in eval(f"scml2021.standard.{_}").__all__
                        ]
                        for _ in dir(scml2021.standard)
                        if ismodule(eval(f"scml2021.standard.{_}"))
                    ]
                    + [
                        [
                            eval(f"scml2021.oneshot.{_}.{a}")
                            for a in eval(f"scml2021.oneshot.{_}").__all__
                        ]
                        for _ in dir(scml2021.oneshot)
                        if ismodule(eval(f"scml2021.oneshot.{_}"))
                    ],
                    [],
                )
            )
    elif isinstance(version, int) and version == 2022:
        if bird_only:
            classes = tuple()
        elif track in ("std", "standard") and winners_only:
            classes = (
                (scml2022.standard.team_137.MAIN_AGENT,),
                (scml2022.standard.team_may.MAIN_AGENT,),
                (scml2022.standard.wabisabikoalas.MAIN_AGENT,),
            )
        elif track in ("coll", "collusion") and winners_only:
            classes = ((scml2022.collusion.team_may.MAIN_AGENT,),)
        elif track in ("one", "oneshot") and winners_only:
            classes = (
                (scml2022.oneshot.team_134.MAIN_AGENT,),
                (scml2022.oneshot.team_102.MAIN_AGENT,),
                (scml2022.oneshot.team_126.MAIN_AGENT,),
            )
        elif track in ("any", "all") and winners_only:
            classes = (
                scml2022.standard.team_137.MAIN_AGENT,
                scml2022.standard.team_may.MAIN_AGENT,
                scml2022.standard.wabisabikoalas.MAIN_AGENT,
                scml2022.collusion.team_may.MAIN_AGENT,
                scml2022.oneshot.team_134.MAIN_AGENT,
                scml2022.oneshot.team_102.MAIN_AGENT,
                scml2022.oneshot.team_126.MAIN_AGENT,
            )
        elif track in ("std", "standard") and finalists_only:
            classes = (
                scml2022.standard.team_137.MAIN_AGENT,
                scml2022.standard.team_may.MAIN_AGENT,
                scml2022.standard.wabisabikoalas.MAIN_AGENT,
                scml2022.standard.team_100.MAIN_AGENT,
                scml2022.standard.bossagent.MAIN_AGENT,
            )
        elif track in ("coll", "collusion") and finalists_only:
            classes = (
                scml2022.collusion.team_may.MAIN_AGENT,
                scml2022.collusion.bossagent.MAIN_AGENT,
            )
        elif track in ("oneshot", "one") and finalists_only:
            classes = (
                scml2022.oneshot.team_134.MAIN_AGENT,
                scml2022.oneshot.team_102.MAIN_AGENT,
                scml2022.oneshot.team_126.MAIN_AGENT,
                scml2022.oneshot.team_106.MAIN_AGENT,
                scml2022.oneshot.team_107.MAIN_AGENT,
                scml2022.oneshot.team_124.MAIN_AGENT,
                scml2022.oneshot.team_131.MAIN_AGENT,
                scml2022.oneshot.team_123.MAIN_AGENT,
            )
        elif track in ("all", "any") and finalists_only:
            classes = (
                scml2022.standard.team_137.MAIN_AGENT,
                scml2022.standard.team_may.MAIN_AGENT,
                scml2022.standard.wabisabikoalas.MAIN_AGENT,
                scml2022.standard.team_100.MAIN_AGENT,
                scml2022.standard.bossagent.MAIN_AGENT,
                scml2022.oneshot.team_134.MAIN_AGENT,
                scml2022.oneshot.team_102.MAIN_AGENT,
                scml2022.oneshot.team_126.MAIN_AGENT,
                scml2022.oneshot.team_106.MAIN_AGENT,
                scml2022.oneshot.team_107.MAIN_AGENT,
                scml2022.oneshot.team_124.MAIN_AGENT,
                scml2022.oneshot.team_131.MAIN_AGENT,
                scml2022.oneshot.team_123.MAIN_AGENT,
                scml2022.collusion.team_may.MAIN_AGENT,
                scml2022.collusion.bossagent.MAIN_AGENT,
            )
        elif track in ("std", "standard") and qualified_only:
            classes = (
                scml2022.standard.team_137.MAIN_AGENT,
                scml2022.standard.team_may.MAIN_AGENT,
                scml2022.standard.wabisabikoalas.MAIN_AGENT,
                scml2022.standard.team_100.MAIN_AGENT,
                scml2022.standard.bossagent.MAIN_AGENT,
                scml2022.standard.team_9.MAIN_AGENT,
                scml2022.standard.team_99.MAIN_AGENT,
            )
        elif track in ("coll", "collusion") and qualified_only:
            classes = (
                scml2022.collusion.team_may.MAIN_AGENT,
                scml2022.collusion.bossagent.MAIN_AGENT,
            )
        elif track in ("oneshot", "one") and qualified_only:
            classes = (
                scml2022.oneshot.team_134.MAIN_AGENT,
                scml2022.oneshot.team_102.MAIN_AGENT,
                scml2022.oneshot.team_126.MAIN_AGENT,
                scml2022.oneshot.team_106.MAIN_AGENT,
                scml2022.oneshot.team_107.MAIN_AGENT,
                scml2022.oneshot.team_124.MAIN_AGENT,
                scml2022.oneshot.team_131.MAIN_AGENT,
                scml2022.oneshot.team_123.MAIN_AGENT,
                scml2022.oneshot.team_94.MAIN_AGENT,
                scml2022.oneshot.team_96.MAIN_AGENT,
                scml2022.oneshot.team_105.MAIN_AGENT,
                scml2022.oneshot.team_103.MAIN_AGENT,
                scml2022.oneshot.team_62.MAIN_AGENT,
            )
        elif track in ("all", "any") and qualified_only:
            classes = (
                scml2022.oneshot.team_134.MAIN_AGENT,
                scml2022.oneshot.team_102.MAIN_AGENT,
                scml2022.oneshot.team_126.MAIN_AGENT,
                scml2022.oneshot.team_106.MAIN_AGENT,
                scml2022.oneshot.team_107.MAIN_AGENT,
                scml2022.oneshot.team_124.MAIN_AGENT,
                scml2022.oneshot.team_131.MAIN_AGENT,
                scml2022.oneshot.team_123.MAIN_AGENT,
                scml2022.oneshot.team_94.MAIN_AGENT,
                scml2022.oneshot.team_96.MAIN_AGENT,
                scml2022.oneshot.team_105.MAIN_AGENT,
                scml2022.oneshot.team_103.MAIN_AGENT,
                scml2022.oneshot.team_62.MAIN_AGENT,
                scml2021.oneshot.team_86.MAIN_AGENT,
                scml2021.oneshot.team_50.MAIN_AGENT,
                scml2022.standard.team_137.MAIN_AGENT,
                scml2022.standard.team_may.MAIN_AGENT,
                scml2022.standard.wabisabikoalas.MAIN_AGENT,
                scml2022.standard.team_100.MAIN_AGENT,
                scml2022.standard.bossagent.MAIN_AGENT,
                scml2022.standard.team_9.MAIN_AGENT,
                scml2022.standard.team_99.MAIN_AGENT,
                scml2021.standard.wabisabikoalas.MAIN_AGENT,
                scml2022.collusion.team_may.MAIN_AGENT,
                scml2022.collusion.bossagent.MAIN_AGENT,
            )
        elif track in ("std", "coll", "standard", "collusion"):
            classes = tuple(
                sum(
                    (
                        [
                            eval(f"scml2022.standard.{_}.{a}")
                            for a in eval(f"scml2022.standard.{_}").__all__
                        ]
                        for _ in dir(scml2022.standard)
                        if ismodule(eval(f"scml2022.standard.{_}"))
                    ),
                    [],
                )
            )
        elif track in ("one", "oneshot"):
            classes = tuple(
                sum(
                    (
                        [
                            eval(f"scml2022.oneshot.{_}.{a}")
                            for a in eval(f"scml2022.oneshot.{_}").__all__
                        ]
                        for _ in dir(scml2022.oneshot)
                        if ismodule(eval(f"scml2022.oneshot.{_}"))
                    ),
                    [],
                )
            )
        elif track in ("any", "all"):
            classes = tuple(
                sum(
                    [
                        [
                            eval(f"scml2022.standard.{_}.{a}")
                            for a in eval(f"scml2022.standard.{_}").__all__
                        ]
                        for _ in dir(scml2022.standard)
                        if ismodule(eval(f"scml2022.standard.{_}"))
                    ]
                    + [
                        [
                            eval(f"scml2022.collusion.{_}.{a}")
                            for a in eval(f"scml2022.collusion.{_}").__all__
                        ]
                        for _ in dir(scml2022.collusion)
                        if ismodule(eval(f"scml2022.collusion.{_}"))
                    ]
                    + [
                        [
                            eval(f"scml2022.oneshot.{_}.{a}")
                            for a in eval(f"scml2022.oneshot.{_}").__all__
                        ]
                        for _ in dir(scml2022.oneshot)
                        if ismodule(eval(f"scml2022.oneshot.{_}"))
                    ],
                    [],
                )
            )
    elif isinstance(version, str) and version == "contrib":
        classes = tuple()
    else:
        raise ValueError(
            f"The version {version} is unknown. Valid versions are 2019, 2020 (as ints), 'contrib' as a string"
        )
    if as_class:
        classes = tuple(get_class(_) for _ in classes)
    else:
        classes = tuple(get_full_type_name(_) for _ in classes)

    if top_only is not None:
        n = int(top_only) if top_only >= 1 else (top_only * len(classes))
        if n > 0:
            return classes[: min(n, len(classes))]

    return classes
