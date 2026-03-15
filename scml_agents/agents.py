from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from negmas.helpers import get_class, get_full_type_name
from negmas.situated import Agent

if TYPE_CHECKING:
    pass

__all__ = ["get_agents", "FAILING_AGENTS"]

# =============================================================================
# Hardcoded MAIN_AGENT mappings to avoid importing heavy modules
# Format: "year.track.team" -> "full.class.path"
# =============================================================================

# SCML 2019 agents (no MAIN_AGENT pattern, direct classes)
_AGENTS_2019 = {
    "FJ2FactoryManager": "scml_agents.scml2019.fj2.FJ2FactoryManager",
    "RaptFactoryManager": "scml_agents.scml2019.rapt_fm.RaptFactoryManager",
    "InsuranceFraudFactoryManager": "scml_agents.scml2019.iffm.InsuranceFraudFactoryManager",
    "SAHAFactoryManager": "scml_agents.scml2019.saha.SAHAFactoryManager",
    "CheapBuyerFactoryManager": "scml_agents.scml2019.cheap_buyer.cheapbuyer.CheapBuyerFactoryManager",
    "NVMFactoryManager": "scml_agents.scml2019.nvm.nmv_agent.NVMFactoryManager",
    "Monopoly": "scml_agents.scml2019.monopoly.Monopoly",
    "PenaltySabotageFactoryManager": "scml_agents.scml2019.psfm.PenaltySabotageFactoryManager",
}

# SCML 2020 agents - teams with __all__ exports
_AGENTS_2020_TEAMS = {
    "team_may": "scml_agents.scml2020.team_may",
    "team_22": "scml_agents.scml2020.team_22",
    "team_25": "scml_agents.scml2020.team_25",
    "team_15": "scml_agents.scml2020.team_15",
    "bargent": "scml_agents.scml2020.bargent",
    "agent0x111": "scml_agents.scml2020.agent0x111",
    "a_sengupta": "scml_agents.scml2020.a_sengupta",
    "past_frauds": "scml_agents.scml2020.past_frauds",
    "monty_hall": "scml_agents.scml2020.monty_hall",
    "team_19": "scml_agents.scml2020.team_19",
    "team_17": "scml_agents.scml2020.team_17",
    "team_10": "scml_agents.scml2020.team_10",
    "threadfield": "scml_agents.scml2020.threadfield",
    "team_29": "scml_agents.scml2020.team_29",
    "team_20": "scml_agents.scml2020.team_20",
    "team_27": "scml_agents.scml2020.team_27",
    "team_18": "scml_agents.scml2020.team_18",
    "biu_th": "scml_agents.scml2020.biu_th",
    "team_32": "scml_agents.scml2020.team_32",
}

# SCML 2021+ MAIN_AGENT mappings
# Format: (year, track, team) -> class_name
_MAIN_AGENTS: dict[tuple[int, str, str], str] = {
    # 2021 Standard
    (
        2021,
        "standard",
        "bossagent",
    ): "scml_agents.scml2021.standard.bossagent.CharliesAgent",
    (
        2021,
        "standard",
        "iyibiteam",
    ): "scml_agents.scml2021.standard.iyibiteam.IYIBIAgent",
    (2021, "standard", "team_41"): "scml_agents.scml2021.standard.team_41.SorceryAgent",
    (2021, "standard", "team_44"): "scml_agents.scml2021.standard.team_44.Agent68",
    (2021, "standard", "team_45"): "scml_agents.scml2021.standard.team_45.StingyAgent",
    (2021, "standard", "team_46"): "scml_agents.scml2021.standard.team_46.SolidAgent",
    (2021, "standard", "team_49"): "scml_agents.scml2021.standard.team_49.E3BIUagent",
    (
        2021,
        "standard",
        "team_53",
    ): "scml_agents.scml2021.standard.team_53.MyPaibiuAgent",
    (
        2021,
        "standard",
        "team_67",
    ): "scml_agents.scml2021.standard.team_67.PolymorphicAgent",
    (2021, "standard", "team_78"): "scml_agents.scml2021.standard.team_78.YIYAgent",
    (
        2021,
        "standard",
        "team_82",
    ): "scml_agents.scml2021.standard.team_82.PerryTheAgent",
    (2021, "standard", "team_91"): "scml_agents.scml2021.standard.team_91.BlueWolf",
    (2021, "standard", "team_may"): "scml_agents.scml2021.standard.team_may.M4",
    (
        2021,
        "standard",
        "team_mediocre",
    ): "scml_agents.scml2021.standard.team_mediocre.Mediocre",
    (
        2021,
        "standard",
        "wabisabikoalas",
    ): "scml_agents.scml2021.standard.wabisabikoalas.ArtisanKangaroo",
    # 2021 Oneshot
    (
        2021,
        "oneshot",
        "staghunter",
    ): "scml_agents.scml2021.oneshot.staghunter.StagHunterV7",
    (2021, "oneshot", "team_50"): "scml_agents.scml2021.oneshot.team_50.Agent74",
    (2021, "oneshot", "team_51"): "scml_agents.scml2021.oneshot.team_51.QlAgent",
    (2021, "oneshot", "team_54"): "scml_agents.scml2021.oneshot.team_54.TheSopranos78",
    (2021, "oneshot", "team_55"): "scml_agents.scml2021.oneshot.team_55.Zilberan",
    (2021, "oneshot", "team_61"): "scml_agents.scml2021.oneshot.team_61.BondAgent",
    (
        2021,
        "oneshot",
        "team_62",
    ): "scml_agents.scml2021.oneshot.team_62.UcOneshotAgent3_4",
    (2021, "oneshot", "team_72"): "scml_agents.scml2021.oneshot.team_72.Agent97",
    (2021, "oneshot", "team_73"): "scml_agents.scml2021.oneshot.team_73.Gentle",
    (2021, "oneshot", "team_86"): "scml_agents.scml2021.oneshot.team_86.Agent112",
    (2021, "oneshot", "team_90"): "scml_agents.scml2021.oneshot.team_90.PDPSyncAgent",
    (
        2021,
        "oneshot",
        "team_corleone",
    ): "scml_agents.scml2021.oneshot.team_corleone.GoldfishParetoEmpiricalGodfatherAgent",
    # 2022 Standard
    (
        2022,
        "standard",
        "bossagent",
    ): "scml_agents.scml2022.standard.bossagent.CharliesAgent",
    (2022, "standard", "team_9"): "scml_agents.scml2022.standard.team_9.SalesAgent",
    (2022, "standard", "team_99"): "scml_agents.scml2022.standard.team_99.SmartAgent",
    (2022, "standard", "team_100"): "scml_agents.scml2022.standard.team_100.SkyAgent",
    (2022, "standard", "team_137"): "scml_agents.scml2022.standard.team_137.Lobster",
    (2022, "standard", "team_may"): "scml_agents.scml2022.standard.team_may.M5",
    (
        2022,
        "standard",
        "wabisabikoalas",
    ): "scml_agents.scml2022.standard.wabisabikoalas.ArtisanKangaroo",
    # 2022 Collusion
    (
        2022,
        "collusion",
        "bossagent",
    ): "scml_agents.scml2022.collusion.bossagent.CharliesAgentCollusion",
    (
        2022,
        "collusion",
        "team_may",
    ): "scml_agents.scml2022.collusion.team_may.M5Collusion",
    # 2022 Oneshot
    (
        2022,
        "oneshot",
        "team_62",
    ): "scml_agents.scml2022.oneshot.team_62.UcOneshotAgent3_4",
    (
        2022,
        "oneshot",
        "team_94",
    ): "scml_agents.scml2022.oneshot.team_94.AdaptiveQlAgent",
    (2022, "oneshot", "team_96"): "scml_agents.scml2022.oneshot.team_96.Agent125",
    (2022, "oneshot", "team_102"): "scml_agents.scml2022.oneshot.team_102.GentleS",
    (
        2022,
        "oneshot",
        "team_103",
    ): "scml_agents.scml2022.oneshot.team_103.MMMPersonalized",
    (
        2022,
        "oneshot",
        "team_105",
    ): "scml_agents.scml2022.oneshot.team_105.AdaptivePercentile",
    (2022, "oneshot", "team_106"): "scml_agents.scml2022.oneshot.team_106.AdamAgent",
    (2022, "oneshot", "team_107"): "scml_agents.scml2022.oneshot.team_107.EVEAgent",
    (2022, "oneshot", "team_123"): "scml_agents.scml2022.oneshot.team_123.Neko",
    (
        2022,
        "oneshot",
        "team_124",
    ): "scml_agents.scml2022.oneshot.team_124.LearningAdaptiveAgent",
    (2022, "oneshot", "team_126"): "scml_agents.scml2022.oneshot.team_126.AgentSAS",
    (2022, "oneshot", "team_131"): "scml_agents.scml2022.oneshot.team_131.AgentRM",
    (2022, "oneshot", "team_134"): "scml_agents.scml2022.oneshot.team_134.PatientAgent",
    # 2023 Standard
    (2023, "standard", "team_140"): "scml_agents.scml2023.standard.team_140.AgentVSC",
    (2023, "standard", "team_150"): "scml_agents.scml2023.standard.team_150.AgentSDH",
    # 2023 Collusion
    (2023, "collusion", "team_140"): "scml_agents.scml2023.collusion.team_140.AgentVSC",
    (2023, "collusion", "team_150"): "scml_agents.scml2023.collusion.team_150.AgentSDH",
    # 2023 Oneshot
    (2023, "oneshot", "team_102"): "scml_agents.scml2023.oneshot.team_102.RLIndAgent",
    (2023, "oneshot", "team_123"): "scml_agents.scml2023.oneshot.team_123.AgentNeko23",
    (2023, "oneshot", "team_126"): "scml_agents.scml2023.oneshot.team_126.AgentSAS",
    (2023, "oneshot", "team_127"): "scml_agents.scml2023.oneshot.team_127.PHLA",
    (
        2023,
        "oneshot",
        "team_134",
    ): "scml_agents.scml2023.oneshot.team_134.MatchingAgent",
    (2023, "oneshot", "team_139"): "scml_agents.scml2023.oneshot.team_139.TwoOneFive",
    (2023, "oneshot", "team_143"): "scml_agents.scml2023.oneshot.team_143.KanbeAgent",
    (2023, "oneshot", "team_144"): "scml_agents.scml2023.oneshot.team_144.CCAgent",
    (2023, "oneshot", "team_145"): "scml_agents.scml2023.oneshot.team_145.ForestAgent",
    (
        2023,
        "oneshot",
        "team_148",
    ): "scml_agents.scml2023.oneshot.team_148.AgentVSCforOneShot",
    (2023, "oneshot", "team_149"): "scml_agents.scml2023.oneshot.team_149.Shochan",
    (2023, "oneshot", "team_151"): "scml_agents.scml2023.oneshot.team_151.NegoAgent",
    (
        2023,
        "oneshot",
        "team_poli_usp",
    ): "scml_agents.scml2023.oneshot.team_poli_usp.QuantityOrientedAgent",
    # 2024 Standard
    (2024, "standard", "coyoteteam"): "scml_agents.scml2024.standard.coyoteteam.Group2",
    (2024, "standard", "team_178"): "scml_agents.scml2024.standard.team_178.AX",
    (2024, "standard", "team_181"): "scml_agents.scml2024.standard.team_181.DogAgent",
    (
        2024,
        "standard",
        "team_193",
    ): "scml_agents.scml2024.standard.team_193.MatchingPennies",
    (
        2024,
        "standard",
        "team_atsunaga",
    ): "scml_agents.scml2024.standard.team_atsunaga.S5s",
    (
        2024,
        "standard",
        "team_miyajima_std",
    ): "scml_agents.scml2024.standard.team_miyajima_std.CautiousStdAgent",
    (
        2024,
        "standard",
        "team_penguin",
    ): "scml_agents.scml2024.standard.team_penguin.PenguinAgent",
    (
        2024,
        "standard",
        "teamyuzuru",
    ): "scml_agents.scml2024.standard.teamyuzuru.QuickDecisionAgent",
    # 2024 Oneshot
    (2024, "oneshot", "coyoteteam"): "scml_agents.scml2024.oneshot.coyoteteam.Group2",
    (2024, "oneshot", "ozug4"): "scml_agents.scml2024.oneshot.ozug4.PeakPact",
    (2024, "oneshot", "team_144"): "scml_agents.scml2024.oneshot.team_144.CPDistAgent",
    (2024, "oneshot", "team_164"): "scml_agents.scml2024.oneshot.team_164.RTAgent",
    (
        2024,
        "oneshot",
        "team_171",
    ): "scml_agents.scml2024.oneshot.team_171.DistRedistAgent",
    (2024, "oneshot", "team_172"): "scml_agents.scml2024.oneshot.team_172.FairT4T",
    (
        2024,
        "oneshot",
        "team_193",
    ): "scml_agents.scml2024.oneshot.team_193.MatchingPennies",
    (2024, "oneshot", "team_abc"): "scml_agents.scml2024.oneshot.team_abc.SuzukaAgent",
    (
        2024,
        "oneshot",
        "team_miyajima_oneshot",
    ): "scml_agents.scml2024.oneshot.team_miyajima_oneshot.CautiousOneShotAgent",
    (
        2024,
        "oneshot",
        "teamyuzuru",
    ): "scml_agents.scml2024.oneshot.teamyuzuru.EpsilonGreedyAgent",
    # 2025 Standard
    (
        2025,
        "standard",
        "team_253",
    ): "scml_agents.scml2025.standard.team_253.XenoSotaAgent",
    (
        2025,
        "standard",
        "team_254",
    ): "scml_agents.scml2025.standard.team_254.UltraSuperMiracleSoraFinalAgentZ",
    (
        2025,
        "standard",
        "team_255",
    ): "scml_agents.scml2025.standard.team_255.PonponAgent",
    (
        2025,
        "standard",
        "team_268",
    ): "scml_agents.scml2025.standard.team_268.KATSUDONAgent",
    (
        2025,
        "standard",
        "team_276",
    ): "scml_agents.scml2025.standard.team_276.LitaAgentYR",
    (
        2025,
        "standard",
        "team_280",
    ): "scml_agents.scml2025.standard.team_280.PriceTrendStdAgent",
    (
        2025,
        "standard",
        "team_atsunaga",
    ): "scml_agents.scml2025.standard.team_atsunaga.AS0",
    # 2025 Oneshot
    (2025, "oneshot", "mat"): "scml_agents.scml2025.oneshot.mat.MATAgent",
    (2025, "oneshot", "takafam"): "scml_agents.scml2025.oneshot.takafam.Rchan",
    (2025, "oneshot", "team_276"): "scml_agents.scml2025.oneshot.team_276.LitaAgentYR",
    (
        2025,
        "oneshot",
        "team_283",
    ): "scml_agents.scml2025.oneshot.team_283.AnalysisAgent",
    (
        2025,
        "oneshot",
        "team_284",
    ): "scml_agents.scml2025.oneshot.team_284.AlmostEqualAgent",
    (2025, "oneshot", "team_293"): "scml_agents.scml2025.oneshot.team_293.PPOAgent",
    (
        2025,
        "oneshot",
        "team_star_up",
    ): "scml_agents.scml2025.oneshot.team_star_up.HoriYamaAgent",
    (
        2025,
        "oneshot",
        "team_ukku",
    ): "scml_agents.scml2025.oneshot.team_ukku.DistRedistAgent",
    (
        2025,
        "oneshot",
        "teamyuzuru",
    ): "scml_agents.scml2025.oneshot.teamyuzuru.CostAverseAgent",
}

# Teams that export __all__ (for "default" track queries that iterate all teams)
_TEAMS_WITH_ALL: dict[tuple[int, str], list[str]] = {
    # 2021
    (2021, "standard"): [
        "bossagent",
        "iyibiteam",
        "team_41",
        "team_44",
        "team_45",
        "team_46",
        "team_49",
        "team_53",
        "team_67",
        "team_78",
        "team_82",
        "team_91",
        "team_may",
        "team_mediocre",
        "wabisabikoalas",
    ],
    (2021, "oneshot"): [
        "staghunter",
        "team_50",
        "team_51",
        "team_54",
        "team_55",
        "team_61",
        "team_62",
        "team_72",
        "team_73",
        "team_86",
        "team_90",
        "team_corleone",
    ],
    # 2022
    (2022, "standard"): [
        "bossagent",
        "team_9",
        "team_99",
        "team_100",
        "team_137",
        "team_may",
        "wabisabikoalas",
    ],
    (2022, "collusion"): ["bossagent", "team_may"],
    (2022, "oneshot"): [
        "team_62",
        "team_94",
        "team_96",
        "team_102",
        "team_103",
        "team_105",
        "team_106",
        "team_107",
        "team_123",
        "team_124",
        "team_126",
        "team_131",
        "team_134",
    ],
    # 2023
    (2023, "standard"): ["team_140", "team_150"],
    (2023, "collusion"): ["team_140", "team_150"],
    (2023, "oneshot"): [
        "team_102",
        "team_123",
        "team_126",
        "team_127",
        "team_134",
        "team_139",
        "team_143",
        "team_144",
        "team_145",
        "team_148",
        "team_149",
        "team_151",
        "team_poli_usp",
    ],
    # 2024
    (2024, "standard"): [
        "coyoteteam",
        "team_178",
        "team_181",
        "team_193",
        "team_atsunaga",
        "team_miyajima_std",
        "team_penguin",
        "teamyuzuru",
    ],
    (2024, "oneshot"): [
        "coyoteteam",
        "ozug4",
        "team_144",
        "team_164",
        "team_171",
        "team_172",
        "team_193",
        "team_abc",
        "team_miyajima_oneshot",
        "teamyuzuru",
    ],
    # 2025
    (2025, "standard"): [
        "team_253",
        "team_254",
        "team_255",
        "team_268",
        "team_276",
        "team_280",
        "team_atsunaga",
    ],
    (2025, "oneshot"): [
        "mat",
        "takafam",
        "team_276",
        "team_283",
        "team_284",
        "team_293",
        "team_star_up",
        "team_ukku",
        "teamyuzuru",
    ],
}


def _agent(year: int, track: str, team: str) -> str:
    """Get the full class path for a MAIN_AGENT."""
    return _MAIN_AGENTS[(year, track, team)]


# Hardcoded FAILING_AGENTS to avoid imports
FAILING_AGENTS: dict[str, str] = {
    "scml_agents.scml2021.standard.team_78.YIYAgent": "Needs scikit-learn<=1.3.* and is tested on python 3.10 only",
    "scml_agents.scml2021.oneshot.team_51.QlAgent": "Needs scikit-learn<=1.3.* and is tested on python 3.10 only",
    "scml_agents.scml2022.oneshot.team_94.AdaptiveQlAgent": "Needs scikit-learn<=1.3.* and is tested on python 3.10 only",
}


@overload
def get_agents(  # type: ignore
    version: str | int,
    *,
    track: str = "any",
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
    bird_only: bool = False,
    top_only: int | float | None = None,
    ignore_failing: bool = False,
    as_class: Literal[False] = False,
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
    ignore_failing: bool = False,
    as_class: Literal[True] = True,
) -> tuple[type[Agent], ...]:
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
    ignore_failing: bool = False,
    as_class: bool = True,
) -> tuple[type[Agent] | str, ...]:
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
        ignore_failing: If true, agents known to fail with current dependencies will be excluded.
    """
    if version in ("all", "any"):
        results = []
        for v in (2019, 2020, 2021, 2022, 2023, 2024, 2025, "contrib"):
            results += list(
                get_agents(  # type: ignore
                    v,
                    track=track,
                    qualified_only=qualified_only,
                    finalists_only=finalists_only,
                    winners_only=winners_only,
                    bird_only=bird_only,
                    top_only=top_only,
                    as_class=as_class,  # type: ignore
                    ignore_failing=ignore_failing,
                )
            )
        return tuple(results)

    classes: tuple[str, ...] = tuple()
    track = track.lower()

    if isinstance(version, int) and version == 2019:
        if track in ("any", "all") and not winners_only:
            classes = (
                _AGENTS_2019["FJ2FactoryManager"],
                _AGENTS_2019["RaptFactoryManager"],
                _AGENTS_2019["InsuranceFraudFactoryManager"],
                _AGENTS_2019["SAHAFactoryManager"],
                _AGENTS_2019["CheapBuyerFactoryManager"],
                _AGENTS_2019["NVMFactoryManager"],
                _AGENTS_2019["Monopoly"],
                _AGENTS_2019["PenaltySabotageFactoryManager"],
            )
        if track in ("std", "standard", "collusion") and not winners_only:
            classes = (
                _AGENTS_2019["FJ2FactoryManager"],
                _AGENTS_2019["RaptFactoryManager"],
                _AGENTS_2019["InsuranceFraudFactoryManager"],
                _AGENTS_2019["SAHAFactoryManager"],
                _AGENTS_2019["CheapBuyerFactoryManager"],
                _AGENTS_2019["NVMFactoryManager"],
            )
        if track == "sabotage" and not winners_only:
            classes = (
                _AGENTS_2019["Monopoly"],
                _AGENTS_2019["PenaltySabotageFactoryManager"],
            )
        elif track in ("std", "standard") and winners_only:
            classes = (
                _AGENTS_2019["InsuranceFraudFactoryManager"],
                _AGENTS_2019["NVMFactoryManager"],
                _AGENTS_2019["SAHAFactoryManager"],
            )
        elif track in ("any", "all") and winners_only:
            classes = (
                _AGENTS_2019["InsuranceFraudFactoryManager"],
                _AGENTS_2019["NVMFactoryManager"],
                _AGENTS_2019["SAHAFactoryManager"],
                _AGENTS_2019["FJ2FactoryManager"],
            )
        elif track in ("col", "collusion") and winners_only:
            classes = (
                _AGENTS_2019["InsuranceFraudFactoryManager"],
                _AGENTS_2019["NVMFactoryManager"],
                _AGENTS_2019["FJ2FactoryManager"],
            )
        elif track in ("sabotage",) and winners_only:
            classes = tuple()

    elif isinstance(version, int) and version == 2020:
        # 2020 uses a different pattern - teams export __all__ with agent names
        # We need to import modules to get __all__, but we can cache the results
        import importlib

        def _get_2020_agents(team_names: list[str]) -> tuple[str, ...]:
            result = []
            for team in team_names:
                module = importlib.import_module(f"scml_agents.scml2020.{team}")
                for agent_name in module.__all__:
                    result.append(f"scml_agents.scml2020.{team}.{agent_name}")
            return tuple(result)

        if track in ("std", "standard") and finalists_only:
            classes = _get_2020_agents(
                [
                    "team_may",
                    "team_22",
                    "team_25",
                    "team_15",
                    "a_sengupta",
                    "monty_hall",
                    "team_17",
                    "team_10",
                    "threadfield",
                    "team_20",
                    "biu_th",
                    "team_32",
                ]
            )
        elif track in ("col", "collusion") and finalists_only:
            classes = _get_2020_agents(
                ["team_17", "team_may", "team_25", "team_15", "a_sengupta", "team_20"]
            )
        elif (
            track in ("any", "all", "std", "standard", "collusion") and not winners_only
        ):
            classes = _get_2020_agents(
                [
                    "team_may",
                    "team_22",
                    "team_25",
                    "team_15",
                    "bargent",
                    "agent0x111",
                    "a_sengupta",
                    "past_frauds",
                    "monty_hall",
                    "team_19",
                    "team_17",
                    "team_10",
                    "threadfield",
                    "team_29",
                    "team_20",
                    "team_27",
                    "team_18",
                    "biu_th",
                    "team_32",
                ]
            )
        elif track in ("std", "standard") and winners_only:
            classes = _get_2020_agents(["team_15", "team_25"])
        elif track in ("any", "all") and winners_only:
            classes = _get_2020_agents(["team_15", "team_may", "team_25", "a_sengupta"])
        elif track in ("col", "collusion") and winners_only:
            classes = _get_2020_agents(["team_may", "a_sengupta"])

    elif isinstance(version, int) and version == 2021:
        if bird_only:
            classes = (_agent(2021, "oneshot", "team_corleone"),)
        elif track in ("std", "standard") and winners_only:
            classes = (
                _agent(2021, "standard", "team_may"),
                _agent(2021, "standard", "bossagent"),
                _agent(2021, "standard", "wabisabikoalas"),
            )
        elif track in ("col", "collusion") and winners_only:
            classes = (
                _agent(2021, "standard", "team_may"),
                _agent(2021, "standard", "bossagent"),
            )
        elif track in ("one", "oneshot") and winners_only:
            classes = (
                _agent(2021, "oneshot", "team_86"),
                _agent(2021, "oneshot", "team_73"),
                _agent(2021, "oneshot", "team_50"),
                _agent(2021, "oneshot", "team_62"),
            )
        elif track in ("any", "all") and winners_only:
            classes = (
                _agent(2021, "standard", "team_may"),
                _agent(2021, "standard", "bossagent"),
                _agent(2021, "standard", "wabisabikoalas"),
                _agent(2021, "oneshot", "team_86"),
                _agent(2021, "oneshot", "team_73"),
                _agent(2021, "oneshot", "team_50"),
                _agent(2021, "oneshot", "team_62"),
            )
        elif track in ("std", "standard") and finalists_only:
            classes = (
                _agent(2021, "standard", "team_may"),
                _agent(2021, "standard", "bossagent"),
                _agent(2021, "standard", "wabisabikoalas"),
                _agent(2021, "standard", "team_mediocre"),
                _agent(2021, "standard", "team_53"),
            )
        elif track in ("col", "collusion") and finalists_only:
            classes = (
                _agent(2021, "standard", "team_may"),
                _agent(2021, "standard", "bossagent"),
                _agent(2021, "standard", "wabisabikoalas"),
                _agent(2021, "standard", "team_mediocre"),
                _agent(2021, "standard", "team_53"),
            )
        elif track in ("oneshot", "one") and finalists_only:
            classes = (
                _agent(2021, "oneshot", "team_86"),
                _agent(2021, "oneshot", "team_50"),
                _agent(2021, "oneshot", "team_73"),
                _agent(2021, "oneshot", "team_62"),
                _agent(2021, "oneshot", "team_54"),
                _agent(2021, "oneshot", "staghunter"),
                _agent(2021, "oneshot", "team_corleone"),
                _agent(2021, "oneshot", "team_55"),
            )
        elif track in ("all", "any") and finalists_only:
            classes = (
                _agent(2021, "standard", "team_may"),
                _agent(2021, "standard", "bossagent"),
                _agent(2021, "standard", "wabisabikoalas"),
                _agent(2021, "standard", "team_mediocre"),
                _agent(2021, "standard", "team_53"),
                _agent(2021, "oneshot", "team_86"),
                _agent(2021, "oneshot", "team_50"),
                _agent(2021, "oneshot", "team_73"),
                _agent(2021, "oneshot", "team_62"),
                _agent(2021, "oneshot", "team_54"),
                _agent(2021, "oneshot", "staghunter"),
                _agent(2021, "oneshot", "team_corleone"),
                _agent(2021, "oneshot", "team_55"),
            )
        elif track in ("std", "standard") and qualified_only:
            classes = (
                _agent(2021, "standard", "bossagent"),
                _agent(2021, "standard", "iyibiteam"),
                _agent(2021, "standard", "team_41"),
                _agent(2021, "standard", "team_44"),
                _agent(2021, "standard", "team_45"),
                _agent(2021, "standard", "team_46"),
                _agent(2021, "standard", "team_49"),
                _agent(2021, "standard", "team_53"),
                _agent(2021, "standard", "team_67"),
                _agent(2021, "standard", "team_78"),
                _agent(2021, "standard", "team_82"),
                _agent(2021, "standard", "team_91"),
                _agent(2021, "standard", "team_may"),
                _agent(2021, "standard", "team_mediocre"),
                _agent(2021, "standard", "wabisabikoalas"),
            )
        elif track in ("col", "collusion") and qualified_only:
            classes = (
                _agent(2021, "standard", "bossagent"),
                _agent(2021, "standard", "iyibiteam"),
                _agent(2021, "standard", "team_41"),
                _agent(2021, "standard", "team_44"),
                _agent(2021, "standard", "team_45"),
                _agent(2021, "standard", "team_46"),
                _agent(2021, "standard", "team_49"),
                _agent(2021, "standard", "team_53"),
                _agent(2021, "standard", "team_67"),
                _agent(2021, "standard", "team_78"),
                _agent(2021, "standard", "team_82"),
                _agent(2021, "standard", "team_91"),
                _agent(2021, "standard", "team_may"),
                _agent(2021, "standard", "team_mediocre"),
                _agent(2021, "standard", "wabisabikoalas"),
            )
        elif track in ("oneshot", "one") and qualified_only:
            classes = (
                _agent(2021, "oneshot", "staghunter"),
                _agent(2021, "oneshot", "team_50"),
                _agent(2021, "oneshot", "team_51"),
                _agent(2021, "oneshot", "team_54"),
                _agent(2021, "oneshot", "team_55"),
                _agent(2021, "oneshot", "team_62"),
                _agent(2021, "oneshot", "team_72"),
                _agent(2021, "oneshot", "team_73"),
                _agent(2021, "oneshot", "team_86"),
                _agent(2021, "oneshot", "team_90"),
                _agent(2021, "oneshot", "team_corleone"),
            )
        elif track in ("all", "any") and qualified_only:
            classes = (
                _agent(2021, "standard", "bossagent"),
                _agent(2021, "standard", "iyibiteam"),
                _agent(2021, "standard", "team_41"),
                _agent(2021, "standard", "team_44"),
                _agent(2021, "standard", "team_45"),
                _agent(2021, "standard", "team_46"),
                _agent(2021, "standard", "team_49"),
                _agent(2021, "standard", "team_53"),
                _agent(2021, "standard", "team_67"),
                _agent(2021, "standard", "team_78"),
                _agent(2021, "standard", "team_82"),
                _agent(2021, "standard", "team_91"),
                _agent(2021, "standard", "team_may"),
                _agent(2021, "standard", "team_mediocre"),
                _agent(2021, "standard", "wabisabikoalas"),
                _agent(2021, "oneshot", "staghunter"),
                _agent(2021, "oneshot", "team_50"),
                _agent(2021, "oneshot", "team_51"),
                _agent(2021, "oneshot", "team_54"),
                _agent(2021, "oneshot", "team_55"),
                _agent(2021, "oneshot", "team_62"),
                _agent(2021, "oneshot", "team_72"),
                _agent(2021, "oneshot", "team_73"),
                _agent(2021, "oneshot", "team_86"),
                _agent(2021, "oneshot", "team_90"),
                _agent(2021, "oneshot", "team_corleone"),
            )
        elif track in ("std", "col", "standard", "collusion"):
            classes = tuple(
                _agent(2021, "standard", t) for t in _TEAMS_WITH_ALL[(2021, "standard")]
            )
        elif track in ("one", "oneshot"):
            classes = tuple(
                _agent(2021, "oneshot", t) for t in _TEAMS_WITH_ALL[(2021, "oneshot")]
            )
        elif track in ("any", "all"):
            classes = tuple(
                _agent(2021, "standard", t) for t in _TEAMS_WITH_ALL[(2021, "standard")]
            ) + tuple(
                _agent(2021, "oneshot", t) for t in _TEAMS_WITH_ALL[(2021, "oneshot")]
            )

    elif isinstance(version, int) and version == 2022:
        if bird_only:
            classes = tuple()
        elif track in ("std", "standard") and winners_only:
            classes = (
                _agent(2022, "standard", "team_137"),
                _agent(2022, "standard", "team_may"),
                _agent(2022, "standard", "wabisabikoalas"),
            )
        elif track in ("col", "collusion") and winners_only:
            classes = (_agent(2022, "collusion", "team_may"),)
        elif track in ("one", "oneshot") and winners_only:
            classes = (
                _agent(2022, "oneshot", "team_134"),
                _agent(2022, "oneshot", "team_102"),
                _agent(2022, "oneshot", "team_126"),
            )
        elif track in ("any", "all") and winners_only:
            classes = (
                _agent(2022, "standard", "team_137"),
                _agent(2022, "standard", "team_may"),
                _agent(2022, "standard", "wabisabikoalas"),
                _agent(2022, "collusion", "team_may"),
                _agent(2022, "oneshot", "team_134"),
                _agent(2022, "oneshot", "team_102"),
                _agent(2022, "oneshot", "team_126"),
            )
        elif track in ("std", "standard") and finalists_only:
            classes = (
                _agent(2022, "standard", "team_137"),
                _agent(2022, "standard", "team_may"),
                _agent(2022, "standard", "wabisabikoalas"),
                _agent(2022, "standard", "team_100"),
                _agent(2022, "standard", "bossagent"),
            )
        elif track in ("col", "collusion") and finalists_only:
            classes = (
                _agent(2022, "collusion", "team_may"),
                _agent(2022, "collusion", "bossagent"),
            )
        elif track in ("oneshot", "one") and finalists_only:
            classes = (
                _agent(2022, "oneshot", "team_134"),
                _agent(2022, "oneshot", "team_102"),
                _agent(2022, "oneshot", "team_126"),
                _agent(2022, "oneshot", "team_106"),
                _agent(2022, "oneshot", "team_107"),
                _agent(2022, "oneshot", "team_124"),
                _agent(2022, "oneshot", "team_131"),
                _agent(2022, "oneshot", "team_123"),
            )
        elif track in ("all", "any") and finalists_only:
            classes = (
                _agent(2022, "standard", "team_137"),
                _agent(2022, "standard", "team_may"),
                _agent(2022, "standard", "wabisabikoalas"),
                _agent(2022, "standard", "team_100"),
                _agent(2022, "standard", "bossagent"),
                _agent(2022, "oneshot", "team_134"),
                _agent(2022, "oneshot", "team_102"),
                _agent(2022, "oneshot", "team_126"),
                _agent(2022, "oneshot", "team_106"),
                _agent(2022, "oneshot", "team_107"),
                _agent(2022, "oneshot", "team_124"),
                _agent(2022, "oneshot", "team_131"),
                _agent(2022, "oneshot", "team_123"),
                _agent(2022, "collusion", "team_may"),
                _agent(2022, "collusion", "bossagent"),
            )
        elif track in ("std", "standard") and qualified_only:
            classes = (
                _agent(2022, "standard", "team_137"),
                _agent(2022, "standard", "team_may"),
                _agent(2022, "standard", "wabisabikoalas"),
                _agent(2022, "standard", "team_100"),
                _agent(2022, "standard", "bossagent"),
                _agent(2022, "standard", "team_9"),
                _agent(2022, "standard", "team_99"),
            )
        elif track in ("col", "collusion") and qualified_only:
            classes = (
                _agent(2022, "collusion", "team_may"),
                _agent(2022, "collusion", "bossagent"),
            )
        elif track in ("oneshot", "one") and qualified_only:
            classes = (
                _agent(2022, "oneshot", "team_134"),
                _agent(2022, "oneshot", "team_102"),
                _agent(2022, "oneshot", "team_126"),
                _agent(2022, "oneshot", "team_106"),
                _agent(2022, "oneshot", "team_107"),
                _agent(2022, "oneshot", "team_124"),
                _agent(2022, "oneshot", "team_131"),
                _agent(2022, "oneshot", "team_123"),
                _agent(2022, "oneshot", "team_94"),
                _agent(2022, "oneshot", "team_96"),
                _agent(2022, "oneshot", "team_105"),
                _agent(2022, "oneshot", "team_103"),
                _agent(2022, "oneshot", "team_62"),
            )
        elif track in ("all", "any") and qualified_only:
            classes = (
                _agent(2022, "oneshot", "team_134"),
                _agent(2022, "oneshot", "team_102"),
                _agent(2022, "oneshot", "team_126"),
                _agent(2022, "oneshot", "team_106"),
                _agent(2022, "oneshot", "team_107"),
                _agent(2022, "oneshot", "team_124"),
                _agent(2022, "oneshot", "team_131"),
                _agent(2022, "oneshot", "team_123"),
                _agent(2022, "oneshot", "team_94"),
                _agent(2022, "oneshot", "team_96"),
                _agent(2022, "oneshot", "team_105"),
                _agent(2022, "oneshot", "team_103"),
                _agent(2022, "oneshot", "team_62"),
                _agent(2021, "oneshot", "team_86"),
                _agent(2021, "oneshot", "team_50"),
                _agent(2022, "standard", "team_137"),
                _agent(2022, "standard", "team_may"),
                _agent(2022, "standard", "wabisabikoalas"),
                _agent(2022, "standard", "team_100"),
                _agent(2022, "standard", "bossagent"),
                _agent(2022, "standard", "team_9"),
                _agent(2022, "standard", "team_99"),
                _agent(2021, "standard", "wabisabikoalas"),
                _agent(2022, "collusion", "team_may"),
                _agent(2022, "collusion", "bossagent"),
            )
        elif track in ("std", "col", "standard", "collusion"):
            classes = tuple(
                _agent(2022, "standard", t) for t in _TEAMS_WITH_ALL[(2022, "standard")]
            )
        elif track in ("one", "oneshot"):
            classes = tuple(
                _agent(2022, "oneshot", t) for t in _TEAMS_WITH_ALL[(2022, "oneshot")]
            )
        elif track in ("any", "all"):
            classes = (
                tuple(
                    _agent(2022, "standard", t)
                    for t in _TEAMS_WITH_ALL[(2022, "standard")]
                )
                + tuple(
                    _agent(2022, "collusion", t)
                    for t in _TEAMS_WITH_ALL[(2022, "collusion")]
                )
                + tuple(
                    _agent(2022, "oneshot", t)
                    for t in _TEAMS_WITH_ALL[(2022, "oneshot")]
                )
            )

    elif isinstance(version, int) and version == 2023:
        if bird_only:
            classes = tuple()
        elif track in ("std", "standard") and winners_only:
            classes = (_agent(2023, "standard", "team_150"),)
        elif track in ("col", "collusion") and winners_only:
            classes = (_agent(2023, "collusion", "team_150"),)
        elif track in ("one", "oneshot") and winners_only:
            classes = (
                _agent(2023, "oneshot", "team_poli_usp"),
                _agent(2023, "oneshot", "team_144"),
                _agent(2023, "oneshot", "team_143"),
            )
        elif track in ("any", "all") and winners_only:
            classes = (
                _agent(2023, "oneshot", "team_poli_usp"),
                _agent(2023, "oneshot", "team_144"),
                _agent(2023, "oneshot", "team_143"),
                _agent(2023, "collusion", "team_150"),
            )
        elif track in ("std", "standard") and (finalists_only or qualified_only):
            classes = (
                _agent(2023, "standard", "team_150"),
                _agent(2023, "standard", "team_140"),
            )
        elif track in ("col", "collusion") and (finalists_only or qualified_only):
            classes = (
                _agent(2023, "collusion", "team_150"),
                _agent(2023, "collusion", "team_140"),
            )
        elif track in ("oneshot", "one") and finalists_only:
            classes = (
                _agent(2023, "oneshot", "team_poli_usp"),
                _agent(2023, "oneshot", "team_144"),
                _agent(2023, "oneshot", "team_143"),
                _agent(2023, "oneshot", "team_148"),
                _agent(2023, "oneshot", "team_145"),
                _agent(2023, "oneshot", "team_127"),
                _agent(2023, "oneshot", "team_126"),
                _agent(2023, "oneshot", "team_151"),
            )
        elif track in ("all", "any") and finalists_only:
            classes = (
                _agent(2023, "oneshot", "team_poli_usp"),
                _agent(2023, "oneshot", "team_144"),
                _agent(2023, "oneshot", "team_143"),
                _agent(2023, "oneshot", "team_148"),
                _agent(2023, "oneshot", "team_145"),
                _agent(2023, "oneshot", "team_127"),
                _agent(2023, "oneshot", "team_126"),
                _agent(2023, "oneshot", "team_151"),
                _agent(2023, "collusion", "team_150"),
                _agent(2023, "collusion", "team_140"),
            )
        elif track in ("oneshot", "one") and qualified_only:
            classes = (
                _agent(2023, "oneshot", "team_102"),
                _agent(2023, "oneshot", "team_123"),
                _agent(2023, "oneshot", "team_126"),
                _agent(2023, "oneshot", "team_127"),
                _agent(2023, "oneshot", "team_134"),
                _agent(2023, "oneshot", "team_139"),
                _agent(2023, "oneshot", "team_143"),
                _agent(2023, "oneshot", "team_144"),
                _agent(2023, "oneshot", "team_145"),
                _agent(2023, "oneshot", "team_148"),
                _agent(2023, "oneshot", "team_149"),
                _agent(2023, "oneshot", "team_151"),
                _agent(2023, "oneshot", "team_poli_usp"),
            )
        elif track in ("all", "any") and qualified_only:
            classes = (
                _agent(2023, "oneshot", "team_102"),
                _agent(2023, "oneshot", "team_123"),
                _agent(2023, "oneshot", "team_126"),
                _agent(2023, "oneshot", "team_127"),
                _agent(2023, "oneshot", "team_134"),
                _agent(2023, "oneshot", "team_139"),
                _agent(2023, "oneshot", "team_143"),
                _agent(2023, "oneshot", "team_144"),
                _agent(2023, "oneshot", "team_145"),
                _agent(2023, "oneshot", "team_148"),
                _agent(2023, "oneshot", "team_149"),
                _agent(2023, "oneshot", "team_151"),
                _agent(2023, "oneshot", "team_poli_usp"),
                _agent(2023, "collusion", "team_150"),
                _agent(2023, "collusion", "team_140"),
            )
        elif track in ("std", "col", "standard", "collusion"):
            classes = tuple(
                _agent(2023, "standard", t) for t in _TEAMS_WITH_ALL[(2023, "standard")]
            )
        elif track in ("one", "oneshot"):
            classes = tuple(
                _agent(2023, "oneshot", t) for t in _TEAMS_WITH_ALL[(2023, "oneshot")]
            )
        elif track in ("any", "all"):
            classes = (
                tuple(
                    _agent(2023, "standard", t)
                    for t in _TEAMS_WITH_ALL[(2023, "standard")]
                )
                + tuple(
                    _agent(2023, "collusion", t)
                    for t in _TEAMS_WITH_ALL[(2023, "collusion")]
                )
                + tuple(
                    _agent(2023, "oneshot", t)
                    for t in _TEAMS_WITH_ALL[(2023, "oneshot")]
                )
            )

    elif isinstance(version, int) and version == 2024:
        if bird_only:
            classes = tuple()
        elif track in ("collusion", "col"):
            classes = tuple()
        elif track in ("std", "standard") and winners_only:
            classes = (_agent(2024, "standard", "team_penguin"),)
        elif track in ("one", "oneshot") and winners_only:
            classes = (
                _agent(2024, "oneshot", "team_miyajima_oneshot"),
                _agent(2024, "oneshot", "team_193"),
                _agent(2024, "oneshot", "team_171"),
                _agent(2024, "oneshot", "teamyuzuru"),
            )
        elif track in ("any", "all") and winners_only:
            classes = (
                _agent(2024, "standard", "team_penguin"),
                _agent(2024, "oneshot", "team_miyajima_oneshot"),
                _agent(2024, "oneshot", "team_193"),
                _agent(2024, "oneshot", "team_171"),
                _agent(2024, "oneshot", "teamyuzuru"),
            )
        elif track in ("std", "standard") and (finalists_only or qualified_only):
            classes = (
                _agent(2024, "standard", "team_penguin"),
                _agent(2024, "standard", "team_miyajima_std"),
                _agent(2024, "standard", "team_181"),
                _agent(2024, "standard", "team_178"),
                _agent(2024, "standard", "teamyuzuru"),
            )
        elif track in ("oneshot", "one") and finalists_only:
            classes = (
                _agent(2024, "oneshot", "team_miyajima_oneshot"),
                _agent(2024, "oneshot", "team_193"),
                _agent(2024, "oneshot", "team_171"),
                _agent(2024, "oneshot", "teamyuzuru"),
                _agent(2024, "oneshot", "team_abc"),
            )
        elif track in ("all", "any") and finalists_only:
            classes = (
                _agent(2024, "standard", "team_penguin"),
                _agent(2024, "standard", "team_miyajima_std"),
                _agent(2024, "standard", "team_181"),
                _agent(2024, "standard", "team_178"),
                _agent(2024, "standard", "teamyuzuru"),
            )
        elif track in ("std", "standard"):
            classes = tuple(
                _agent(2024, "standard", t) for t in _TEAMS_WITH_ALL[(2024, "standard")]
            )
        elif track in ("one", "oneshot"):
            classes = tuple(
                _agent(2024, "oneshot", t) for t in _TEAMS_WITH_ALL[(2024, "oneshot")]
            )
        elif track in ("any", "all"):
            classes = tuple(
                _agent(2024, "standard", t) for t in _TEAMS_WITH_ALL[(2024, "standard")]
            ) + tuple(
                _agent(2024, "oneshot", t) for t in _TEAMS_WITH_ALL[(2024, "oneshot")]
            )

    elif isinstance(version, int) and version == 2025:
        if bird_only:
            classes = tuple()
        elif track in ("collusion", "col"):
            classes = tuple()
        elif track in ("std", "standard") and winners_only:
            classes = (
                _agent(2025, "standard", "team_atsunaga"),
                _agent(2025, "standard", "team_253"),
                _agent(2025, "standard", "team_254"),
            )
        elif track in ("one", "oneshot") and winners_only:
            classes = (
                _agent(2025, "oneshot", "teamyuzuru"),
                _agent(2025, "oneshot", "takafam"),
                _agent(2025, "oneshot", "team_284"),
            )
        elif track in ("any", "all") and winners_only:
            classes = (
                _agent(2025, "oneshot", "teamyuzuru"),
                _agent(2025, "oneshot", "takafam"),
                _agent(2025, "oneshot", "team_284"),
                _agent(2025, "standard", "team_atsunaga"),
                _agent(2025, "standard", "team_253"),
                _agent(2025, "standard", "team_254"),
            )
        elif track in ("std", "standard") and finalists_only:
            classes = (
                _agent(2025, "standard", "team_atsunaga"),
                _agent(2025, "standard", "team_253"),
                _agent(2025, "standard", "team_254"),
                _agent(2025, "standard", "team_280"),
            )
        elif track in ("oneshot", "one") and finalists_only:
            classes = (
                _agent(2025, "oneshot", "teamyuzuru"),
                _agent(2025, "oneshot", "takafam"),
                _agent(2025, "oneshot", "team_284"),
                _agent(2025, "oneshot", "team_ukku"),
                _agent(2025, "oneshot", "team_star_up"),
            )
        elif track in ("all", "any") and finalists_only:
            classes = (
                _agent(2025, "oneshot", "teamyuzuru"),
                _agent(2025, "oneshot", "takafam"),
                _agent(2025, "oneshot", "team_284"),
                _agent(2025, "oneshot", "team_ukku"),
                _agent(2025, "oneshot", "team_star_up"),
                _agent(2025, "standard", "team_253"),
                _agent(2025, "standard", "team_254"),
                _agent(2025, "standard", "team_280"),
                _agent(2025, "standard", "team_atsunaga"),
            )
        elif track in ("standard", "std") and qualified_only:
            classes = (
                _agent(2025, "standard", "team_atsunaga"),
                _agent(2025, "standard", "team_253"),
                _agent(2025, "standard", "team_254"),
                _agent(2025, "standard", "team_280"),
                _agent(2025, "standard", "team_255"),
                _agent(2025, "standard", "team_268"),
                _agent(2025, "standard", "team_276"),
            )
        elif track in ("oneshot", "one") and qualified_only:
            classes = (
                _agent(2025, "oneshot", "teamyuzuru"),
                _agent(2025, "oneshot", "takafam"),
                _agent(2025, "oneshot", "team_284"),
                _agent(2025, "oneshot", "team_ukku"),
                _agent(2025, "oneshot", "team_star_up"),
                _agent(2025, "oneshot", "mat"),
                _agent(2025, "oneshot", "team_283"),
                _agent(2025, "oneshot", "team_276"),
                _agent(2025, "oneshot", "team_293"),
            )
        elif track in ("all", "any") and qualified_only:
            classes = (
                _agent(2025, "oneshot", "mat"),
                _agent(2025, "oneshot", "takafam"),
                _agent(2025, "oneshot", "team_276"),
                _agent(2025, "oneshot", "team_283"),
                _agent(2025, "oneshot", "team_284"),
                _agent(2025, "oneshot", "team_293"),
                _agent(2025, "oneshot", "team_star_up"),
                _agent(2025, "oneshot", "team_ukku"),
                _agent(2025, "oneshot", "teamyuzuru"),
                _agent(2025, "standard", "team_253"),
                _agent(2025, "standard", "team_254"),
                _agent(2025, "standard", "team_255"),
                _agent(2025, "standard", "team_268"),
                _agent(2025, "standard", "team_276"),
                _agent(2025, "standard", "team_280"),
                _agent(2025, "standard", "team_atsunaga"),
            )
        elif track in ("std", "standard"):
            classes = tuple(
                _agent(2025, "standard", t) for t in _TEAMS_WITH_ALL[(2025, "standard")]
            )
        elif track in ("one", "oneshot"):
            classes = tuple(
                _agent(2025, "oneshot", t) for t in _TEAMS_WITH_ALL[(2025, "oneshot")]
            )
        elif track in ("any", "all"):
            classes = tuple(
                _agent(2025, "standard", t) for t in _TEAMS_WITH_ALL[(2025, "standard")]
            ) + tuple(
                _agent(2025, "oneshot", t) for t in _TEAMS_WITH_ALL[(2025, "oneshot")]
            )

    elif isinstance(version, str) and version == "contrib":
        classes = tuple()

    else:
        raise ValueError(
            f"The version {version} is unknown. Valid versions are 2019, 2020, 2021, 2022, 2023, 2024, 2025 (as ints), 'contrib' as a string"
        )

    # At this point, classes contains full class path strings
    if as_class:
        classes = tuple(get_class(_) for _ in classes)  # type: ignore
    # else: classes is already tuple of strings

    if ignore_failing:
        classes = tuple(
            [
                _
                for _ in classes
                if (get_full_type_name(_) if as_class else _)
                not in FAILING_AGENTS.keys()
            ]
        )

    if top_only is not None:
        n = int(top_only) if top_only >= 1 else int(top_only * len(classes))
        if n > 0:
            return tuple(classes[: min(n, len(classes))])

    return classes  # type: ignore
