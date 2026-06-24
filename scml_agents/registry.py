"""Generated registry of participants per (year, track).

WRITTEN by scmlweb/python/update_agents_repo.py and friends. Edits
made by hand will be overwritten on the next run -- use
set_finalists.py / set_winners.py to flip the per-entry metadata
flags.

`get_participants(year, track=None, qualified_only=False,
finalists_only=False, winners_only=False)` returns the participants (or
a filtered subset) for a given year / track. `track` is required for
SCML and unused for ANL/HAN. `qualified_only` drops disqualified
entries (a qualified agent is any non-disqualified participant).
"""
from __future__ import annotations

import json
from typing import Optional


# (year, track-or-None) -> list of {class_path, metadata}.
# Stored as JSON (loaded at import) so the booleans/None serialise correctly —
# a raw Python-literal paste would emit JSON `false`/`true`/`null` and break.
_REGISTRY: dict = json.loads(r"""
{
    "2026|oneshot": [
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21022.efr_dist.agent.EFRDistOneShotAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21022",
                "name": "EFRDist"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21041.penalty_avoid.penalty_avoid_agent.PenaltyAvoidAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21041",
                "name": "PenaltyAvoid"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21056.steadysyncagent.agent.SteadySyncAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21056",
                "name": "SteadySyncAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21089.strategy.AssariAsari",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21089",
                "name": "AssariAsari"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21095.isobeagent.IsobeAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21095",
                "name": "IsobeAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21155.time_weighted_agent2.TimeWeightedAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21155",
                "name": "TimeWeightedAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21196.yamashitaagent.YamashitaAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21196",
                "name": "Yamashitaagent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21244.agent03.core.agent.Agent03Agent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21244",
                "name": "agent03"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21484.latice_oneshot_agent.LatticeOneShotAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21484",
                "name": "LatticeOneshotAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21562.group3.Group3",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21562",
                "name": "Group3"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21619.supvelikos.Supvelikos",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21619",
                "name": "supvelikos"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21637.bayesian_agent.BayesianAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21637",
                "name": "BayesianAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21798.hi.oneshot.Oneshot",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21798",
                "name": "heyoneshot"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21799.ysi.Ysi",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21799",
                "name": "Ysi"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21804.oneshot_agent.entry.OneshotOmegaNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21804",
                "name": "CodexAgentOneshot"
            }
        },
        {
            "class_path": "scml_agents.scml2026.oneshot.team_21813.sbdoneshot_agent.SBDOneShot",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21813",
                "name": "SBDOneShot"
            }
        }
    ],
    "2026|standard": [
        {
            "class_path": "scml_agents.scml2026.standard.team_20941.okagent.OkAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": false,
                "has_description": true,
                "team_id": "20941",
                "name": "okagent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_20964.balanced_greedy.balanced_greedy_std_agent.BalancedGreedyStdAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "20964",
                "name": "BalancedGreedyStdAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21021.arion_strategists.arion_agent.ArionAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21021",
                "name": "ArionAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21084.emsel.EmSel",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21084",
                "name": "EmSel"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21127.strategy.ShimijimiShijimi",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21127",
                "name": "ShimijimiShijimi"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21165.age_age_agent.age_age_agent.AgeAgeAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21165",
                "name": "AgeAgeAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21195.agent01std.core.agent.Agent01StdAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21195",
                "name": "agent01std"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21201.gs3_4_submit.GS3",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21201",
                "name": "GS3"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21282.taka_link_agent.TakaLinkAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21282",
                "name": "TakaLinkAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21303.cow.COW",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21303",
                "name": "COW"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21313.sugai_agent.sugai_agent.SugaiAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21313",
                "name": "SugaiAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21559.group4.Group4",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21559",
                "name": "MyAgent17803196313323"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21650.sbd_agent.SBD",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21650",
                "name": "SBD"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21653.horizon_aware_agent.HorizonAwareAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21653",
                "name": "HorizonAwareAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21683.agent_v2.SuperimagentZ",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21683",
                "name": "SuperimagentZ"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21695.rohn.Rohn",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21695",
                "name": "Rohn"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21747.kotaagent.kotaagent.KotaAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21747",
                "name": "KotaAgent"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21800.hi.std.Std",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21800",
                "name": "heystd"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_21803.std_agent.entry.StdOmegaNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21803",
                "name": "CodexAgentStd"
            }
        },
        {
            "class_path": "scml_agents.scml2026.standard.team_22272.supmerkos_v3.SupmerkosV3",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "22272",
                "name": "SupmerkosV3"
            }
        }
    ]
}
""")


def get_participants(
    year: int,
    track: Optional[str] = None,
    *,
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
) -> tuple[str, ...]:
    """Return the dotted Python paths of registered participants."""
    # _REGISTRY keys are the JSON-serialised "year|track" strings (see
    # rewrite_registry); build the same form rather than a tuple.
    key = f"{int(year)}|{track.lower() if track else ''}"
    entries = _REGISTRY.get(key, [])
    out = []
    for e in entries:
        meta = e.get("metadata", {})
        if qualified_only and meta.get("disqualified"):
            continue
        if finalists_only and not meta.get("finalist"):
            continue
        if winners_only and not meta.get("winner"):
            continue
        out.append(e["class_path"])
    return tuple(out)
