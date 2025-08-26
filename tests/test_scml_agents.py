from pytest import mark

from scml_agents import get_agents


@mark.parametrize(
    "version",
    [2019, 2020, 2021, 2022],
)
def test_get_agents_per_year(version):
    from pprint import pprint

    agents = set(get_agents(version, as_class=False))
    pprint(agents)
    if version == 2019:
        assert agents == {
            "scml_agents.scml2019.fj2.FJ2FactoryManager",
            "scml_agents.scml2019.rapt_fm.RaptFactoryManager",
            "scml_agents.scml2019.iffm.InsuranceFraudFactoryManager",
            "scml_agents.scml2019.saha.SAHAFactoryManager",
            "scml_agents.scml2019.cheap_buyer.cheapbuyer.CheapBuyerFactoryManager",
            "scml_agents.scml2019.nvm.nmv_agent.NVMFactoryManager",
            "scml_agents.scml2019.monopoly.Monopoly",
            "scml_agents.scml2019.psfm.PenaltySabotageFactoryManager",
        }
    elif version == 2020:
        assert agents == {
            "scml_agents.scml2020.team_may.MMM",
            "scml_agents.scml2020.team_22.SavingAgent",
            "scml_agents.scml2020.team_25.Agent30",
            "scml_agents.scml2020.team_15.SteadyMgr",
            "scml_agents.scml2020.bargent.BARGentCovid19",
            "scml_agents.scml2020.agent0x111.ASMASH",
            "scml_agents.scml2020.a_sengupta.Merchant",
            "scml_agents.scml2020.past_frauds.MhiranoAgent",
            "scml_agents.scml2020.monty_hall.MontyHall",
            "scml_agents.scml2020.team_19.Ashgent",
            "scml_agents.scml2020.team_17.WhAgent",
            "scml_agents.scml2020.team_10.UnicornAgent",
            "scml_agents.scml2020.threadfield.GreedyFactoryManager2",
            "scml_agents.scml2020.team_29.BIUDODY",
            "scml_agents.scml2020.team_20.CrescentAgent",
            "scml_agents.scml2020.team_27.AgentProjectGC",
            "scml_agents.scml2020.team_18.MercuAgent",
            "scml_agents.scml2020.biu_th.THBiuAgent",
            "scml_agents.scml2020.team_32.BeerAgent",
        }
    elif version == 2021:
        assert agents == {
            "scml_agents.scml2021.standard.bossagent.boss_agent.CharliesAgent",
            "scml_agents.scml2021.standard.iyibiteam.agent.IYIBIAgent",
            "scml_agents.scml2021.standard.team_41.a.SteadyMgr",
            "scml_agents.scml2021.standard.team_41.Augur_agent.AugurAgent",
            "scml_agents.scml2021.standard.team_41.sorcery.SorceryAgent",
            "scml_agents.scml2021.standard.team_44.agent68.Agent68",
            "scml_agents.scml2021.standard.team_45.stingy.StingyAgent",
            "scml_agents.scml2021.standard.team_46.solid.SolidAgent",
            "scml_agents.scml2021.standard.team_49.agent.E3BIUagent",
            "scml_agents.scml2021.standard.team_53.my_paibiu.MyPaibiuAgent",
            "scml_agents.scml2021.standard.team_67.polymorphic_agent.PolymorphicAgent",
            "scml_agents.scml2021.standard.team_78.yiy_agent.YIYAgent",
            "scml_agents.scml2021.standard.team_82.perry.PerryTheAgent",
            "scml_agents.scml2021.standard.team_91.bluewolf.BlueWolf",
            "scml_agents.scml2021.standard.team_may.m4.M4",
            "scml_agents.scml2021.standard.team_mediocre.mediocre.Mediocre",
            "scml_agents.scml2021.standard.wabisabikoalas.artisan_kangaroo.ArtisanKangaroo",
            "scml_agents.scml2021.oneshot.staghunter.myagent.StagHunterTough",
            "scml_agents.scml2021.oneshot.staghunter.myagent2.StagHunterV7",
            "scml_agents.scml2021.oneshot.team_50.sagiagent.Agent74",
            "scml_agents.scml2021.oneshot.team_51.qlagent_extended_state.QlAgent",
            "scml_agents.scml2021.oneshot.team_54.sopranos.TheSopranos78",
            "scml_agents.scml2021.oneshot.team_55.agent.Zilberan",
            "scml_agents.scml2021.oneshot.team_55.worker_agents.SimpleAgent",
            "scml_agents.scml2021.oneshot.team_55.worker_agents.BetterAgent",
            "scml_agents.scml2021.oneshot.team_55.worker_agents.AdaptiveAgent",
            "scml_agents.scml2021.oneshot.team_55.worker_agents.LearningAgent",
            "scml_agents.scml2021.oneshot.team_55.worker_agents.ImprovedLearningAgent",
            "scml_agents.scml2021.oneshot.team_61.agents.SimpleAgent",
            "scml_agents.scml2021.oneshot.team_61.agents.BetterAgent",
            "scml_agents.scml2021.oneshot.team_61.agents.BondAgent",
            "scml_agents.scml2021.oneshot.team_62.uc_oneshot_agent_v3_4.UcOneshotAgent3_4",
            "scml_agents.scml2021.oneshot.team_72.agent97.Agent97",
            "scml_agents.scml2021.oneshot.team_72.learning_agent.SimpleAgent",
            "scml_agents.scml2021.oneshot.team_72.learning_agent.BetterAgent",
            "scml_agents.scml2021.oneshot.team_72.learning_agent.AdaptiveAgent",
            "scml_agents.scml2021.oneshot.team_72.learning_agent.LearningAgent",
            "scml_agents.scml2021.oneshot.team_73.past_agents.SimpleAgent",
            "scml_agents.scml2021.oneshot.team_73.past_agents.BetterAgent",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AdaptiveAgent",
            "scml_agents.scml2021.oneshot.team_73.oneshot_agents.Gentle",
            "scml_agents.scml2021.oneshot.team_73.past_agents.SimpleAgent",
            "scml_agents.scml2021.oneshot.team_73.past_agents.BetterAgent",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AdaptiveAgent",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT064",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT063",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT062",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT061",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT060",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT056",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT055",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT054",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT053",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT052",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT051",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT050",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT049",
            "scml_agents.scml2021.oneshot.team_73.past_agents.AgentT048",
            "scml_agents.scml2021.oneshot.team_86.agent112.SimpleAgent",
            "scml_agents.scml2021.oneshot.team_86.agent112.BetterAgent",
            "scml_agents.scml2021.oneshot.team_86.agent112.Agent112",
            "scml_agents.scml2021.oneshot.team_90.run.PDPSyncAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.GoldfishParetoEmpiricalGodfatherAgent",
        }
    elif version == 2022:
        assert agents == {
            "scml_agents.scml2022.standard.bossagent.charlies.CharliesAgent",
            "scml_agents.scml2022.standard.team_100.skyagent.SkyAgent",
            "scml_agents.scml2022.standard.team_137.lobster.Lobster",
            "scml_agents.scml2022.standard.team_9.salesagent.SalesAgent",
            "scml_agents.scml2022.standard.team_99.smartagent.SmartAgent",
            "scml_agents.scml2022.standard.team_may.m5.M5",
            "scml_agents.scml2022.standard.wabisabikoalas.artisan_kangaroo.ArtisanKangaroo",
            "scml_agents.scml2022.collusion.bossagent.charlies.CharliesAgentCollusion",
            "scml_agents.scml2022.collusion.team_may.m5.M5Collusion",
            "scml_agents.scml2022.oneshot.team_102.agents.GentleS",
            "scml_agents.scml2022.oneshot.team_102.agents.LearningSyncAgent",
            "scml_agents.scml2022.oneshot.team_103.agent.MMMPersonalized",
            "scml_agents.scml2022.oneshot.team_105.agent.AdaptivePercentile",
            "scml_agents.scml2022.oneshot.team_106.moving_average_agent.AdamAgent",
            "scml_agents.scml2022.oneshot.team_107.regression_agent.EVEAgent",
            "scml_agents.scml2022.oneshot.team_123.neko.Neko",
            "scml_agents.scml2022.oneshot.team_124.agent.LearningAdaptiveAgent",
            "scml_agents.scml2022.oneshot.team_126.agents_learning.AgentSAS",
            "scml_agents.scml2022.oneshot.team_131.agentrm.AgentRM",
            "scml_agents.scml2022.oneshot.team_134.agent119.PatientAgent",
            "scml_agents.scml2022.oneshot.team_62.uc_oneshot_agent_v3_4.UcOneshotAgent3_4",
            "scml_agents.scml2022.oneshot.team_94.qlagent3.AdaptiveQlAgent",
            "scml_agents.scml2022.oneshot.team_96.agent125.Agent125",
        }


def test_finalists_2020():
    agents = get_agents(2020, track="std", finalists_only=True)
    assert len(agents) == 12
    agents = get_agents(2020, track="collusion", finalists_only=True)
    assert len(agents) == 6


def test_finalists_2021():
    agents = get_agents(2021, track="std", finalists_only=True)
    assert len(agents) == 5
    agents = get_agents(2021, track="collusion", finalists_only=True)
    assert len(agents) == 5
    agents = get_agents(2021, track="oneshot", finalists_only=True)
    assert len(agents) == 8


def test_finalists_2022():
    agents = get_agents(2022, track="std", finalists_only=True)
    assert len(agents) == 5
    agents = get_agents(2022, track="collusion", finalists_only=True)
    assert len(agents) == 2
    agents = get_agents(2022, track="oneshot", finalists_only=True)
    assert len(agents) == 8


def test_winners_2019():
    agents = get_agents(2019, track="std", winners_only=True)
    assert len(agents) == 3
    agents = get_agents(2019, track="collusion", winners_only=True)
    assert len(agents) == 3
    agents = get_agents(2019, track="all", winners_only=True)
    assert len(agents) == 4
    agents = get_agents(2019, track="sabotage", winners_only=True)
    assert len(agents) == 0


def test_winners_2021():
    agents = get_agents(2021, track="std", winners_only=True)
    assert len(agents) == 3, f"{agents}"
    agents = get_agents(2021, track="collusion", winners_only=True)
    assert len(agents) == 2, f"{agents}"
    agents = get_agents(2021, track="all", winners_only=True)
    assert len(agents) == 7, f"{agents}"
    agents = get_agents(2021, track="oneshot", winners_only=True)
    assert len(agents) == 4, f"{agents}"


def test_winners_2022():
    agents = get_agents(2022, track="std", winners_only=True)
    assert len(agents) == 3, f"{agents}"
    agents = get_agents(2022, track="collusion", winners_only=True)
    assert len(agents) == 1, f"{agents}"
    agents = get_agents(2022, track="all", winners_only=True)
    assert len(agents) == 7, f"{agents}"
    agents = get_agents(2022, track="oneshot", winners_only=True)
    assert len(agents) == 3, f"{agents}"


def test_winners_2020():
    agents = get_agents(2020, track="std", winners_only=True)
    assert len(agents) == 2
    agents = get_agents(2020, track="collusion", winners_only=True)
    assert len(agents) == 2
    agents = get_agents(2020, track="all", winners_only=True)
    assert len(agents) == 4


def test_winners_2024():
    agents = get_agents(2024, track="std", winners_only=True)
    assert len(agents) == 1
    agents = get_agents(2024, track="oneshot", winners_only=True)
    assert len(agents) == 4
    agents = get_agents(2024, track="collusion", winners_only=True)
    assert len(agents) == 0
    agents = get_agents(2024, track="all", winners_only=True)
    assert len(agents) == 5


def test_winners_2025():
    agents = get_agents(2025, track="std", winners_only=True)
    assert len(agents) == 3
    agents = get_agents(2025, track="oneshot", winners_only=True)
    assert len(agents) == 3
    agents = get_agents(2025, track="collusion", winners_only=True)
    assert len(agents) == 0
    agents = get_agents(2025, track="all", winners_only=True)
    assert len(agents) == 6


def test_finalists_2025():
    agents = get_agents(2025, track="std", finalists_only=True)
    assert len(agents) == 4
    agents = get_agents(2025, track="oneshot", finalists_only=True)
    assert len(agents) == 5
    agents = get_agents(2025, track="collusion", finalists_only=True)
    assert len(agents) == 0
    agents = get_agents(2025, track="all", finalists_only=True)
    assert len(agents) == 9
