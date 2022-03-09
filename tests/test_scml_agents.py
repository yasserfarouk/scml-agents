from pytest import mark

from scml_agents import get_agents


@mark.parametrize(
    "version",
    [2019, 2020, 2021],
)
def test_get_agents_per_year(version):
    from pprint import pprint

    agents = get_agents(version, as_class=False)
    pprint(agents)
    if version == 2019:
        assert agents == (
            "scml_agents.scml2019.fj2.FJ2FactoryManager",
            "scml_agents.scml2019.rapt_fm.RaptFactoryManager",
            "scml_agents.scml2019.iffm.InsuranceFraudFactoryManager",
            "scml_agents.scml2019.saha.SAHAFactoryManager",
            "scml_agents.scml2019.cheap_buyer.cheapbuyer.CheapBuyerFactoryManager",
            "scml_agents.scml2019.nvm.nmv_agent.NVMFactoryManager",
            "scml_agents.scml2019.monopoly.Monopoly",
            "scml_agents.scml2019.psfm.PenaltySabotageFactoryManager",
        )
    elif version == 2020:
        assert agents == (
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
        )
    elif version == 2021:
        assert agents == (
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
            "scml_agents.scml2021.oneshot.staghunter.myagent2.StagHunter",
            "scml_agents.scml2021.oneshot.staghunter.myagent.StagHunterTough",
            "scml_agents.scml2021.oneshot.staghunter.myagent2.StagHunter",
            "scml_agents.scml2021.oneshot.staghunter.myagent2.StagHunterV5",
            "scml_agents.scml2021.oneshot.staghunter.myagent2.StagHunterV6",
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
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.MinDisagreementGodfatherAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.MinEmpiricalGodfatherAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.AspirationUniformGodfatherAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.NonconvergentGodfatherAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.ParetoEmpiricalGodfatherAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.GoldfishParetoEmpiricalGodfatherAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.SlowGoldfish",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.HardnosedGoldfishGodfatherAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.HardnosedGoldfishBiggerAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.HardnosedGoldfishSmallerAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.SoftnosedGoldfishGodfatherAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.QuickLearningGodfather",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.MediumLearningGodfather",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.SlowLearningGodfather",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.ZooGodfather",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.TrainingCollectionGodfatherAgent",
            "scml_agents.scml2021.oneshot.team_corleone.godfather.godfather.ChristopherTheGoldfishAgent",
        )


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
    assert len(agents) == 3
    agents = get_agents(2021, track="collusion", winners_only=True)
    assert len(agents) == 2
    agents = get_agents(2021, track="oneshot", winners_only=True)
    assert len(agents) == 3 and len(agents[-1]) == 2
    agents = get_agents(2021, track="all", winners_only=True)
    assert len(agents) == 7


def test_winners_2020():
    agents = get_agents(2020, track="std", winners_only=True)
    assert len(agents) == 2
    agents = get_agents(2020, track="collusion", winners_only=True)
    assert len(agents) == 2
    agents = get_agents(2020, track="all", winners_only=True)
    assert len(agents) == 4
