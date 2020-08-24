from scml_agents import get_agents
from pytest import mark


@mark.parametrize(
    "version", [2019, 2020],
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


def test_winners_2020():
    # agents = get_agents(2020, track="collusion", winners_only=True)
    # assert len(agents) == 3
    # agents = get_agents(2020, track="all", winners_only=True)
    # assert len(agents) == 4
    pass


def test_winners_2019():
    agents = get_agents(2019, track="std", winners_only=True)
    assert len(agents) == 3
    agents = get_agents(2019, track="collusion", winners_only=True)
    assert len(agents) == 3
    agents = get_agents(2019, track="all", winners_only=True)
    assert len(agents) == 4
    agents = get_agents(2019, track="sabotage", winners_only=True)
    assert len(agents) == 0
