This repository contains agents for the SCML world.

To install this package just run:

>>> pip install scml-agents

There are two ways to submit agents to this repository:

1. Participate in the SCML competition `https://scml.cs.brown.edu <https://scml.cs.brown.edu>`_
2. Submit a pull-request with your agent added to the contrib directory.


Getting lists of agents
=======================

You can get any specific subset of the agents in the library using `get_agents()`. This function
has the following parameters:

* version: Either a competition year (2019, 2020, 2021, ....) or the value "contrib" for all other agents. You can also pass "all" or "any" to get all agents.
* track: The track (any, collusion, std, sabotage[only for 2019], oneshot [from 2021]).
* qualified_only: If true, only agents that were submitted to SCML and ran in the qualifications round will be
                  returned
* finalists_only: If true, only agents that were submitted to SCML and passed qualifications will be
                  returned
* winners_only: If true, only winners of SCML (the given version) will be returned.
* bird_only: If true, only winners the bird award are returned (new in 2021)
* top_only: Either a fraction of finalists or the top n finalists with highest scores in the finals of
            SCML
* as_class: If true, the agent classes will be returned otherwise their full class names.


For example, to get the top 10% of the Oneshot track finalists in year 2021 as strings, you can use:

>>> get_agents(version=2021, track="oneshot", finalists_only=True, top_only=0.1, as_class=False)

Winners of the SCML 2021 Competition
====================================

Oneshot Track
-------------
* First Place: Assaf Tirangel, Yossi Weizman, Inbal Avraham for **Agent112**
* Second Place: Takumu Shimizu for **Gentle**
* Third Place (tie): Sagi Nachum for **Agent74**
* Third Place (tie): Yuchen Liu, Rafik Hadfi and Takayuki Ito for **UCOneshotAgent**

You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2021, track="oneshot", winners_only=True)

Standard Track
--------------
* First Place: Kazuki Komori for **M4**
* Second Place: Mehmet Onur Keskin, Umit Cakan, Gevher Yesevi, Reyhan Aydogan, Amy Greenwald for **CharliesAgent**
* Third Place: Koki Katagiri for **Artisan Kangaroo**

You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2021, track="std", winners_only=True)

Collusion Track
---------------
* First Place: Kazuki Komori for **M4**
* Second Place: Mehmet Onur Keskin, Umit Cakan, Gevher Yesevi, Reyhan Aydogan, Amy Greenwald for **CharliesAgent**

You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2021, track="collusion", winners_only=True)

Winners of the SCML 2020 Competition
====================================

Standard Track
--------------
* First Place: Masahito Okuno for **SteadyMgr**
* Second Place: Guy Heller, E. Gerson, I. Hen and M. Akrabi for **Agent30**

You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2020, track="std", winners_only=True)

Collusion Track
---------------
* First Place: Kazuki Komori for **MMM**
* Second Place: Ayan Sengupta for **Merchant**

You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2020, track="collusion", winners_only=True)


Agents accepted for the SCML 2021 qualifications
================================================

A list of all agents accepted for the SCML 2021 qualifications round can be found at `https://scml.cs.brown.edu <https://scml.cs.brown.edu>`_

Agents accepted for the SCML 2020 qualifications
================================================

This is a list of all the agents accepted for the SCML 2020 qualifications round.

============= ============= =======================  ====================================================================================================
  Team          Identifier    Agent/Class name         Team Members
============= ============= =======================  ====================================================================================================
  a-sengupta    a-sengupta    Merchant                 Ayan Sengupta
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  Past Frauds   past_frauds   MhiranoAgent             Masanori Hirano
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  Team 15       team_15       SteadyMgr                Masahito Okuno
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  Team 17       team_17       WhAgent                  Noriko Yuasa
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  Team 18       team_18       Mercu                    Kazuto Kakutani
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  Team 19       team_19       Ashgent                  Shuhei Aoyama
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  Team 20       team_20       CrescentAgent            Yuki Yoshimura
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  Team 22       team_22       SavingAgent              Takuma Kawamura
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  ThreadField   threadfield   GreedyFactoryManager2    Yuta Hosokawa
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  Team May      team_may      MMM                      Kazuki Komori
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  Team 10       team_10       UnicornAgent             Dolev Mutzari
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  BARgent       bargent       BARGentCovid19           Zacharie Cohen, O. Fogler, D. Neuman and R. Cohen
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  BIU-TH        biu_th        THBiu                    Haim Nafcha
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  agent0x111    agent0x111    ASMASH                   Matanya, Shmulik, Assaf
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  Team 25       team_25       Agent30                  Guy Heller, E. Gerson, I. Hen and M. Akrabi
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
 Team 29       team_29       BIUDODY                  Dror Levy, D. Joffe and O. Nagar
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
 Team 32       team_32       BeerAgent                Benjamin Wexler
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
 Team 27       team_27       AgentProjectGC           Cihan Eran and Gevher Yesevi
------------- ------------- -----------------------  ----------------------------------------------------------------------------------------------------
  MontyHall     montyhall     MontyHall                Enrique Areyan Viqueira, E. Li, D. Silverston, A. Sridhar, J. Tsatsaros, A. Yuan and A. Greenwald
============= ============= =======================  ====================================================================================================

 You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2020, track="any")


Installation Note
=================

If you are on Apple M1, you will need to install tensorflow **before** installing this package on conda using the method described `here <https://developer.apple.com/metal/tensorflow-plugin/>`_
