This repository contains agents for the SCML world.

To install this package just run:

>>> pip install scml-agents

There are two ways to submit agents to this repository:

1. Participate in the SCML competition `https://scml.cs.brown.edu <https://scml.cs.brown.edu>`_
2. Submit a pull-request with your agent added to the contrib directory.


**Please note that torch does not yet fully support python 3.11. If you face issues installing (especially on a mac), try to use Python 3.10.**

Getting lists of agents
=======================

You can get any specific subset of the agents in the library using `get_agents()`. This function
has the following parameters:

* version: Either a competition year (2019, 2020, 2021, ....) or the value "contrib" for all other agents. You can also pass "all" or "any" to get all agents.
* track: The track (any, std, oneshot [from 2021], sabotage[only for 2019], collusion [until 2023]).
* qualified_only: If true, only agents that were submitted to SCML and ran in the qualifications round will be
                  returned.
* finalists_only: If true, only agents that were submitted to SCML and passed qualifications will be
                  returned.
* winners_only: If true, only winners of SCML (the given version) will be returned.
* bird_only: If true, only winners the bird award are returned (new in 2021).
* top_only: Either a fraction of finalists or the top n finalists with highest scores in the finals of
            SCML.
* as_class: If true, the agent classes will be returned otherwise their full class names.


For example, to get the top 10% of the Oneshot track finalists in year 2024 as strings, you can use:

>>> get_agents(version=2025, track="oneshot", finalists_only=True, top_only=0.1, as_class=False)

.. BEGIN generated standings region

.. BEGIN generated standings: 2026

ANAC 2026 Results
=================

OneShot Track
-------------

Qualified agents (16)
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - #
     - Agent
     - ID
     - Author
     - Team
     - Institute
     - Country
   * - 1
     - agent03
     - 21244
     - Yifan Zhang
     - Team 403
     - South China University of Technology
     - China
   * - 2
     - AssariAsari
     - 21089
     - Rinon Asanuma
     - Team 376
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 3
     - BayesianAgent
     - 21637
     - kouta miyamoto
     - Team 371
     - Kyoto University
     - Japan
   * - 4
     - CodexAgentOneshot
     - 21804
     - Ryota GENSEKI
     - Team 507
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 5
     - EFRDist
     - 21022
     - Mason Hagan
     - Team 353
     - Brown University
     - United States
   * - 6
     - Group3
     - 21562
     - Yeabsira Simeka Maru
     - Team 491
     - Özyeğin University
     - Türkiye
   * - 7
     - heyoneshot
     - 21798
     - カズマ
     - チーム298
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 8
     - IsobeAgent
     - 21095
     - Hayate Ogata
     - Team 380
     - Nagoya Institute of Technology
     - Japan
   * - 9
     - LatticeOneshotAgent
     - 21484
     - Yuzuru Kitamura
     - Team179
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 10
     - PenaltyAvoid
     - 21041
     - Tarek Medhat
     - Team 357
     - Assiut University
     - Egypt
   * - 11
     - SBDOneShot
     - 21813
     - Hajime Endo
     - Team Ukku
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 12
     - SteadySyncAgent
     - 21056
     - Sora Kawase
     - Team 364
     - Kyoto University
     - Japan
   * - 13
     - supvelikos
     - 21619
     - Kosuke Nakata
     - Team 412
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 14
     - TimeWeightedAgent
     - 21155
     - ishigami
     - Team 400
     - Nagoya Institute of Technology
     - Japan
   * - 15
     - Yamashitaagent
     - 21196
     - Masaki Yamashita
     - Team 369
     - Kyoto University
     - Japan
   * - 16
     - Ysi
     - 21799
     - Ryoshin Hatakeyama
     - Team 497
     - Tokyo University of Agriculture and Technology
     - Japan

Get them after install with:

>>> scml_agents.get_agents(2026, track="oneshot", qualified_only=True)

**Disqualified (1):** SKRI

Standard Track
--------------

Qualified agents (20)
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - #
     - Agent
     - ID
     - Author
     - Team
     - Institute
     - Country
   * - 1
     - AgeAgeAgent
     - 21165
     - MorikawaYuto
     - Team 404
     - Nagoya Institute of Technology
     - Japan
   * - 2
     - agent01std
     - 21195
     - Yifan Zhang
     - Team 403
     - South China University of Technology
     - China
   * - 3
     - ArionAgent
     - 21021
     - Muhammad Raees Azam
     - Arion Strategists
     - Özyeğin University
     - Türkiye
   * - 4
     - BalancedGreedyStdAgent
     - 20964
     - Takaya
     - Team 341
     - Nagoya Institute of Technology
     - Japan
   * - 5
     - CodexAgentStd
     - 21803
     - Ryota GENSEKI
     - Team 507
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 6
     - COW
     - 21303
     - Naoki Ushida
     - Team 389
     - Nagoya Institute of Technology
     - Japan
   * - 7
     - EmSel
     - 21084
     - Emre Karaarslan
     - Team EmSel
     - Özyeğin University
     - Türkiye
   * - 8
     - GS3
     - 21201
     - nakayamarintaro
     - Team 411
     - Nagoya Institute of Technology
     - Japan
   * - 9
     - heystd
     - 21800
     - カズマ
     - チーム298
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 10
     - HorizonAwareAgent
     - 21653
     - Yuzuru Kitamura
     - Team179
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 11
     - KotaAgent
     - 21747
     - Hibino-Kota
     - Team 337
     - Nagoya Institute of Technology
     - Japan
   * - 12
     - MyAgent17803196313323
     - 21559
     - Hamid Majidi Balanji
     - Team 484
     - Özyeğin University
     - Türkiye
   * - 13
     - okagent
     - 20941
     - Kohga Okashita
     - Team 336
     - Nagoya Institute of Technology
     - Japan
   * - 14
     - Rohn
     - 21695
     - Ryoshin Hatakeyama
     - Team 497
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 15
     - SBD
     - 21650
     - Hajime Endo
     - Team Ukku
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 16
     - ShimijimiShijimi
     - 21127
     - Rinon Asanuma
     - Team 376
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 17
     - SugaiAgent
     - 21313
     - Toshiki Sugai
     - Team 328
     - NIT
     - Japan
   * - 18
     - SuperimagentZ
     - 21683
     - Aoi Imazeki
     - Team 342
     - Nagoya Institute of Technology
     - Japan
   * - 19
     - SupmerkosV3
     - 22272
     - Kosuke Nakata
     - Team 412
     - Tokyo University of Agriculture and Technology
     - Japan
   * - 20
     - TakaLinkAgent
     - 21282
     - Takafumi Omori
     - TakaTakaWorks
     - Nagoya Institute of Technology
     - Japan

Get them after install with:

>>> scml_agents.get_agents(2026, track="standard", qualified_only=True)

.. END generated standings: 2026

.. BEGIN generated standings: 2019

ANAC 2019 Results
=================

Standard Track
--------------

Qualified agents (6)
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - #
     - Agent
     - ID
     - Author
     - Team
     - Institute
     - Country
   * - 1
     - CheapBuyerFactoryManager
     - 21821
     - cheap_buyer
     - cheap_buyer
     - —
     - —
   * - 2
     - FJ2FactoryManager
     - 21817
     - fj2
     - fj2
     - —
     - —
   * - 3
     - InsuranceFraudFactoryManager
     - 21819
     - iffm
     - iffm
     - —
     - —
   * - 4
     - NVMFactoryManager
     - 21822
     - nvm
     - nvm
     - —
     - —
   * - 5
     - RaptFactoryManager
     - 21818
     - rapt_fm
     - rapt_fm
     - —
     - —
   * - 6
     - SAHAFactoryManager
     - 21820
     - saha
     - saha
     - —
     - —

.. END generated standings: 2019

.. END generated standings region

Winners of the SCML 2025 Competition
====================================

Oneshot Track
-------------
* First Place (tie): Yuzuru Kitamura for **CostAverseAgent**
* First Place (tie): Shota Takayama for **Rchan**
* First Place (tie): Rikuto Takano and Takeaki Sakabe for **AlmostEqualAgent**

You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2024, track="oneshot", winners_only=True)

Standard Track
--------------
* First Place: Atsunaga Sadahiro for **AS0**
* Second Place: Sota Sakaguchi and Takanobu Otsuka for **XenoSotaAgent**
* Third Place: Sora Nishizaki and Takanobu Otsuka for **UltraSuperMiracleSoraFinalAgentZ**

You can get this agent after installing scml-agents by running:

>>> scml_agents.get_agents(2024, track="std", winners_only=True)

Winners of the SCML 2024 Competition
====================================

Oneshot Track
-------------
* First Place: Ryoga Miyajima for **CautiousOneShotAgent**
* Second Place: Arnie He, Akash Singirikonda, and Amy Greenwald for **MatchingPennies**
* Third Place (tie): Hajime Endo for **DistRedistAgent**
* Third Place (tie): Yuzuru Kitamura for **EpsilonGreedyAgent**

You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2024, track="oneshot", winners_only=True)

Standard Track
--------------
* First Place: Gou Kazusa for **PenguinAgent**

You can get this agent after installing scml-agents by running:

>>> scml_agents.get_agents(2024, track="std", winners_only=True)

Winners of the SCML 2023 Competition
====================================

Oneshot Track
-------------
* First Place: Pedro Hrosz Turini and Jaime Sichman for **QuantityOrientedAgent**
* Second Place: Shota Kimata and Yuko Sakurai for **CCAgent**
* Third Place: Masato Kijima for **KanbeAgent**

You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2023, track="oneshot", winners_only=True)

Standard Track
--------------
* First Place: Sadahiro Atsunaga for **AgentSDH**

You can get this agent after installing scml-agents by running:

>>> scml_agents.get_agents(2023, track="std", winners_only=True)

Collusion Track
---------------
* Honorary Mention: Kazuki Komori for **M5**

You can get this agent after installing scml-agents by running:

>>> scml_agents.get_agents(2023, track="collusion", winners_only=True)

Winners of the SCML 2022 Competition
====================================

Oneshot Track
-------------
* First Place: Chris Mascioli and Amy Greenwald for **PatientAgent**
* Second Place: Takumu Shimizu for **GentleS**
* Third Place: Shiraz Nave, Amit Dayan, Sariel Turayfor **AgentSAS**

You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2022, track="oneshot", winners_only=True)

Standard Track
--------------
* First Place: Ito Nobuhiro and Takanobu Otsukafor **Lobster**
* Second Place: Kazuki Komori for **M5**
* Third Place: Koki Katagiri and Tatanobu Otsuka for **Artisan Kangaroo**

You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2022, track="std", winners_only=True)

Collusion Track
---------------
* Honorary Mention: Kazuki Komori for **M5**

You can get this agent after installing scml-agents by running:

>>> scml_agents.get_agents(2022, track="collusion", winners_only=True)

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


Agents accepted for the SCML 2025 qualifications
================================================

A list of all agents accepted for the SCML 2025 qualifications round can be found at `https://scml.cs.brown.edu/scml2025 <https://scml.cs.brown.edu/scml2022>`_

Agents accepted for the SCML 2024 qualifications
================================================

A list of all agents accepted for the SCML 2024 qualifications round can be found at `https://scml.cs.brown.edu/scml2024 <https://scml.cs.brown.edu/scml2022>`_

Agents accepted for the SCML 2023 qualifications
================================================

A list of all agents accepted for the SCML 2023 qualifications round can be found at `https://scml.cs.brown.edu/scml2023 <https://scml.cs.brown.edu/scml2022>`_

Agents accepted for the SCML 2022 qualifications
================================================

A list of all agents accepted for the SCML 2022 qualifications round can be found at `https://scml.cs.brown.edu/scml2022 <https://scml.cs.brown.edu/scml2022>`_

Agents accepted for the SCML 2021 qualifications
================================================

A list of all agents accepted for the SCML 2021 qualifications round can be found at `https://scml.cs.brown.edu/scml2021 <https://scml.cs.brown.edu/scml2021>`_


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
