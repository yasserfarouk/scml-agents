This repository contains agents for the SCML world.

To install this package just run:

>>> pip install scml-agents

There are two ways to submit agents to this repository:

1. Participate in the SCML competition `https://scml.cs.brown.edu <https://scml.cs.brown.edu>`_
2. Submit a pull-request with your agent added to the contrib directory.


Winners of the SCML 2020 Competition
=====================================

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


Finalists of the SCML 2020 Competition
======================================

Standard Track
--------------

* Number of Configurations: 1,256
* Number of Simulation: 195,936
* Number of Instantiations: 15,072 in 15,072 simulations


=== ================ ======== ======== ========= ========= ============ ======= ======
 #   Agent             mean     std      min       25%       Median       75%     max  
=== ================ ======== ======== ========= ========= ============ ======= ======
 1   SteadyMgr        0.0758   0.1627   -0.3536   -0.0078   **0.0783**   0.161   7.52 
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 2   Agent30          0.0199   0.0376   -0.1813   -0.0002   **0.0127**   0.041   1.21 
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 3   CrescentAgent    -0.000   0.0015   -0.0375   0         **0**        0       0.05 
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 4   Whagent          -0.034   0.1333   -2.5990   -0.0061   **0**        0       2.47 
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 5   Merchant         -0.033   0.0818   -0.5046   -0.0575   **-0.019**   -0.00   1.87 
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 6   MMM              -0.036   0.1635   -2.5582   -0.0829   **-0.022**   0.015   3.28 
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 7   MontyHall        0.0006   0.2187   -0.5264   -0.0394   **-0.022**   -0.00   4.63 
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 8   BeerAgent        -0.022   0.1185   -0.5417   -0.0791   **-0.025**   0.042   8.16 
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 9   UnicornAgent     -0.067   0.1886   -2.1693   -0.1271   **-0.047**   0.008   7.42 
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 *   Decentralizing   -0.124   0.2660   -2.4583   -0.1763   **-0.081**   -0.01   5.07 
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 10  Saving_Agent     -0.127   0.2775   -2.4318   -0.1785   **-0.082**   -0.02   8.94 
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 11  THBiuAgent       -0.125   0.2538   -2.4092   -0.1790   **-0.083**   -0.02   5.38
--- ---------------- -------- -------- --------- --------- ------------ ------- ------
 12  GFM2             -0.109   0.1978   -0.6301   -0.2021   **-0.087**   -0.01   7.62 
=== ================ ======== ======== ========= ========= ============ ======= ======

You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2020, track="std", finalists_only=True)

Collusion Track
---------------

* Number of Configurations: 940
* Number of Simulation: 39,480
* Number of Instantiations: 16,920 in 5,640 simulations


==== =============== ====== ===== ====== ====== ====== ====== =====
  #   Agent Type      mean   std   min    25%   Median  75%    max
==== =============== ====== ===== ====== ====== ====== ====== =====
 1   MMM             0.4547 1.379 -2.770 -0.191 **0**  0.3301 15.66
---- --------------- ------ ----- ------ ------ ------ ------ -----
 2   Merchant        0.1127 0.681 -0.801 -0.792 **0**  0.5656 1.624
---- --------------- ------ ----- ------ ------ ------ ------ -----
 3   SteadyMgr       0.0390 0.300 -0.445 -0.054 **0**  0.0792 6.678
---- --------------- ------ ----- ------ ------ ------ ------ -----
 4   Agent30         0.0215 0.087 -0.095 -0.002 **0**  0.0174 1.086
---- --------------- ------ ----- ------ ------ ------ ------ -----
 5   Whagent         -0.020 0.057 -0.738 0      **0**  0      0.145
---- --------------- ------ ----- ------ ------ ------ ------ -----
 6   CrescentAgent   -0.001 0.004 -0.048 0      **0**  0      0.165
---- --------------- ------ ----- ------ ------ ------ ------ -----
 -   Decentralizing  -0.017 0.485 -2.340 -0.111 -0.06  -0.002 8.633
==== =============== ====== ===== ====== ====== ====== ====== =====


You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2020, track="collusion", finalists_only=True)


Agents accepted for the SCML 2020 qualifications
================================================

This is a list of all the agents accepted for the SCML 2020 qualifications round. 

 ============= ============= =======================  =============================================
  Team          Identifier    Agent/Class name         Team Members
 ============= ============= =======================  =============================================
  a-sengupta    a-sengupta    Merchant                 Ayan Sengupta
 ------------- ------------- -----------------------  ---------------------------------------------
  Past Frauds   past_frauds   MhiranoAgent             Masanori Hirano
 ------------- ------------- -----------------------  ---------------------------------------------
  Team 15       team_15       SteadyMgr                Masahito Okuno
 ------------- ------------- -----------------------  ---------------------------------------------
  Team 17       team_17       WhAgent                  Noriko Yuasa
 ------------- ------------- -----------------------  ---------------------------------------------
  Team 18       team_18       Mercu                    Kazuto Kakutani
 ------------- ------------- -----------------------  ---------------------------------------------
  Team 19       team_19       Ashgent                  Shuhei Aoyama
 ------------- ------------- -----------------------  ---------------------------------------------
  Team 20       team_20       CrescentAgent            Yuki Yoshimura
 ------------- ------------- -----------------------  ---------------------------------------------
  Team 22       team_22       SavingAgent              Takuma Kawamura
 ------------- ------------- -----------------------  ---------------------------------------------
  ThreadField   threadfield   GreedyFactoryManager2    Yuta Hosokawa
 ------------- ------------- -----------------------  ---------------------------------------------
  Team May      team_may      MMM                      Kazuki Komori
 ------------- ------------- -----------------------  ---------------------------------------------
  Team 10       team_10       UnicornAgent             Dolev Mutzari
 ------------- ------------- -----------------------  ---------------------------------------------
  BARgent       bargent       BARGentCovid19           Zacharie Cohen, O. Fogler, D. Neuman and R. Cohen
 ------------- ------------- -----------------------  ---------------------------------------------
  BIU-TH        biu_th        THBiu                    Haim Nafcha
 ------------- ------------- -----------------------  ---------------------------------------------
  agent0x111    agent0x111    ASMASH                   Matanya, Shmulik, Assaf
 ------------- ------------- -----------------------  ---------------------------------------------
  Team 25       team_25       Agent30                  Guy Heller, E. Gerson, I. Hen and M. Akrabi
 ------------- ------------- -----------------------  ---------------------------------------------
  Team 29       team_29       BIUDODY                  Dror Levy, D. Joffe and O. Nagar
 ------------- ------------- -----------------------  ---------------------------------------------
  Team 32       team_32       BeerAgent                Benjamin Wexler
 ------------- ------------- -----------------------  ---------------------------------------------
  Team 27       team_27       AgentProjectGC           Cihan Eran and Gevher Yesevi
 ------------- ------------- -----------------------  ---------------------------------------------
  MontyHall     montyhall     MontyHall                Enrique Areyan Viqueira, E. Li, D. Silverston, A. Sridhar, J. Tsatsaros, A. Yuan and A. Greenwald
 ============= ============= =======================  =============================================
 
 You can get these agents after installing scml-agents by running:

>>> scml_agents.get_agents(2020, track="collusion")
