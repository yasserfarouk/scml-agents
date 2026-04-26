import operator

import numpy
from biu_dody import DecentralizingAgent, DrorOmer, DrorStepAgent, return_agent_scores
from biu_dody1 import DrorDanaStepAgent
from biu_dody2 import DrorDana2StepAgent
from scml import SCML2020World


def learn_param():
    max = -1000
    index = 0
    for i in numpy.arange(0.01, 1, 0.1):
        test_agent = DrorStepAgent
        test_agent.set_param(test_agent, float(i))
        res = play_n_game(10, test_agent)
        if res > max:
            max = res
            index = i
    print(f"index = {index}, max = {max}")

    max = -1000
    for i in numpy.arange(index - 0.2, index + 0.2, 0.02):
        test_agent = DrorStepAgent
        test_agent.set_param(test_agent, float(i))
        res = play_n_game(15, test_agent)
        if res > max:
            max = res
            index = i
    print(f"index = {index}, max = {max}")


def play_n_game(n, dror_agent):
    scores = {}
    scores["DrorStepAgent"] = 0
    scores["DecentralizingAgent"] = 0

    for n_simulations in range(n):
        world = SCML2020World(
            **SCML2020World.generate(
                [dror_agent, DecentralizingAgent, IndDecentralizingAgent], n_steps=10
            ),
            construct_graphs=True,
        )

        world.run()
        returned_scores = return_agent_scores(world)
        winner = (
            "DrorStepAgent"
            if returned_scores.get("DrorStepAgent", -20)
            >= returned_scores.get("DecentralizingAgent", -20)
            else "DecentralizingAgent"
        )

        scores[winner] += 1
    return scores["DrorStepAgent"] - scores["DecentralizingAgent"]


def play_n_games(n, compare0, compare1, compare2, compare3):
    scores = {}
    scores["DrorStepAgent"] = 0
    scores["DecentralizingAgent"] = 0
    scores["DrorDanaStepAgent"] = 0
    scores["DrorOmer"] = 0
    scores["DrorDana2StepAgent"] = 0
    for n_simulations in range(n):
        world = SCML2020World(
            **SCML2020World.generate(
                [compare0, DecentralizingAgent, compare1, compare2], n_steps=30
            ),
            construct_graphs=True,
        )

        world.run()
        returned_scores = return_agent_scores(world)
        winner = max(returned_scores.items(), key=operator.itemgetter(1))[0]
        scores[winner] += 1
    return scores
    # print("step:%s, simulation:%s, winner_is:%s" %(output_step, n_simulations, winner))
    # print("dror:%s , decent...:%s , ind...:%s" % (returned_scores.get('DrorStepAgent'),  returned_scores.get('DecentralizingAgent'), returned_scores.get('IndDecentralizingAgent')))

    # names = ['DrorStepAgent', 'DecentralizingAgent', 'IndDecentralizingAgent']
    #
    #
    # for name in names:
    #     if scores[name].get(output_step):
    #         scores[name][output_step] += returned_scores.get(name, 0)
    #     else:
    #         scores[name][output_step] = returned_scores.get(name, 0)


# simulation with our agent

# learn_param()
scores = play_n_games(
    30, DrorStepAgent, DrorDanaStepAgent, DrorOmer, DrorDana2StepAgent
)
print(scores)
# scores = {}
# wins = {}
# scores['DrorStepAgent'] = {}
# scores['DecentralizingAgent'] = {}
# scores['IndDecentralizingAgent'] = {}
# exec_fraction = 0.1
# for output_step in range(3,8):
#     for n_simulations in range(10):
#         #test_agent = DrorStepAgent(exec=0.1)
#         # test_agent = DrorStepAgent
#         # test_agent.set_param(test_agent, exec_fraction=output_step/10)
#         exec_fraction = output_step/10
#         world = SCML2020World(
#             **SCML2020World.generate([DrorStepAgent, DecentralizingAgent, IndDecentralizingAgent], n_steps=10),
#             construct_graphs=True,
#         )
#
#         world.run()
#         returned_scores = return_agent_scores(world)
#         winner = 'DrorStepAgent' if returned_scores.get('DrorStepAgent', -20) >= returned_scores.get('DecentralizingAgent', -20) else 'DecentralizingAgent'
#         print("step:%s, simulation:%s, winner_is:%s" %(output_step, n_simulations, winner))
#         print("dror:%s , decent...:%s , ind...:%s" % (returned_scores.get('DrorStepAgent'),  returned_scores.get('DecentralizingAgent'), returned_scores.get('IndDecentralizingAgent')))
#
#         names = ['DrorStepAgent', 'DecentralizingAgent', 'IndDecentralizingAgent']
#         for name in names:
#             if scores[name].get(output_step):
#                 scores[name][output_step] += returned_scores.get(name, 0)
#             else:
#                 scores[name][output_step] = returned_scores.get(name, 0)
#
# print (scores)
