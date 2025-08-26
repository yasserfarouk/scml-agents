import unittest
from .litaagent_std import litaagent_y, litaagent_yr


class DummyState:
    def __init__(self, rel):
        self.relative_time = rel


class DummyAWI:
    current_step = 0
    current_shortfall_penalty = 10


def stub_agent(agent):
    agent._best_price = lambda pid: 100
    agent._calc_opponent_concession = lambda pid, price: 0.2
    agent._expected_price = lambda pid, default: default
    agent._is_consumer = lambda pid: True
    agent._awi = DummyAWI()
    agent._clamp_price = lambda pid, price: price
    agent.im = None


class TestAgentEquivalence(unittest.TestCase):
    def test_helper_functions(self):
        partners = ["a", "b", "c", "d", "e"]
        self.assertEqual(
            litaagent_y._split_partners(partners),
            litaagent_yr._split_partners(partners),
        )
        import random
        import numpy as np

        random.seed(0)
        np.random.seed(0)
        res1 = litaagent_y._distribute(7, 3)
        random.seed(0)
        np.random.seed(0)
        res2 = litaagent_yr._distribute(7, 3)
        self.assertEqual(res1, res2)

    def test_concession_equivalence(self):
        ay = litaagent_y.LitaAgentY()
        ayr = litaagent_yr.LitaAgentYR()
        stub_agent(ay)
        stub_agent(ayr)
        state = DummyState(0.5)
        price_y = ay._apply_concession("p", 110, state, 120)
        price_yr = ayr._calc_conceded_price("p", 110, state, 120)
        self.assertAlmostEqual(price_y, price_yr)

    def test_update_strategy(self):
        ay = litaagent_y.LitaAgentY()
        ayr = litaagent_yr.LitaAgentYR()
        ay.update_profit_strategy(min_profit_margin=0.2, cheap_price_discount=0.5)
        ayr.update_profit_strategy(min_profit_ratio=0.2, bargain_threshold=0.5)
        self.assertEqual(ay.min_profit_margin, ayr.min_profit_ratio)
        self.assertEqual(ay.cheap_price_discount, ayr.bargain_threshold)


if __name__ == "__main__":
    unittest.main()
