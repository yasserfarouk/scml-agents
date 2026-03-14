#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import LinearRegression

from .other_agents.agent_team86 import AgentOneOneTwo

QUANTITY = 0
TIME = 1
UNIT_PRICE = 2


__all__ = [
    "EVEAgent",
]


class EVEAgent(AgentOneOneTwo):
    def _price_range(self, nmi):
        mn, mx = super()._price_range(nmi)
        if self._is_selling(nmi):
            if len(self._best_selling) > 0:
                x = np.array([i for i in range(len(self._best_selling))]).reshape(-1, 1)
                reg = LinearRegression().fit(
                    x, np.array(self._best_selling).reshape(-1, 1)
                )
                mn = max(mn, reg.predict(np.array([len(x) + 1]).reshape(-1, 1)).item())
        else:
            if len(self._best_buying) > 0:
                x = np.array([i for i in range(len(self._best_buying))]).reshape(-1, 1)
                reg = LinearRegression().fit(
                    x, np.array(self._best_buying).reshape(-1, 1)
                )
                mx = min(mx, reg.predict(np.array([len(x) + 1]).reshape(-1, 1)).item())

        return mn, mx


if __name__ == "__main__":
    from other_agents.agent_template import print_type_scores, try_agent

    world, ascores, tscores = try_agent(EVEAgent)
    print_type_scores(tscores)
