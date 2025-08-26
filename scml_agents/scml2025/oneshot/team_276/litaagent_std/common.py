# exports the name of the training algorithm
from pathlib import Path

import numpy as np
from gymnasium import spaces
from negmas.outcomes import Outcome
from scml.oneshot.rl.observation import FlexibleObservationManager
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.context import GeneralContext, SupplierContext, ConsumerContext
from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm

TrainingAlgorithm: type[BaseAlgorithm] = A2C
"""The algorithm used for training. You can use any stable_baselines3 algorithm or develop your own"""

MODEL_PATH = Path(__file__).parent / "models" / "mymodel"
"""The path in which train.py saves the trained model and from which litaagent_y.py loads it"""


def make_context(as_supplier: bool) -> GeneralContext:
    """Generates a context as a supplier or as a consumer"""
    if as_supplier:
        return SupplierContext()

    return ConsumerContext()


class MyObservationManager(FlexibleObservationManager):
    """This is my observation manager implementing encoding and decoding the state used by the RL algorithm"""

    def make_space(self) -> spaces.MultiDiscrete | spaces.Box:
        """Creates the observation space"""
        return super().make_space()

    def encode(self, awi: OneShotAWI) -> np.ndarray:
        """Encodes an observation from the agent's state"""
        return super().encode(awi)

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return super().make_first_observation(awi)

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, Outcome | None]:
        """Gets the offers from an encoded state"""
        return super().get_offers(awi, encoded)


# ensure that the folder containing models is created
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
