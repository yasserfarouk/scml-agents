# -*- coding: utf-8 -*-
from .godfather import *
from .godfather import (
    MinDisagreementGodfatherAgent,
    MinEmpiricalGodfatherAgent,
    AspirationUniformGodfatherAgent,
    NonconvergentGodfatherAgent,
    ParetoEmpiricalGodfatherAgent,
    SlowGoldfish,
    HardnosedGoldfishGodfatherAgent,
    HardnosedGoldfishBiggerAgent,
    HardnosedGoldfishSmallerAgent,
    SoftnosedGoldfishGodfatherAgent,
    QuickLearningGodfather,
    MediumLearningGodfather,
    SlowLearningGodfather,
    ZooGodfather,
    TrainingCollectionGodfatherAgent,
    ChristopherTheGoldfishAgent,
)


MAIN_AGENT = GoldfishParetoEmpiricalGodfatherAgent

EXTRA_AGENTS = [
    MinDisagreementGodfatherAgent,
    MinEmpiricalGodfatherAgent,
    AspirationUniformGodfatherAgent,
    NonconvergentGodfatherAgent,
    ParetoEmpiricalGodfatherAgent,
    SlowGoldfish,
    HardnosedGoldfishGodfatherAgent,
    HardnosedGoldfishBiggerAgent,
    HardnosedGoldfishSmallerAgent,
    SoftnosedGoldfishGodfatherAgent,
    QuickLearningGodfather,
    MediumLearningGodfather,
    SlowLearningGodfather,
    ZooGodfather,
    TrainingCollectionGodfatherAgent,
    ChristopherTheGoldfishAgent,
]

__all__ = godfather.__all__
