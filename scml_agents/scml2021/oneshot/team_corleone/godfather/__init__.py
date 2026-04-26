# -*- coding: utf-8 -*-
from .godfather import *
from .godfather import (
    AspirationUniformGodfatherAgent,
    ChristopherTheGoldfishAgent,
    HardnosedGoldfishBiggerAgent,
    HardnosedGoldfishGodfatherAgent,
    HardnosedGoldfishSmallerAgent,
    MediumLearningGodfather,
    MinDisagreementGodfatherAgent,
    MinEmpiricalGodfatherAgent,
    NonconvergentGodfatherAgent,
    ParetoEmpiricalGodfatherAgent,
    QuickLearningGodfather,
    SlowGoldfish,
    SlowLearningGodfather,
    SoftnosedGoldfishGodfatherAgent,
    TrainingCollectionGodfatherAgent,
    ZooGodfather,
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
