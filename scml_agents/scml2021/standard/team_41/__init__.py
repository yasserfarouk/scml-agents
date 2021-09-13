# -*- coding: utf-8 -*-
from .a import *
from .Augur_agent import *
from .sorcery import *

MAIN_AGENT = SorceryAgent
__all__ = a.__all__ + Augur_agent.__all__ + sorcery.__all__
