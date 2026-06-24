# -*- coding: utf-8 -*-
from .a import *
from .Augur_agent import *
from .sorcery import *

MAIN_AGENT = SorceryAgent
__all__ = a.__all__ + Augur_agent.__all__ + sorcery.__all__

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = None
NAME = 'SorceryAgent'
CLASS_NAME = 'SorceryAgent'
VERSION = ''
TEAM = 'Team 41'
AUTHOR = 'Takemoto Midori'
MEMBERS = [{'name': 'Takemoto Midori', 'institution': 'Nagoya Institute of Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Nagoya Institute of Technology'
TAGS = []
USES_LLM = False
DESC = ''
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
