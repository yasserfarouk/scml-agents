# -*- coding: utf-8 -*-
from .godfather import *
from .godfather import EXTRA_AGENTS

MAIN_AGENT = GoldfishParetoEmpiricalGodfatherAgent

EXTRA_AGENTS = godfather.EXTRA_AGENTS

__all__ = godfather.__all__

__author__ = "Jackson de Campos, Ben Fiske, Chris Mascioli, Amy Greenwald"
__team__ = "Team Corleone"
__email__ = "amy_greenwald@brown.edu"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20035
NAME = 'GoldfishParetoEmpiricalGodfatherAgent'
CLASS_NAME = 'GoldfishParetoEmpiricalGodfatherAgent'
VERSION = '0.0.2'
TEAM = 'Team Corleone'
AUTHOR = 'Jackson de Campos'
MEMBERS = [{'name': 'Jackson de Campos', 'institution': 'Brown University', 'country': 'United States'}, {'name': 'Benjamin Fiske', 'institution': 'Brown University', 'country': 'United States'}, {'name': 'Chris Mascioli', 'institution': 'Brown University', 'country': 'United States'}, {'name': 'Amy Greenwald', 'institution': 'Brown University', 'country': 'United States'}]
COUNTRY = 'United States'
INSTITUTION = 'Brown University'
TAGS = ['Supervised Learning', 'Optimization']
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
