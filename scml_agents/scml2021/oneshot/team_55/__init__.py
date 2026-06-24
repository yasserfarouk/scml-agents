# -*- coding: utf-8 -*-
from .agent import *

# from .most_common_agent import *
from .worker_agents import *

MAIN_AGENT = Zilberan
__all__ = agent.__all__ + worker_agents.__all__

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = None
NAME = 'Zilberan'
CLASS_NAME = 'Zilberan'
VERSION = ''
TEAM = 'Team 55'
AUTHOR = 'Ofir Moshe'
MEMBERS = [{'name': 'Ofir Moshe', 'institution': 'Bar-Ilan University', 'country': 'Israel'}, {'name': 'Ido Azulay', 'institution': 'Bar-Ilan University', 'country': 'Israel'}]
COUNTRY = 'Israel'
INSTITUTION = 'Bar-Ilan University'
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
