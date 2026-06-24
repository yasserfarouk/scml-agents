# -*- coding: utf-8 -*-
from .agent97 import *
from .learning_agent import *

MAIN_AGENT = Agent97
__all__ = agent97.__all__ + learning_agent.__all__

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = None
NAME = 'Agent97'
CLASS_NAME = 'Agent97'
VERSION = ''
TEAM = 'Team 72'
AUTHOR = 'Kayama'
MEMBERS = [{'name': 'Kayama', 'institution': 'Kyoto University', 'country': 'China'}]
COUNTRY = 'China'
INSTITUTION = 'Kyoto University'
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
