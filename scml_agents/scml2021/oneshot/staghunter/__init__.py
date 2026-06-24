# -*- coding: utf-8 -*-
from .myagent import *
from .myagent2 import *

MAIN_AGENT = StagHunterV7
__all__ = myagent.__all__ + myagent2.__all__

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = None
NAME = 'StagHunterV7'
CLASS_NAME = 'StagHunterV7'
VERSION = ''
TEAM = 'StagHunter'
AUTHOR = 'Zun Li'
MEMBERS = [{'name': 'Zun Li', 'institution': 'University of Michigan', 'country': 'United States'}, {'name': 'Michael P. Wellman', 'institution': 'University of Michigan', 'country': 'United States'}]
COUNTRY = 'United States'
INSTITUTION = 'University of Michigan'
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
