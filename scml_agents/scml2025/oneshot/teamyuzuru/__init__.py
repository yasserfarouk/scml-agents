from .agent import *

MAIN_AGENT = CostAverseAgent
__all__ = agent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20221
NAME = 'CostAverseAgent'
CLASS_NAME = 'CostAverseAgent'
VERSION = '3.11.8'
TEAM = 'Team179'
AUTHOR = 'Yuzuru Kitamura'
MEMBERS = [{'name': 'Yuzuru Kitamura', 'institution': 'Tokyo University of Agriculture and Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Tokyo University of Agriculture and Technology'
TAGS = []
USES_LLM = False
DESC = 'For the given "BetterSyncAgent", I have made improvements to the strategy of acception. Furthermore, as a change for 2025, an acceptance strategy that takes into account the ratio of disposal costs to shorfall penalty has been adopted.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
