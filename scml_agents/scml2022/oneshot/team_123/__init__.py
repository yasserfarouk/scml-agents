from .neko import *

MAIN_AGENT = Neko
__all__ = neko.__all__

__author__ = "Ryota Arakawa"
__team__ = "Team 123"
__email__ = "arakawa@katfuji.lab.tuat.ac.jp"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20143
NAME = 'Neko'
CLASS_NAME = 'Neko'
VERSION = '1.0.0'
TEAM = 'Team 123'
AUTHOR = 'Ryota Arakawa'
MEMBERS = [{'name': 'Ryota Arakawa', 'institution': 'Tokyo University of Agriculture and Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Tokyo University of Agriculture and Technology'
TAGS = []
USES_LLM = False
DESC = "This agent changes the unit prices of its offers depending on the outcome of the negotiations on the previous day. If this agent made agreements with two or more agents, it will be too cooperative. Therefore, it will be aggressive to them by offering better unit prices the next day. On the other hand, if this agent couldn't make any agreements, it will be too aggressive. Therefore, this agent makes offers with the worse unit prices the next day. Also, the unit prices will be limited by the trading price depending on the market trend. The quantity of its offers is decided based on the previous opponent's offers to make an agreement easier."
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
