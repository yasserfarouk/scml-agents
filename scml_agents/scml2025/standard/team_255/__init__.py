from .myagent import *
from .proactive_agent import *

MAIN_AGENT = PonponAgent
__all__ = myagent.__all__ + proactive_agent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20387
NAME = 'MyAgent'
CLASS_NAME = 'MyAgent'
VERSION = '3.11.8'
TEAM = 'Team 255'
AUTHOR = 'Kento Fukuda'
MEMBERS = [{'name': 'Kento Fukuda', 'institution': 'Nagoya Institute of Technology', 'country': 'Japan'}]
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
