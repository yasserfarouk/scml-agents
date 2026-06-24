from .suzuka_agent import *

MAIN_AGENT = SuzukaAgent
__all__ = suzuka_agent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20198
NAME = 'SuzukaAgent'
CLASS_NAME = 'SuzukaAgent'
VERSION = '3.11.8'
TEAM = 'Team ABC'
AUTHOR = 'Hisakawa Soto'
MEMBERS = [{'name': 'Hisakawa Soto', 'institution': 'Kyushu University', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Kyushu University'
TAGS = ['Reinforcement Learning']
USES_LLM = False
DESC = 'SuzukaAgent always proposes the best unit price. However, upon acceptance, it adjusts the degree\r\nof compromise on the unit price based on the advantages or disadvantages of the situation. We\r\nconsider the following three situations as disadvantageous.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
