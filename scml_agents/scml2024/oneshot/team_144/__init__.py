from .cpdist_agent import *

MAIN_AGENT = CPDistAgent
__all__ = cpdist_agent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20163
NAME = 'CPDistAgent'
CLASS_NAME = 'CPDistAgent'
VERSION = '3.11.8'
TEAM = 'Team 144'
AUTHOR = 'Shota Kimata'
MEMBERS = [{'name': 'Shota Kimata', 'institution': 'Nagoya Institute of Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Nagoya Institute of Technology'
TAGS = []
USES_LLM = False
DESC = 'CPDistAgent is an agent that can make "cooperative proposals" and "distributions". The strategy is based on the base agent\'s strategy of dividing the quantity in advance and assigning them to the opponent agents. In addition to this strategy, we adopt a "cooperative strategy" that takes into account the proposal quantity from the negotiation partner. By using this cooperative strategy, it is possible to propose a quantity that the other party is more likely to accept, thus increasing the chances of a successful negotiation.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
