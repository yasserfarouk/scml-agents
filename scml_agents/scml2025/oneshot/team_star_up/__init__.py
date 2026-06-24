from .agent import *

MAIN_AGENT = HoriYamaAgent
__all__ = agent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20438
NAME = 'HoriYamaAgent'
CLASS_NAME = 'HoriYamaAgent'
VERSION = '3.11.8'
TEAM = 'Team STAR UP'
AUTHOR = 'Hayaki Horinouchi'
MEMBERS = [{'name': 'HAYAKI HORINOUCHI', 'institution': 'Kyoto University', 'country': 'Japan'}, {'name': 'RINTARO TOYAMA', 'institution': 'Kyoto University', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Kyoto University'
TAGS = ['Reinforcement Learning', 'Psychology']
USES_LLM = False
DESC = 'HoriYamaAgent enhances the BetterSyncAgent by incorporating market-aware negotiation strategies. It dynamically adjusts offer behavior based on both negotiation time and real-time supply-demand conditions. As a seller or buyer, the agent calculates a market factor from the ratio of counterpart agents and uses it to modulate thresholds for offer acceptance. This prevents over-reliance on time-based tolerances and reduces randomness in pricing. It also considers remaining needs, placing offers slightly above calculated demand to ensure contract fulfillment. These improvements aim to make HoriYamaAgent more resilient and strategic in SCML-OneShot simulations.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
