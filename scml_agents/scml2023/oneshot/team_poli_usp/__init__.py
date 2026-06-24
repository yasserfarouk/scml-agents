from .quantity_oriented_agent import *

MAIN_AGENT = QuantityOrientedAgent
__all__ = quantity_oriented_agent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20166
NAME = 'QuantityOrientedAgent'
CLASS_NAME = 'QuantityOrientedAgent'
VERSION = '0.0.2'
TEAM = 'Team Poli-USP'
AUTHOR = 'Pedro Hrosz Turini'
MEMBERS = [{'name': 'Pedro Hrosz Turini', 'institution': 'University of São Paulo', 'country': 'Brazil'}]
COUNTRY = 'Brazil'
INSTITUTION = 'University of São Paulo'
TAGS = []
USES_LLM = False
DESC = "A simple OneShot agent that tries to match the exact amount specified by the exogenous contract, orienting its responses based on quantity and not price. The agent's proposals are made in order to mimimize the chances of rejection based on quantity, and changes according to the simulation step and current needs."
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
