from .cautious import *

MAIN_AGENT = CautiousStdAgent
__all__ = cautious.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20158
NAME = 'CautiousStdAgent'
CLASS_NAME = 'CautiousStdAgent'
VERSION = '3.11.1'
TEAM = 'Team Miyajima Std'
AUTHOR = 'Ryoga Miyajima'
MEMBERS = [{'name': 'Ryoga Miyajima', 'institution': 'Tokyo University of Agriculture and Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Tokyo University of Agriculture and Technology'
TAGS = []
USES_LLM = False
DESC = 'CautiousStdAgent is the agent for SCML2024 Standard Track. It adopts a cautious strategy focused on inventory control. It aims to buy materials as inexpensively as possible and sell out of inventory as soon as possible every day.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
