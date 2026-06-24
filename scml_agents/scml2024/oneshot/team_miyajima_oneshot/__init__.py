from .cautious import *

MAIN_AGENT = CautiousOneShotAgent
__all__ = cautious.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20167
NAME = 'CautiousOneShotAgent'
CLASS_NAME = 'CautiousOneShotAgent'
VERSION = '3.11.1'
TEAM = 'Team Miyajima OneShot'
AUTHOR = 'Ryoga Miyajima'
MEMBERS = [{'name': 'Ryoga Miyajima', 'institution': 'Tokyo University of Agriculture and Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Tokyo University of Agriculture and Technology'
TAGS = []
USES_LLM = False
DESC = 'CautiousOneShotAgent is an improved SyncRandomOneShotAgent to achieve better agreements. SyncRandomOneShotAgent is a sample agent for SCML2024 OneShot Track, which negotiates multiple partners synchronously to meet the required quantity. It is generally a good strategy because quantity is a much more important issue than price in SCML2024 OneShot Track. However, it still leaves some room for improvement, so I improved it to be more efficient and lower risk.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
