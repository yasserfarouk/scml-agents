# -*- coding: utf-8 -*-
from .oneshot_agents import *
from .past_agents import *

MAIN_AGENT = Gentle
__all__ = oneshot_agents.__all__ + past_agents.__all__

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20041
NAME = 'Gentle'
CLASS_NAME = 'Gentle'
VERSION = '0.6.1'
TEAM = 'Team 73'
AUTHOR = 'Takumu Shimizu'
MEMBERS = [{'name': 'Takumu Shimizu', 'institution': 'Tokyo University of Agriculture and Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Tokyo University of Agriculture and Technology'
TAGS = []
USES_LLM = False
DESC = 'Gentle inherits AdaptiveAgent and negotiates with opponents independently. Negotiation choices are the same way of the inheritance class, and Gentle continues to negotiate with all opponents until its needs are fulfilled. We don’t set utility function and the decision of Gentle is mainly based on the unit price. The offer and acceptance are determined from some negotiation information like the unit price of agreements. To deal with the risk of not being able to agree at all, Gentle makes concessional offers based on the self factor. The self factor represents how well Gentle is doing at a specific time. We tested Gentle in simulations against LearningAgent and AdaptiveAgent. As a result, Gentle outperforms other agents in many cases.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
