from .charlies import *

MAIN_AGENT = CharliesAgentCollusion
__all__ = charlies.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20080
NAME = 'BossAgent'
CLASS_NAME = 'BossAgent'
VERSION = '0.0.7'
TEAM = 'BossAgent'
AUTHOR = 'Umut Çakan'
MEMBERS = [{'name': 'Umut Cakan', 'institution': 'Özyeğin University', 'country': 'Türkiye'}, {'name': 'Mehmet Onur Keskin', 'institution': 'Özyeğin University', 'country': 'Türkiye'}, {'name': 'Gevher Yesevi', 'institution': 'Özyeğin University', 'country': 'Türkiye'}, {'name': 'Reyhan Aydogan', 'institution': 'Özyeğin University', 'country': 'Türkiye'}, {'name': 'Amy Greenwald', 'institution': 'Özyeğin University', 'country': 'Türkiye'}, {'name': 'Umut Çakan', 'institution': 'Özyeğin University', 'country': 'Türkiye'}]
COUNTRY = 'Türkiye'
INSTITUTION = 'Özyeğin University'
TAGS = ['Supervised Learning']
USES_LLM = False
DESC = 'Our negotiating agent, namely Charlie’s agent, has five main modules that manage his daily interactions with other agents and obtain a profitable trade. These five modules are ‘Business Planner’, ‘3D Dispatcher’, ‘Production Scheduler’, ‘Trading Strategy’ and ‘Analytics’ modules. Business Planner is responsible for managing the pre-negotiation settings (like defining the boundaries of proposals, selecting partners). The dispatcher tries to match buyer and seller contracts for all three issues simultaneously with the objective of signing profitable and deliverable agreements. The available production slots are constantly checked during the negotiation with the issues of the offers. According to the issues of the synchronized offers, the dispatcher labels different matching circumstances to decide on the negotiation strategy for each agent it negotiates. Besides the labels based on matching situations, Charlie’s agent also considers some cases for channeling its negotiation approaches, such as interacting with a partner that has a high number of unsigned contracts or high bankruptcy probability. It also shows adaptive behavior when it negotiates in a monopoly or volatile market. The trading strategy handles the signing, execution, and bankrupt processes. It signs the contracts that are feasible for the dispatcher and fits into the boundaries that are taken from the planner. In case of having a partner that bankrupts, it tries to find new buyers or sellers for the canceled contracts. The production scheduler schedules the contract once it is signed. The analytics module is responsible for the predictions used for deciding the proposal boundaries and foreseeing risky agreements. Overall, these modules constitute a comprehensive agent that can profitably manage its daily actions. Charlie’s agent obtains higher signing rates when it interacts with an agent that shows similar behavior. Therefore, this is advantageous for the Collusion Track.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
