from .sources import *

MAIN_AGENT = RLIndAgent
__all__ = sources.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20138
NAME = 'RLIndAgent'
CLASS_NAME = 'RLIndAgent'
VERSION = '0.0.1'
TEAM = 'Team 102'
AUTHOR = 'Takumu Shimizu'
MEMBERS = [{'name': 'Takumu Shimizu', 'institution': 'Tokyo University of Agriculture and Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Tokyo University of Agriculture and Technology'
TAGS = ['Deep Learning', 'Reinforcement Learning', 'Transfer / Meta Learning']
USES_LLM = False
DESC = 'In the SCML OneShot Track, few agents have utilized neural networks or machine learning, and none have been trained using reinforcement learning (RL). In this work, we propose two negotiation strategies based on RL. The first, RLAgent, negotiates with opponents independently and inherits the SimpleAgent. The second, RLSyncAgent, inherits the SyncAgent and is capable of receiving and responding to multiple opponent offers. To apply RL to the agents, we define the Markov Decision Process for the OneShot Track. The state consists of the current number of rounds, the current needs, and the opponent’s offer. The action consists of the accept signal, and the counter offer. The reward is the profit of the day. The definition is slight different RLAgent and RLSyncAgent to adjust their actions. We evaluate the trained agents and select RLAgent as the agent to submit to the competition.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
