from .katsudon_agent import *
from .optimistic_agent import *

MAIN_AGENT = KATSUDONAgent
__all__ = katsudon_agent.__all__ + optimistic_agent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20411
NAME = 'KATSUDON'
CLASS_NAME = 'KATSUDONAgent'
VERSION = '3.11.8'
TEAM = 'MISOKATSU-TEISYOKU'
AUTHOR = 'Yamadori Kohki'
MEMBERS = [{'name': 'Yamadori Kohki', 'institution': 'Nagoya Institute of Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Nagoya Institute of Technology'
TAGS = ['Reinforcement Learning', 'Psychology']
USES_LLM = False
DESC = 'Title: KATSUDON: An Adaptive Concession-Based Agent with Heuristic Deadline Awareness and Utility Structuring\r\n\r\nKATSUDON is a negotiation agent designed to perform robustly in diverse negotiation domains by dynamically adjusting its concession strategy based on time, opponent behavior, and utility structure. The agent employs a multi-stage tactic:\r\n\r\nInitial phase: it proposes high-utility bids close to its own preference using a similarity-augmented sampling mechanism.\r\nMid-phase: it gradually concedes based on a sigmoid decay function tuned by deadline proximity.\r\nFinal phase: it utilizes opponent modeling to estimate acceptable offers using frequency-based bid profiling and Nash product heuristics.\r\nKATSUDON uses an adaptive acceptance strategy that considers both the offer utility and potential future gains. It is designed to balance assertiveness and cooperativeness, leading to favorable outcomes across varied opponents and domains.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
