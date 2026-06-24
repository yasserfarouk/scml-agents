from .mat import *

# __all__ = mat.__all__

MAIN_AGENT = MATAgent

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20520
NAME = 'MATAgent_Prelim'
CLASS_NAME = 'MATAgent'
VERSION = '3.11.1'
TEAM = 'ST'
AUTHOR = 'Tyrone Serapio'
MEMBERS = [{'name': 'Tyrone Serapio', 'institution': 'Brown University', 'country': 'United States'}]
COUNTRY = 'United States'
INSTITUTION = 'Brown University'
TAGS = ['Multi-Agent Reinforcement Learning', 'Game Theory']
USES_LLM = False
DESC = "We chose a CFR-based (Counterfactual Regret Minimization) approach for MATAgent, given CFR's suitability in extensive-form game-solving. In particular, our agent applies a tabular CFR to each bilateral negotiation in the OneShot framework. We discretize the continuous decision of ``how much and at what price” into a small finite set of actions, allowing it to learn a mixed strategy offline, and then simply sample from that strategy at the actual competition (runtime).\r\n\r\nOn top of this, we use a trust meta-agent that distributes needs to the different agents at run-time. We maintain a heuristic of trust, and at the same time, use an adaptive strategy at runtime that allows us to distribute needs at runtime and manage prices."
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
