from .agent import *

MAIN_AGENT = AnalysisAgent
__all__ = agent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20456
NAME = 'AnalysisAgent'
CLASS_NAME = 'AnalysisAgent'
VERSION = '3.12.7'
TEAM = 'Team 283'
AUTHOR = 'Eito Sugita'
MEMBERS = [{'name': 'Eito Sugita', 'institution': 'Nagoya Institute of Technology', 'country': 'Japan'}, {'name': 'Ryota Kaneko', 'institution': 'Nagoya Institute of Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Nagoya Institute of Technology'
TAGS = []
USES_LLM = False
DESC = 'In ANAC, there are situations where the required negotiation quantity of the negotiation partner is limited and the number of negotiation rounds is small, at 20 per step. Therefore, an effective distribution strategy involves negotiating with multiple partners while keeping the quantity negotiated with each partner relatively low.\r\nAnalysisAgent acquires and accumulates negotiation quantities upon successful negotiations, identifies partners who are easier to trade with and allocates more transactions to them.\r\nLast year\'s winning agent adopted a strategy of concentrating transactions on the partner with the highest number of successful transactions. However, AnalysisAgent considers multiple partners with high successful transaction quantities and distributes transactions accordingly.\r\nAdditionally, the agent increases its weight in the later steps by multiplying the accumulated transaction quantity by "current step" / "maximum number of steps", enabling it to adjust its behaviour based on the number of steps.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
