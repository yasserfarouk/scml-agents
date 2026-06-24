from .price_trend import *

MAIN_AGENT = PriceTrendStdAgent
__all__ = price_trend.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20686
NAME = 'PriceTrendStdAgent'
CLASS_NAME = 'PriceTrendStdAgent'
VERSION = '3.11.9'
TEAM = 'Team 280'
AUTHOR = 'Kawasaki Yuta'
MEMBERS = [{'name': 'Kawasaki Yuta', 'institution': 'Nagoya Institute of Technology', 'country': 'Japan'}, {'name': 'RYUTA SHIRAISHI', 'institution': 'Nagoya Institute of Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Nagoya Institute of Technology'
TAGS = []
USES_LLM = False
DESC = 'PriceTrendStdAgent is an enhanced version of a baseline negotiation agent, with two major improvements. First, it evaluates partner scores based on both deal quantity and price fairness, using exponentially smoothed metrics. These scores guide concentrated proposal distributions toward more trustworthy partners. Second, it records input market price trends and applies exponential moving average smoothing to detect rising or falling patterns. This trend information is used to adjust offer quantities and timing, as well as to decide whether to accept or reject proposals. Through these enhancements, the agent strategically balances market conditions and opponent reliability to maximize its performance.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
