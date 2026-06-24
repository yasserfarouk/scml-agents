from .fairt4t import *

MAIN_AGENT = FairT4T
__all__ = fairt4t.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20214
NAME = 'FairT4T'
CLASS_NAME = 'FairT4T'
VERSION = '3.12.2'
TEAM = 'Team 172'
AUTHOR = 'Merve Dogan'
MEMBERS = [{'name': 'Merve Dogan', 'institution': 'Özyeğin University', 'country': 'Türkiye'}]
COUNTRY = 'Türkiye'
INSTITUTION = 'Özyeğin University'
TAGS = ['Reinforcement Learning', 'Bayesian Methods', 'Psychology']
USES_LLM = False
DESC = 'We implemented our agent by following the implementa- tion of BetterSyncAgent which is an extension of OneShotSyncAgent,\r\nIn the original BetterSyncAgent, distribution of the quantities is just random. In addition, we incorporate Nice Tit for Tat strategy to mimic the opponent’s behavior to some extent. We calculate the utility changes in agent’s opponent’s subsequent offers regarding its utility by keeping a history of opponents’ last bids. The positive changes in utility mean that the opponent concedes; hence, the agent should concede as well. Our agent generates an offer whose utility is closest to the estimated target utility. Hence, by iterating through the partners, the agent checks if a partner has made any offer yet.  If so, the agent mirrors their bid. Otherwise, it makes a new offer: A tuple (q, s, p) where q represents the quantity, s represents the step value, and p represents the price value. \r\nWhen receiving offers in counter-all, we employ ACnext strategy in which case the agent checks whether the utility from these subsets is greater than the utility of the agent’s next offer. If so, it accepts that subset of offers, otherwise it makes a new offer. \r\nAlso in counter-all method, we keep the quantity offered by the partners, as ”opponents last bid” to be used in our cooperative moves to mimic the opponent’s behavior. We also keep the utilities for the offers accepted in this step in ”utility opponent previous offers”, so we can use it for the next calculation of the target utility for our tit-for tat strategy as mentioned above.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
