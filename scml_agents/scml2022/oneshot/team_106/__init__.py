from .moving_average_agent import *

MAIN_AGENT = AdamAgent
__all__ = moving_average_agent.__all__

__author__ = ""
__team__ = ""
__email__ = ""

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20108
NAME = 'LearningAverageAgent'
CLASS_NAME = 'LearningAverageAgent'
VERSION = '0.0.6'
TEAM = 'Team 106'
AUTHOR = 'Eran Hirsch'
MEMBERS = [{'name': 'Eran Hirsch', 'institution': 'Bar-Ilan University', 'country': 'Israel'}, {'name': 'Dorin Keshales', 'institution': 'Bar-Ilan University', 'country': 'Israel'}]
COUNTRY = 'Israel'
INSTITUTION = 'Bar-Ilan University'
TAGS = []
USES_LLM = False
DESC = 'Learning moving average based on the past agreements, finetuning alpha'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
