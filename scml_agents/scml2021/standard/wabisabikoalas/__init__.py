# -*- coding: utf-8 -*-
from .artisan_kangaroo import *

MAIN_AGENT = ArtisanKangaroo
__all__ = artisan_kangaroo.__all__

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20008
NAME = 'ArtisanKangaroo'
CLASS_NAME = 'ArtisanKangaroo'
VERSION = '1.0.0'
TEAM = 'WabiSabiKoalas'
AUTHOR = 'Koki Katagiri'
MEMBERS = [{'name': 'Koki Katagiri', 'institution': 'Nagoya Institute of Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Nagoya Institute of Technology'
TAGS = []
USES_LLM = False
DESC = 'Main concept of my agent is not using utility function in negotiations. The reason is that I think you can’t decide equivalent values among 3 parameters of issues. Even if the selling offer my agent received has a very high cost and utility value, it will end up with a breach unless it has enough input products and production capacity before its delivery time. Therefore, my agent will accept offers only if they satisfy some requirement not to suffer losses.\r\nAnother key feature is keeping track of every contract my agent signed. For example, which input contract will correspond to other output contracts. This feature allows my agent to recognize the situation about contracts. Moreover, it will lead to prevent from signing too many contracts.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
