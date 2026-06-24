#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:09:59 2020

@author: jtsatsaros
"""

from .myagent import *

MAIN_AGENT = MontyHall
__all__ = myagent.__all__

__author__ = ""
__team__ = ""
__email__ = ""

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = None
NAME = 'MontyHall'
CLASS_NAME = 'MontyHall'
VERSION = ''
TEAM = 'Monty Hall'
AUTHOR = 'Monty Hall'
MEMBERS = [{'name': 'Monty Hall', 'institution': 'Brown University', 'country': 'United States'}]
COUNTRY = 'United States'
INSTITUTION = 'Brown University'
TAGS = []
USES_LLM = False
DESC = ''
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
