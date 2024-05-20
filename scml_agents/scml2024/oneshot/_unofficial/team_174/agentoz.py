#!/usr/bin/env python3

from negmas import ResponseType, SAOState

from negmas.outcomes import Outcome
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotAgent
from negmas import ResponseType

__all__ = ["AgentOZ"]

class AgentOZ(OneShotAgent):
    """Based on OneShotAgent"""

    def __init__(self, *args, **kwargs):
        self.secured = 0
        self.max_negotiation_rounds = 100000.0
        self.average_negotiation_round_count = 10
        
        super().__init__(*args, **kwargs)
        
    def before_step(self):
        self.secured = 0
        
    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]
        
    def propose(self, negotiator_id: str, state: SAOState) -> Outcome | None:  
        if (state.step < self.average_negotiation_round_count):
            return self.best_offer(negotiator_id, state)
        else:
            return self.alternative_offer(negotiator_id, state)

    def best_offer(self, negotiator_id, state):
        my_needs = self._needed(negotiator_id)     
        if my_needs <= 0:
            return None    
        
        ami = self.get_nmi(negotiator_id)
           
        if not ami:
            return None
        
        offer = [-1] * 3
             
        offer[QUANTITY] = max(min(my_needs, ami.issues[QUANTITY].max_value), ami.issues[QUANTITY].min_value)
        offer[TIME] = self.awi.current_step
        
        if self._is_selling(ami):
            offer[UNIT_PRICE] = self._find_good_price_according_to_time_for_seller(ami, state, offer[QUANTITY])
        else:
            offer[UNIT_PRICE] = self._find_good_price_according_to_time_for_buyer(ami, state, offer[QUANTITY])
        
        return tuple(offer)
     
     
    def alternative_offer(self, negotiator_id, state):
        my_needs = self._needed(negotiator_id)     
        if my_needs <= 0:
            return None    
        
        ami = self.get_nmi(negotiator_id)
           
        if not ami:
            return None
        
        offer = [-1] * 3
             
        offer[QUANTITY] = max(min(my_needs, ami.issues[QUANTITY].max_value), ami.issues[QUANTITY].min_value)
        offer[TIME] = self.awi.current_step
        
        if self._is_selling(ami):
            offer[UNIT_PRICE] = self._find_good_price_according_to_time_for_buyer(ami, state, offer[QUANTITY])
        else:
            offer[UNIT_PRICE] = self._find_good_price_according_to_time_for_seller(ami, state, offer[QUANTITY])
            
        
        return tuple(offer)
    
    
    def respond(self, negotiator_id: str, state: SAOState, source: str) -> ResponseType:
        offer = state.current_offer
        
        if offer is None:
            return ResponseType.REJECT_OFFER
        
        my_needs = self._needed(negotiator_id)
        
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        elif my_needs == offer[QUANTITY]:
            return ResponseType.ACCEPT_OFFER
        elif offer[QUANTITY] < my_needs:
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER
        
    def _find_max_min_price(self, ami):
        min_price = ami.issues[UNIT_PRICE].min_value
        max_price = ami.issues[UNIT_PRICE].max_value
        return min_price, max_price
    
    def _find_good_price_according_to_time_for_seller(self, ami, state, offer_quantity):
        min_price, max_price = self._find_max_min_price(ami)
        return max_price - ((state.step * ((max_price - min_price) / self.max_negotiation_rounds))) 
            
    def _find_good_price_according_to_time_for_buyer(self, ami, state, offer_quantity):
        min_price, max_price = self._find_max_min_price(ami)
        return min_price + ((state.step * ((max_price - min_price) / self.max_negotiation_rounds)))         
        
    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )
        
    def _is_selling(self, nmi):
        return nmi.annotation["product"] == self.awi.my_output_product
    