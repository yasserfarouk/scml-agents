#!/usr/bin/env python
"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors* 
- Tarek Medhat : tarek.mohamed.medhat@gmail.com

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.
"""
from __future__ import annotations

# required for typing
from typing import Any


# required for development
from scml.oneshot import OneShotAWI, OneShotSyncAgent



from scml.oneshot import *
from negmas import (
    ConstUtilityFunction,
    Contract,
    ControlledNegotiator,
    ControlledSAONegotiator,
    Entity,
    Issue,
    Outcome,
    ResponseType,
    SAOController,
    SAOResponse,
    SAOSingleAgreementController,
    SAOState,
    SAOSyncController,
)


# required for typing
from negmas import Contract, Outcome, SAOResponse, SAOState



import copy
import sys


class PenaltyAvoidAgent(OneShotSyncAgent):
    """
    This is the only class you *need* to implement. The current skeleton has a
    basic do-nothing implementation.
    You can modify any parts of it as you need. You can act in the world by
    calling methods in the agent-world-interface instantiated as `self.awi`
    in your agent. See the documentation for more details


    **Please change the name of this class to match your agent name**

    Remarks:
        - You can get a list of partner IDs using `self.negotiators.keys()`. This will
          always match `self.awi.my_suppliers` (buyer) or `self.awi.my_consumers` (seller).
        - You can get a dict mapping partner IDs to `NegotiationInfo` (including their NMIs)
          using `self.negotiators`. This will include negotiations currently still running
          and concluded negotiations for this day. You can limit the dict to negotiations
          currently running only using `self.active_negotiators`
        - You can access your ufun using `self.ufun` (See `OneShotUFun` in the docs for more details).
    """

    # =====================
    # Negotiation Callbacks
    # =====================

    def first_proposals(self) -> dict[str, Outcome | None]:
        """
        Decide a first proposal for every partner.

        Remarks:
            - During this call, self.active_negotiators and self.negotiators will return the same dict
            - The negotiation issues will ALWAYS be the same for all negotiations running concurrently.
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """

        # Determine if i am seller (agent in first level) or buyer (agent in second level)
        # seller=self.awi.is_first_level

        # get partners
        parteners=self.negotiators.keys()
        
        # number of partners
        n_parteners=len(parteners)

        # current step
        s=self.awi.current_step

        # quntity per agent
        quantity_per_agent=self.remaining_quantity//n_parteners

        # quantity remainder
        quantity_remainder=self.remaining_quantity%n_parteners


        # construct equal proposals
        proposals=dict()
        for partener in parteners:
            proposals.update({partener:(quantity_per_agent+(1 if quantity_remainder>0 else 0),s,self.price)})
            quantity_remainder-=1
        


        return proposals

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        """
        Decide how to respond to every partner with which negotiations are still running.

        Remarks:
            - Returning an empty dictionary is the same as ending all negotiations immediately.
        """

        # get partners
        parteners=offers.keys()
        
        # number of partners
        n_parteners=len(parteners)


        # construct equal proposals
        proposals=dict()

        # sort offers by quantity offered in descending order

        all_offers=[[partner_id,offers[partner_id]] for partner_id in offers.keys()]

        all_offers.sort(key=lambda x:x[1][QUANTITY],reverse=True)

        all_qunatities=[o[1][QUANTITY] for o in all_offers]

        total_offered_quantity=sum(all_qunatities)



        # if total offered quantity is less than remaining quantity, reject all offers with counter offer equal to the original offer but with price equal to my price and quantity proportional to the offered quantity

        if total_offered_quantity < self.remaining_quantity:
            remainder_quant=self.remaining_quantity%total_offered_quantity
            remainder_quant_per_agent=remainder_quant//len(all_offers)+remainder_quant%len(all_offers)
            qunatity_multiplier=self.remaining_quantity//total_offered_quantity
            for partner_id,o in all_offers:
                counter_offer=list(copy.deepcopy(o))
                counter_offer[TIME]=self.awi.current_step
                counter_offer[UNIT_PRICE]=self.price
                counter_offer[QUANTITY]=o[QUANTITY]*qunatity_multiplier
                counter_offer[QUANTITY]+=min(remainder_quant_per_agent,remainder_quant) if remainder_quant>0 else 0
                remainder_quant-=min(remainder_quant_per_agent,remainder_quant)
                
                proposals.update({partner_id:SAOResponse(ResponseType.REJECT_OFFER,tuple(counter_offer))})
            return proposals




        # separate offers with price equal to my price (preferred) and those with different price 
        best_offers=[o for o in all_offers if o[1][UNIT_PRICE]==self.price]
        remaining_offers=[o for o in all_offers if o[1][UNIT_PRICE]!=self.price]


        # buffer for remaining quantity to be fulfilled after accepting best offers - solves conflicts between following for loop and on_negotiation_success feedback -
        rem_quant=self.remaining_quantity

        # buffer for indices of accepted offers to be removed from best_offers list before countering remaining offers
        accepted_indx=[]

        # accept best offers first
        for indx in range(len(best_offers)):
            partner_id,o=best_offers[indx]
            if rem_quant>=o[QUANTITY]:
                accepted_offer=list(copy.deepcopy(o))
                accepted_offer[TIME]=self.awi.current_step
                proposals.update({partner_id:SAOResponse(ResponseType.ACCEPT_OFFER,tuple(accepted_offer))})
                rem_quant-=accepted_offer[QUANTITY]
                n_parteners-=1
                accepted_indx.insert(0,indx)

            if rem_quant <= 0:
                break

        # if reuired quantity is fulfilled after accepting best offers, reject remaining offers by dont countering them.
        if rem_quant<=0 :
            return proposals
        
        # remove accepted offers from best_offers list before countering remaining offers
        for indx in accepted_indx:
            best_offers.pop(indx)
            
        # counter best offers that exceed remaining quantity if there is still quantity to be fulfilled
        if len(best_offers)>0:
            best_offers_quantities=[o[1][QUANTITY] for o in best_offers]
            total_best_offers_quantities=sum(best_offers_quantities)
            
            remainder_quant=rem_quant%total_best_offers_quantities
            remainder_quant_per_agent=remainder_quant//len(best_offers)+remainder_quant%len(best_offers)
            qunatity_multiplier=rem_quant//total_best_offers_quantities

            for partner_id,o in best_offers:
                if o[UNIT_PRICE]==self.price:
                    counter_offer=list(copy.deepcopy(o))
                    counter_offer[TIME]=self.awi.current_step
                    counter_offer[UNIT_PRICE]=self.price
                    counter_offer[QUANTITY]=o[QUANTITY]*qunatity_multiplier
                    counter_offer[QUANTITY]+=min(remainder_quant_per_agent,remainder_quant) if remainder_quant>0 else 0
                    remainder_quant-=min(remainder_quant_per_agent,remainder_quant)
                    
                    proposals.update({partner_id:SAOResponse(ResponseType.REJECT_OFFER,tuple(counter_offer))})
            return proposals





        # accept suitable offers with quantity less than or equal to remaining quantity

        # accepted offers indices buffer to be removed from remaining_offers list before countering remaining offers
        accepted_indx=[]

        for indx in range(len(remaining_offers)):
            partner_id,o=remaining_offers[indx]
            if rem_quant>=o[QUANTITY]:
                accepted_offer=list(copy.deepcopy(o))
                accepted_offer[TIME]=self.awi.current_step
                proposals.update({partner_id:SAOResponse(ResponseType.ACCEPT_OFFER,tuple(accepted_offer))})
                rem_quant-=accepted_offer[QUANTITY]
                n_parteners-=1
                accepted_indx.insert(0,indx)
                

            if rem_quant <= 0:
                break

        # if reuired quantity is fulfilled after accepting suitable offers, reject remaining offers by dont countering them.
        if rem_quant<=0 :
            return proposals

        
        # remove accepted offers from remaining_offers list before countering remaining offers
        for indx in accepted_indx:
            remaining_offers.pop(indx)


        # counter remaining offers if there is still quantity to be fulfilled

        remaining_offers_quantity=[o[1][QUANTITY] for o in remaining_offers]

        total_remaining_offered_quantity=sum(remaining_offers_quantity)

        if total_remaining_offered_quantity==0:
            return proposals
        

        # calculate counter offer quantity for each remaining offer based on the proportion of its quantity in the total remaining offered quantity and the remaining quantity to be fulfilled, then add a proportional part of the quantity remainder to each counter offer, and update the proposals dictionary with the counter offers
        remainder_quant=rem_quant%total_remaining_offered_quantity
        remainder_quant_per_agent=remainder_quant//len(all_offers)+remainder_quant%len(all_offers)
        qunatity_multiplier=rem_quant//total_remaining_offered_quantity

        for partner_id,o in remaining_offers:
            counter_offer=list(copy.deepcopy(o))
            counter_offer[TIME]=self.awi.current_step
            counter_offer[UNIT_PRICE]=self.price
            counter_offer[QUANTITY]=o[QUANTITY]*qunatity_multiplier
            counter_offer[QUANTITY]+=min(remainder_quant_per_agent,remainder_quant) if remainder_quant>0 else 0
            remainder_quant-=min(remainder_quant_per_agent,remainder_quant)
            
            proposals.update({partner_id:SAOResponse(ResponseType.REJECT_OFFER,tuple(counter_offer))})
        


        return proposals


    # =====================
    # Time-Driven Callbacks
    # =====================

    def init(self):
        """Called once after the agent-world interface is initialized"""

    def before_step(self):
        """Called at at the BEGINNING of every production step (day)"""
        # initiate parameters that will be used in counter_all and first_proposal function
        self.seller=self.awi.is_first_level
        self.remaining_quantity=self.awi.current_exogenous_input_quantity if self.seller else self.awi.current_exogenous_output_quantity
        self.issues= self.awi.current_output_issues if self.seller else self.awi.current_input_issues
        self.price=self.issues[UNIT_PRICE].max_value if self.seller else self.issues[UNIT_PRICE].min_value

    def step(self):
        """Called at at the END of every production step (day)"""
        # print('\nThere is quantity Diffrerence !!!!!!!!!!!! remaining_quantity=',self.remaining_quantity)

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(  # type: ignore
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: OneShotAWI,
        state: SAOState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without agreement"""

        # print('Negotiation Failure','x'*25)


    def on_negotiation_success(self, contract: Contract, mechanism: OneShotAWI) -> None:  # type: ignore
        """Called when a negotiation the agent is a party of ends with agreement"""
        # decrease remaining quantity by the quantity in the agreement
        self.remaining_quantity-= contract.agreement['quantity']
        # print('Negotiation Success !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        

