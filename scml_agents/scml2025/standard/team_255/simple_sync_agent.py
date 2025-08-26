from .simple_agent import SimpleAgent
from scml.std.agent import StdSyncAgent
from negmas import SAOResponse, ResponseType


class SimpleSyncAgent(StdSyncAgent, SimpleAgent):
    """An agent that distributes its needs over its partners randomly."""

    def first_proposals(self):
        """Decide a first proposal on every negotiation.
        Returning None for a negotiation means ending it."""
        return {
            partner: SAOResponse(
                ResponseType.REJECT_OFFER, SimpleAgent.propose(self, partner, state)
            )
            for partner, state in self.awi.current_states.items()
        }

    def counter_all(self, offers, states) -> dict:
        """Respond to a set of offers given the negotiation state of each."""
        # find all responses
        responses = {k: SimpleAgent.respond(self, k, s) for k, s in states.items()}
        # find counter offers for rejected offers
        myoffers = {
            k: SimpleAgent.propose(self, k, s)
            for k, s in states.items()
            if responses[k] == ResponseType.REJECT_OFFER
        }
        # compile final responses
        return {
            k: SAOResponse(
                responses[k],
                myoffers.get(
                    k, offers[k] if responses[k] == ResponseType.ACCEPT_OFFER else None
                ),
            )
            for k in states.keys()
        }

    # needed for the way multiple inheritence work in python.
    # we only need these because we inherit from `SimpleAgent`.
    # future agents will not need these two functions.
    def propose(self, negotiator_id, state):
        return StdSyncAgent.propose(self, negotiator_id, state)

    def respond(self, negotiator_id, state, source=""):
        return StdSyncAgent.respond(self, negotiator_id, state, source)
