from enum import Enum, auto


class NegotiationStatus(Enum):
    CONTINUING = auto()
    AGREED_IN_PARTNER_OPINION = auto()
    AGREED_IN_OWN_OPINION = auto()
    FAILED = auto()
