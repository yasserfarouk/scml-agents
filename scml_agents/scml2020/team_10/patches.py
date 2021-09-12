import os
import sys

sys.path.append(os.path.dirname(__file__))

# should patch @property cancelled_contracts in SCML2020World
@property
def cancelled_contracts(self):
    return list(_ for _ in self.saved_contracts if _.get("signed_at", -1) < 0)
