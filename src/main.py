from router import Router
from cfmms import ConstantProduct, GeometricMeanTwoToken
from objective import LinearNonnegative, BasketLiquidation, Swap
import numpy as np

## Parameters
Ai = np.array([0, 1])
fee = 0.003
gamma = 1 - fee
R = np.array([100, 100])
w = np.array([0.3, 0.7])

'''
Arbitrage Example. Sets up a router with two CFMMs, one CPMM and one G3M, along with the optimization routine for arbitrage trades.
Uses the above defined parameters.
'''
## Router Setup
p = np.array([3.0, 1.0])

CFMMs = []
CFMMs.append(ConstantProduct(Ai, R, gamma))
CFMMs.append(GeometricMeanTwoToken(Ai, R, gamma, w))
objective = LinearNonnegative(p)
router = Router(objective, CFMMs, len(p))

## Arbitrage!
print(CFMMs[0].get_price(), CFMMs[1].get_price())
deltain, deltaout = router.route()
for i in range (0,len(CFMMs)):
    CFMMs[i].update_reserves(deltain[i], deltaout[i])
print(CFMMs[0].get_price(), CFMMs[1].get_price())