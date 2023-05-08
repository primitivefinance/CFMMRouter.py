from router import Router
from cfmms import ConstantProduct, GeometricMeanTwoToken
from objective import LinearNonnegative, BasketLiquidation, Swap
import numpy as np

## Parameters
Ai = np.array([0, 1])
fee = 0.003
gamma = 1 - fee
R = np.array([100, 100])
w = np.array([1, 2.5])

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
router = Router(objective, CFMMs, len(p), p)

## Arbitrage!
print("Uni V2 pool price before trade: ", CFMMs[0].get_price())
print("G3M pool price before trade: ", CFMMs[1].get_price())
deltain, deltaout = router.route()
for i in range (0,len(CFMMs)):
    CFMMs[i].R = CFMMs[i].R + CFMMs[i].gamma * deltain[i].value - deltaout[i].value

delta_in = []
delta_out = []
for i in range(len(CFMMs)):
    delta_in.append(deltain[i].value)
    delta_out.append(deltaout[i].value)
print("Arbitrage trade input: ", delta_in)
print("Arbitrage trade output: ", delta_out)
print("Uni V2 pool price after trade: ", CFMMs[0].get_price())
print("G3M pool price after trade", CFMMs[1].get_price())