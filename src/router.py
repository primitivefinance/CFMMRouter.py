import numpy as np
import cvxpy as cvx
from concurrent.futures import ThreadPoolExecutor
from cfmms import zerotrade

class Router:
    def __init__(self, objective, cfmms, number_of_tokens):
       self.objective = objective
       self.cfmms = cfmms
       self.deltain = [zerotrade(c)[0] for c in cfmms]
       self.deltaout = [zerotrade(c)[1] for c in cfmms]
       self.v = np.zeros(number_of_tokens)
    
    def find_arb(self, v):
        # call find_arb on each cfmms
        k = 0
    
    def route(self, v=None):
        # route trrade to each cfmms
        k = 0
 