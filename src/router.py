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
        def sub_method(i):
            return self.cfmms[i].find_arb(v)
        threads = []
        with ThreadPoolExecutor as executor:
            threads = list(executor.map(sub_method, range(len(self.cfmms))))
        self.deltain, self.deltaout = zip(*threads)
        return self.deltain, self.deltaout
    
    def route(self, v=None):
        # route trrade to each cfmms
        k = 0
 