import numpy as np
import scipy.optimize as opt
from concurrent.futures import ThreadPoolExecutor
from cfmms import zerotrade

class Router:
    def __init__(self, objective, cfmms, number_of_tokens):
       self.objective = objective
       self.cfmms = cfmms
       self.deltain = [zerotrade(c)[0] for c in cfmms]
       self.deltaout = [zerotrade(c)[1] for c in cfmms]
       self.v = np.zeros(number_of_tokens) + 1e-8
    
    def find_arb(self, v):
        def sub_method(i):
            return self.cfmms[i].find_arb(v)
        threads = []
        with ThreadPoolExecutor() as executor:
            threads = list(executor.map(sub_method, range(len(self.cfmms))))
        self.deltain, self.deltaout = zip(*threads)
        print(self.deltain, self.deltaout)
        return self.deltain, self.deltaout
    
    def route(self, v=None):
        def fn(v):
            if not np.all(v == self.v):
                self.find_arb(v)
                self.v = v.copy()
            accumulator = 0.0
            for (i,j,k) in zip(self.deltain, self.deltaout, self.cfmms):
                accumulator += np.dot(self.deltaout, v[k.Ai]) - np.dot(self.deltain, v[k.Ai])
            return self.objective.f(v) + accumulator

        if v is None:
            v = self.v
        v = opt.minimize(fun=fn, x0=v, method='L-BFGS-B', jac=self.objective.grad, bounds=list(zip(self.objective.lower_limit(), self.objective.upper_limit())))
        self.v = v
        self.find_arb(self.v)
        return self.deltain, self.deltaout
 