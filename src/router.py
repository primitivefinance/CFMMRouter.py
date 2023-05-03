import numpy as np
import scipy.optimize as opt
from concurrent.futures import ThreadPoolExecutor
from cfmms import zerotrade

class Router:
    def __init__(self, objective, cfmms, number_of_tokens):
       self.objective = objective
       self.cfmms = cfmms
       self.deltain = [zerotrade(c) for c in cfmms]
       self.deltaout = [zerotrade(c) for c in cfmms]
       self.v = np.zeros(number_of_tokens)
    
    def find_arb(self, v):
        def sub_method(i):
            return self.cfmms[i].find_arb(v)
        threads = []
        with ThreadPoolExecutor() as executor:
            threads = list(executor.map(sub_method, range(len(self.cfmms))))
        self.deltain, self.deltaout = zip(*threads)
        return self.deltain, self.deltaout
    
    def route(self, v=None):
        def fn(v):
            if not np.all(v == self.v):
                self.find_arb(v)
                self.v = v.copy()
            accumulator = 0.0
            for i in range (len(self.cfmms)):
                accumulator += np.dot(self.deltaout[i], v[self.cfmms[i].Ai]) - np.dot(self.deltain[i], v[self.cfmms[i].Ai])
            return self.objective.f(v) + accumulator

        def grad(v):
            if not np.all(v == self.v):
                self.find_arb(v)
                self.v = v.copy()
            G = self.objective.grad(v)

            for i in range(len(self.cfmms)):
                G[self.cfmms[i].Ai] += self.deltain[i]
                G[self.cfmms[i].Ai] -= self.deltaout[i]
            return G

        if v is None:
            v = self.v
        fn(v)
        v = opt.minimize(fun=fn, x0=v, method='L-BFGS-B', jac=grad, bounds=list(zip(self.objective.lower_limit(), self.objective.upper_limit())))
        self.v = v.x
        print("Optimization occurs")
        self.find_arb(self.v)
        return self.deltain, self.deltaout
 