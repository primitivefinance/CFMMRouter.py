import numpy as np
import scipy.optimize as opt
from concurrent.futures import ThreadPoolExecutor
from cfmms import zerotrade
import cvxpy as cvx

class Router:
    def __init__(self, objective, cfmms, number_of_tokens, p):
       self.objective = objective
       self.cfmms = cfmms
       self.deltain = [cvx.Variable(len(c.Ai), nonneg=True) for c in cfmms]
       self.deltaout = [cvx.Variable(len(c.Ai), nonneg=True) for c in cfmms]
       self.v = p
    
    def route(self, v=None):
        A = []
        for c in self.cfmms:
            n_i = len(c.Ai)
            A_i = np.zeros((len(self.v), n_i))
            for i, idx in enumerate(c.Ai):
                A_i[idx, i] = 1
            A.append(A_i)

        new_reserves = [c.R + c.gamma * delta - lambdas for c, delta, lambdas in zip(self.cfmms, self.deltaout, self.deltain)]
        psi = cvx.sum([A_i @ (lambdas - delta) for A_i, lambdas, delta in zip(A, self.deltain, self.deltaout)])

        cons = []
        for i in range(len(self.cfmms)):
            cons.append(cvx.geo_mean(new_reserves[i], p=self.cfmms[i].w) >= cvx.geo_mean(self.cfmms[i].R, p=self.cfmms[i].w))
            cons.append(new_reserves[i] >= 0)

        cons.append(psi >= 0) 

        opti = cvx.Maximize(self.v @ psi)
        prob = cvx.Problem(opti, cons)
        prob.solve()
        print(self.deltain[0].value)
        print(self.v)
        return self.deltain, self.deltaout
 