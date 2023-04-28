import numpy as np
import cvxpy as cvx
from threading import Thread

class Router:
    def __init__(self, objective, cfmms, number_of_tokens):
       self.objective = objective
       self.cfmms = cfmms
       self.deltain = [objective.zerotrade(c) for c in cfmms]
       self.deltaout = [objective.zerotrade(c) for c in cfmms]
       self.v = np.zeros(number_of_tokens)

    def no_cfmms(objective, number_of_tokens):
        return Router(objective, [], number_of_tokens)
    
    def find_arb(self, v):
        threads = []

        for i in range(len(self.cfmms)):
            thread = Thread(target=self.objective.find_arb, args=(self.deltain[i], self.deltaout[i], self.cfmms[i], v[self.cfmms[i].Ai]))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    
    def route(self, v=None):
        if v is None:
            self.v = np.ones(len(self.v)) / len(self.v)
        else:
            self.v = v.copy()
        
        var_v = cvx.Variable(len(self.v), nonneg=True)
        constraints = []

        for (i,j,k) in zip(self.deltain, self.deltaout, self.cfmms):
            constraints.append(cvx.sum(j-i) == self.v[k.Ai])

        objective_fn = cvx.Maximize(self.objective.f(var_v) + cvx.sum(self.v))

        prob = cvx.Problem(objective_fn, constraints)
        prob.solve()

        self.v = var_v.value
        self.find_arb(self.v)

    def netflows(self):
        psi = np.zeros_like(self.v)

        for (i,j,k) in zip(self.deltain, self.deltaout, self.cfmms):
            psi[k.Ai] += j-i

        return psi
    
    def update_reserves(self):
        for (i,j,k) in zip(self.deltain, self.deltaout, self.cfmms):
            self.objective.update_reserves(k,i,j,self.v[k.Ai])
    