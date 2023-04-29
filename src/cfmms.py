import numpy as np

class CFMM:
    def __init__(self, Ai, R, gamma):
        self.Ai = Ai
        self.R = R
        self.gamma = gamma

    def trading_function(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def update_reserves(self, deltain, deltaout):
        self.R += deltain - deltaout
    
    def find_arb(self, v):
        raise NotImplementedError("Subclass must implement abstract method")
    

    
class ConstantProduct(CFMM):
    def __init__(self, Ai, R, gamma):
        super().__init__(Ai, R, gamma)

    def trading_function(self):
        return self.R[0] * self.R[1]

    def update_reserves(self, deltain, deltaout):
        super.update_reserves(deltain, deltaout)
    
    def find_arb(self, v):
        def prod_arb_deltain(m, R, k, gamma):
            np.max(np.sqrt(gamma*m*k)-R, 0) / gamma
        def prod_arb_deltaout(m, R, k, gamma):
            np.max(R - np.sqrt(k/(gamma*m)), 0)
        
        k = self.trading_function()
        deltain = []
        deltaout = []
        deltain[0] = prod_arb_deltain(v[1]/v[0], self.R[0], k, self.gamma)
        deltain[1] = prod_arb_deltain(v[0]/v[1], self.R[1], k, self.gamma)
        deltaout[0] = prod_arb_deltaout(v[1]/v[0], self.R[0], k, self.gamma)
        deltaout[1] = prod_arb_deltaout(v[0]/v[1], self.R[1], k, self.gamma)
        return deltain, deltaout

class GeometricMeanTwoToken(CFMM):
    def __init__(self, Ai, R, gamma, w):
        super().__init__(Ai, R, gamma)
        self.w = w

    def trading_function(self):
        return self.R[0]**self.w[0] * self.R[1]**self.w[1]
    
    def update_reserves(self, deltain, deltaout):
        super().update_reserves(deltain, deltaout)

    def find_arb(self, v):
        def geo_arb_deltain(m, R1, R2, gamma, eta):
            np.max((gamma * m * eta * R1 * R2**eta)**(1 / (eta + 1)) - R2, 0) / gamma
        def geo_arb_deltaout(m, R1, R2, gamma, eta):
            np.max(R1 - ((R2 * R1**(1 / eta)) / (eta * gamma * m))**(eta / (1 + eta)), 0)
        
        eta = self.w[0] / self.w[1]
        deltain = []
        deltaout = []
        deltain[0] = geo_arb_deltain(v[1]/v[0], self.R[1], self.R[0], self.gamma, eta)
        deltain[1] = geo_arb_deltain(v[0]/v[1], self.R[0], self.R[1], self.gamma, 1/eta)
        deltaout[0] = geo_arb_deltaout(v[0]/v[1], self.R[0], self.R[1], self.gamma, 1/eta)
        deltaout[1] = geo_arb_deltaout(v[1]/v[0], self.R[1], self.R[0], self.gamma, eta)
        return deltain, deltaout

def zerotrade(c):
    n = len(c.Ai)
    deltain, deltaout = np.zeros(n), np.zeros(n)
    return deltain, deltaout