import numpy as np

class CFMM:
    '''
    CFMM superclass. Initializes CFMM state then defines the generic methods common to all CFMMs.
    --- parameters ---
    `Ai`:       index vector
    `R`:        reserve vector
    `gamma`:    fee regime
    '''
    def __init__(self, Ai, R, gamma):
        self.Ai = Ai            ## Index vector
        self.R = R              ## Reserve vector
        self.gamma = gamma      ## Fee regime

    def trading_function(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def update_reserves(self, deltain, deltaout):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def get_price(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def find_arb(self, v):
        raise NotImplementedError("Subclass must implement abstract method")
    
class ConstantProduct(CFMM):
    '''
    Constant Product CFMM (Uniswap V2). Inherits from CFMM superclass.
    --- parameters ---
    `Ai`:       index vector
    `R`:        reserve vector
    `gamma`:    fee regime
    '''
    def __init__(self, Ai, R, gamma):
        super().__init__(Ai, R, gamma)

    def trading_function(self):
        ## Returns trading function value.
        return self.R[0] * self.R[1]

    def update_reserves(self, deltain, deltaout):
        ## Updates pool reserves after arbitrage.
        for i in range(len(self.R)):
            self.R[i] += deltain[i]
            self.R[i] -= deltaout[i]

    def get_price(self):
        ## Returns the marginal price of token 0 in terms of token 1.
        return self.R[1] / self.R[0]
    
    def find_arb(self, v):
        ## See App. A of "Analysis of Uniswap Markets".
        def prod_arb_deltain(m, R, k, gamma):
            return np.max(np.sqrt(gamma*m*k)-R, 0) / gamma
        def prod_arb_deltaout(m, R, k, gamma):
            return np.max(R - np.sqrt(k/(gamma*m)), 0)
        
        ## Solves the maximum arbitrage problem for 2 token CPMM.
        k = self.trading_function()
        deltain = [0,0]
        deltaout = [0,0]
        deltain[0] = prod_arb_deltain(v[1] / v[0], self.R[0], k, self.gamma)
        deltain[1] = prod_arb_deltain(v[0] / v[1], self.R[1], k, self.gamma)
        deltaout[1] = prod_arb_deltaout(v[1] / v[0], self.R[0], k, self.gamma)
        deltaout[0] = prod_arb_deltaout(v[0] / v[1], self.R[1], k, self.gamma)
        print("trade vectors:", deltain, deltaout)
        return deltain, deltaout

class GeometricMeanTwoToken(CFMM):
    '''
    Two token implementation of the Geometric Mean Market Maker (G3M). Inherits from CFMM superclass.
    --- parameters ---
    `Ai`:       index vector
    `R`:        reserve vector
    `gamma`:    fee regime
    `w`:        weight vector (must satisfy: sum(w) = 1)
    '''
    def __init__(self, Ai, R, gamma, w):
        super().__init__(Ai, R, gamma)
        self.w = w

    def trading_function(self):
        ## Returns trading function value.
        return self.R[0]**self.w[0] * self.R[1]**self.w[1]
    
    def update_reserves(self, deltain, deltaout):
        ## Updates pool reserves after arbitrage.
        for i in range(len(self.R)):
            self.R[i] += deltain[i]
            self.R[i] -= deltaout[i]

    def get_price(self):
        ## Returns the marginal price of token 0 in terms of token 1.
        return (self.w[0] / self.w[1]) * self.trading_function() ** (1 / self.w[1]) * self.R[0] ** (- self.w[0]/self.w[1] - 1)

    def find_arb(self, v):
        def geo_arb_deltain(m, R1, R2, gamma, eta):
            return np.max((gamma * m * eta * R1 * R2**eta) ** (1 / (eta + 1)) - R2, 0) / gamma
        def geo_arb_deltaout(m, R1, R2, gamma, eta):
            return np.max(R1 - ((R2 * R1 ** (1 / eta)) / (eta * gamma * m)) ** (eta / (1 + eta)), 0)
        
        ## Solves the maximum arbitrage problem for 2 token G3M.
        eta = self.w[0] / self.w[1]
        deltain = [0,0]
        deltaout = [0,0]
        deltain[0] = geo_arb_deltain(v[1] / v[0], self.R[1], self.R[0], self.gamma, eta)
        deltain[1] = geo_arb_deltain(v[0] / v[1], self.R[0], self.R[1], self.gamma, 1/eta)
        deltaout[0] = geo_arb_deltaout(v[0] / v[1], self.R[0], self.R[1], self.gamma, 1/eta)
        deltaout[1] = geo_arb_deltaout(v[1] / v[0], self.R[1], self.R[0], self.gamma, eta)
        print("Trade vector G3M:", deltain, deltaout)
        return deltain, deltaout

def zerotrade(c):
    ## Returns zero trade vectors.
    n = len(c.Ai)
    vec = np.zeros(n)
    return vec