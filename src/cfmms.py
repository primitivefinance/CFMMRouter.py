import numpy as np

class CFMM:
    '''
    CFMM superclass. Initializes CFMM state then defines the generic methods common to all CFMMs.
    --- parameters ---
    `Ai`:       index vector
    `R`:        reserve vector
    `gamma`:    fee regime
    '''
    def __init__(self, Ai, R, gamma, w):
        self.Ai = Ai            ## Index vector
        self.R = R              ## Reserve vector
        self.gamma = gamma
        self.w = w      ## Fee regime

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
        w = np.array([1,1])
        super().__init__(Ai, R, gamma, w)

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
        super().__init__(Ai, R, gamma, w)

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

def zerotrade(c):
    ## Returns zero trade vectors.
    n = len(c.Ai)
    vec = np.zeros(n)
    return vec