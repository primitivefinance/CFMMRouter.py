import numpy as np
import cvxpy as cp
from cfmms import ConstantProduct as Uni

# Problem data
global_indices = list(range(4))
local_indices = [
    [0, 1, 2, 3],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1]
]

reserves = list(map(np.array, [
    [4, 4, 4, 4],
    [10, 1],
    [1, 5],
    [40, 50],
    [12, 10]
]))

fees = [
    .998,
    .997,
    .997,
    .997,
    .999
]

# "Market value" of tokens (say, in a centralized exchange)
market_value = [
    1.5,
    4,
    2,
    3
] 

# Build local-global matrices
n = len(global_indices)
m = len(local_indices)

A = []
for l in local_indices:
    n_i = len(l)
    A_i = np.zeros((n, n_i))
    for i, idx in enumerate(l):
        A_i[idx, i] = 1
    A.append(A_i)

# Build variables
deltas = [cp.Variable(len(l), nonneg=True) for l in local_indices]
lambdas = [cp.Variable(len(l), nonneg=True) for l in local_indices]

psi = cp.sum([A_i @ (L - D) for A_i, D, L in zip(A, deltas, lambdas)])
psi2 = cp.sum([A_i @ D for A_i, D, L in zip(A, deltas, lambdas)])
# Objective is to maximize "total market value" of coins out
obj = cp.Maximize(market_value @ psi)

# Reserves after trade
new_reserves = [R + gamma_i*D - L for R, gamma_i, D, L in zip(reserves, fees, deltas, lambdas)]

# Trading function constraints
cons = [
    # Balancer pool with weights 4, 3, 2, 1
    cp.geo_mean(new_reserves[0], p=np.array([4, 3, 2, 1])) >= cp.geo_mean(reserves[0]),

    # Uniswap v2 pools
    cp.geo_mean(new_reserves[1]) >= cp.geo_mean(reserves[1]),
    cp.geo_mean(new_reserves[2]) >= cp.geo_mean(reserves[2]),
    cp.geo_mean(new_reserves[3]) >= cp.geo_mean(reserves[3]),

    # Constant sum pool
    cp.sum(new_reserves[4]) >= cp.sum(reserves[4]),
    new_reserves[4] >= 0,

    # Arbitrage constraint
    psi >= 0
]

# Set up and solve problem
prob = cp.Problem(obj, cons)
prob.solve()
Uni1 = Uni(local_indices[1], reserves[1], fees[1])
Uni2 = Uni(local_indices[2], reserves[2], fees[2])
Uni3 = Uni(local_indices[3], reserves[3], fees[3])
Uni4 = Uni(local_indices[4], reserves[4], fees[4])
print("Pool 1 Price: ", Uni1.get_price())
print("Pool 2 Price: ", Uni2.get_price())
print("Pool 3 Price: ", Uni3.get_price())
print("Pool 4 Price: ", Uni4.get_price())
print("Arbitrage occurs!")
print(f"Total profit: {prob.value}")
Uni1_update = Uni(local_indices[1], Uni1.R + deltas[1].value - lambdas[1].value, Uni1.gamma)
Uni2_update = Uni(local_indices[2], Uni2.R + deltas[2].value - lambdas[2].value, Uni2.gamma)
Uni3_update = Uni(local_indices[3], Uni3.R + deltas[3].value - lambdas[3].value, Uni3.gamma)
Uni4_update = Uni(local_indices[4], Uni4.R + deltas[4].value - lambdas[4].value, Uni4.gamma)
print("Pool 1 Price after Arb: ", Uni1_update.get_price())
print("Pool 2 Price after Arb: ", Uni2_update.get_price())
print("Pool 3 Price after Arb: ", Uni3_update.get_price())
print("Pool 4 Price after Arb: ", Uni4_update.get_price())
