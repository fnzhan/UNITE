"""Implements the (unbiased) Sinkhorn divergence between sampled measures."""

import numpy as np
import torch
from functools import partial
from .utils import scal, squared_distances, distances
from .sinkhorn_divergence import epsilon_schedule, scaling_parameters
from .sinkhorn_divergence import dampening, log_weights, sinkhorn_cost, sinkhorn_loop



# ==============================================================================
#                          backend == "tensorized"
# ==============================================================================

cost_routines = {
    1 : (lambda x, y: distances(x, y)),
    2 : (lambda x, y: squared_distances(x, y) / 2)}

def softmin_tensorized(ε, C, f):
    B = C.shape[0]
    return - ε * (f.view(B, 1, -1) - C/ε).logsumexp(2).view(B, -1)

def sinkhorn_tensorized(α, x, β, y, p=2, blur=.05, reach=None, diameter=None, scaling=.5, cost=None, 
                        debias=True, potentials=False, **kwargs):
    
    B, N, D = x.shape
    _, M, _ = y.shape

    if cost is None:
        cost = cost_routines[p]
        
    C_xx, C_yy = (cost(x, x.detach()), cost(y, y.detach())) if debias else (None, None)  # (B,N,N), (B,M,M)
    C_xy, C_yx = (cost(x, y.detach()), cost(y, x.detach()))  # (B,N,M), (B,M,N)


    diameter, ε, ε_s, ρ = scaling_parameters(x, y, p, blur, reach, diameter, scaling)

    a_x, b_y, a_y, b_x = sinkhorn_loop(softmin_tensorized, log_weights(α), log_weights(β),
                                       C_xx, C_yy, C_xy, C_yx, ε_s, ρ, debias=debias)

    return sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x, batch=True, debias=debias, potentials=potentials)


