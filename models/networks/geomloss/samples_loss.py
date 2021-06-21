import torch
from torch.nn import Module
from functools import partial
import warnings

from .kernel_samples import kernel_tensorized, kernel_online, kernel_multiscale

from .sinkhorn_samples import sinkhorn_tensorized
# from .sinkhorn_samples import sinkhorn_online
# from .sinkhorn_samples import sinkhorn_multiscale


routines = {
    "sinkhorn" : {
        "tensorized" : sinkhorn_tensorized,
        # "online"     : sinkhorn_online,
        }
}


class SamplesLoss(Module):
    """Creates a criterion that computes distances between sampled measures on a vector space.

    Warning:
        If **loss** is ``"sinkhorn"`` and **reach** is **None** (balanced Optimal Transport),
        the resulting routine will expect measures whose total masses are equal with each other.

    Parameters:
        loss (string, default = ``"sinkhorn"``): The loss function to compute.
            The supported values are:

              - ``"sinkhorn"``: (Un-biased) Sinkhorn divergence, which interpolates
                between Wasserstein (blur=0) and kernel (blur= :math:`+\infty` ) distances.
              - ``"hausdorff"``: Weighted Hausdorff distance, which interpolates
                between the ICP loss (blur=0) and a kernel distance (blur= :math:`+\infty` ).
              - ``"energy"``: Energy Distance MMD, computed using the kernel
                :math:`k(x,y) = -\|x-y\|_2`.
              - ``"gaussian"``: Gaussian MMD, computed using the kernel
                :math:`k(x,y) = \exp \\big( -\|x-y\|_2^2 \,/\, 2\sigma^2)`
                of standard deviation :math:`\sigma` = **blur**.
              - ``"laplacian"``: Laplacian MMD, computed using the kernel
                :math:`k(x,y) = \exp \\big( -\|x-y\|_2 \,/\, \sigma)`
                of standard deviation :math:`\sigma` = **blur**.
    """
    def __init__(self, loss="sinkhorn", p=2, blur=.05, reach=None, diameter=None, scaling=.5, truncate=5, cost=None,
                 kernel=None, cluster_scale=None, debias=True, potentials=False, verbose=False, backend="auto"):

        super(SamplesLoss, self).__init__()
        self.loss = loss
        self.backend = backend
        self.p = p
        self.blur = blur
        self.reach = reach
        self.truncate = truncate
        self.diameter = diameter
        self.scaling = scaling
        self.cost = cost
        self.kernel = kernel
        self.cluster_scale = cluster_scale
        self.debias = debias
        self.potentials = potentials
        self.verbose = verbose


    def forward(self, *args):
        """Computes the loss between sampled measures.
        
        Documentation and examples: Soon!
        Until then, please check the tutorials :-)"""
        
        l_x, α, x, l_y, β, y = self.process_args(*args)
        B, N, M, D = self.check_shapes(l_x, α, x, l_y, β, y)

        backend = self.backend  # Choose the backend
        backend = "tensorized"

        if B == 0 and backend == "tensorized":  # tensorized routines work on batched tensors
            α, x, β, y = α.unsqueeze(0), x.unsqueeze(0), β.unsqueeze(0), y.unsqueeze(0)


        # Run --------------------------------------------------------------------------------
        values = routines[self.loss][backend](α, x, β, y,
                    p=self.p, blur=self.blur, reach=self.reach,
                    diameter=self.diameter, scaling=self.scaling, truncate=self.truncate,
                    cost=self.cost, kernel=self.kernel, cluster_scale=self.cluster_scale,
                    debias=self.debias, potentials=self.potentials,
                    labels_x=l_x, labels_y=l_y, verbose=self.verbose)


        # Make sure that the output has the correct shape ------------------------------------
        if self.potentials:  # Return some dual potentials (= test functions) sampled on the input measures
            F, G, loss = values
            return F.view_as(α), G.view_as(β)

        else:  # Return a scalar cost value
            if backend in ["online", "multiscale"]:  # KeOps backends return a single scalar value
                if B == 0: return values           # The user expects a scalar value
                else:      return values.view(-1)  # The user expects a "batch list" of distances

            else:  # "tensorized" backend returns a "batch vector" of values
                if B == 0: return values[0]  # The user expects a scalar value
                else:      return values     # The user expects a "batch vector" of distances
                

    def process_args(self, *args):
        if len(args) == 6:
            return args
        if len(args) == 4:
            α, x, β, y = args
            return None, α, x, None, β, y
        elif len(args) == 2:
            x, y = args
            α = self.generate_weights(x)
            β = self.generate_weights(y)
            return None, α, x, None, β, y
        else:
            raise ValueError("A SamplesLoss accepts two (x, y), four (α, x, β, y) or six (l_x, α, x, l_y, β, y)  arguments.")


    def generate_weights(self, x):
        if x.dim() == 2:  # 
            N = x.shape[0]
            return torch.ones(N).type_as(x) / N
        elif x.dim() == 3:
            B, N, _ = x.shape
            return torch.ones(B,N).type_as(x) / N
        else:
            raise ValueError("Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors.")


    def check_shapes(self, l_x, α, x, l_y, β, y):

        if α.dim() != β.dim(): raise ValueError("Input weights 'α' and 'β' should have the same number of dimensions.")
        if x.dim() != y.dim(): raise ValueError("Input samples 'x' and 'y' should have the same number of dimensions.")
        if x.shape[-1] != y.shape[-1]: raise ValueError("Input samples 'x' and 'y' should have the same last dimension.")

        if x.dim() == 2:  # No batch --------------------------------------------------------------------
            B = 0  # Batchsize
            N, D = x.shape  # Number of "i" samples, dimension of the feature space
            M, _ = y.shape  # Number of "j" samples, dimension of the feature space
            
            if α.dim() not in [1,2] :
                raise ValueError("Without batches, input weights 'α' and 'β' should be encoded as (N,) or (N,1) tensors.")
            elif α.dim() == 2:
                if α.shape[1] > 1: raise ValueError("Without batches, input weights 'α' should be encoded as (N,) or (N,1) tensors.")
                if β.shape[1] > 1: raise ValueError("Without batches, input weights 'β' should be encoded as (M,) or (M,1) tensors.")
                α, β = α.view(-1), β.view(-1)

            if l_x is not None:
                if l_x.dim() not in [1,2] :
                    raise ValueError("Without batches, the vector of labels 'l_x' should be encoded as an (N,) or (N,1) tensor.")
                elif l_x.dim() == 2:
                    if l_x.shape[1] > 1: raise ValueError("Without batches, the vector of labels 'l_x' should be encoded as (N,) or (N,1) tensors.")
                    l_x = l_x.view(-1)
                if len(l_x) != N : raise ValueError("The vector of labels 'l_x' should have the same length as the point cloud 'x'.")
            
            if l_y is not None:
                if l_y.dim() not in [1,2] :
                    raise ValueError("Without batches, the vector of labels 'l_y' should be encoded as an (M,) or (M,1) tensor.")
                elif l_y.dim() == 2:
                    if l_y.shape[1] > 1: raise ValueError("Without batches, the vector of labels 'l_y' should be encoded as (M,) or (M,1) tensors.")
                    l_y = l_y.view(-1)
                if len(l_y) != M : raise ValueError("The vector of labels 'l_y' should have the same length as the point cloud 'y'.")

            N2, M2 = α.shape[0], β.shape[0]

        elif x.dim() == 3:  # batch computation ---------------------------------------------------------
            B, N, D = x.shape  # Batchsize, number of "i" samples, dimension of the feature space
            B2,M, _ = y.shape  # Batchsize, number of "j" samples, dimension of the feature space
            if B != B2: raise ValueError("Samples 'x' and 'y' should have the same batchsize.")

            if α.dim() not in [2,3] :
                raise ValueError("With batches, input weights 'α' and 'β' should be encoded as (B,N) or (B,N,1) tensors.")
            elif α.dim() == 3:
                if α.shape[2] > 1: raise ValueError("With batches, input weights 'α' should be encoded as (B,N) or (B,N,1) tensors.")
                if β.shape[2] > 1: raise ValueError("With batches, input weights 'β' should be encoded as (B,M) or (B,M,1) tensors.")
                α, β = α.squeeze(-1), β.squeeze(-1)
            
            if l_x is not None: raise NotImplementedError('The "multiscale" backend has not been implemented with batches.')
            if l_y is not None: raise NotImplementedError('The "multiscale" backend has not been implemented with batches.')

            B2, N2 = α.shape
            B3, M2 = β.shape
            if B != B2: raise ValueError("Samples 'x' and weights 'α' should have the same batchsize.")
            if B != B3: raise ValueError("Samples 'y' and weights 'β' should have the same batchsize.")

        else:
            raise ValueError("Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors.")
        
        if N != N2: raise ValueError("Weights 'α' and samples 'x' should have compatible shapes.")
        if M != M2: raise ValueError("Weights 'β' and samples 'y' should have compatible shapes.")

        return B, N, M, D
