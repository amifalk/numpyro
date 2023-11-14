from collections import namedtuple

import jax
from numpyro.infer.ensemble_util import batch_ravel_pytree
from jax import random, vmap
import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.util import identity

# this should just be a reimplementation of AIES
# (affine-invariant ensemble sampler)

EnsembleSamplerState = namedtuple("EnsembleSamplerState", ["z", "rng_key"])


def stretch_move(rng_key, active, inactive):
    """emcee"""
    n_active_chains, n_params = active.shape
    unif_key, idx_key = random.split(rng_key)

    # <hyperparams>
    a = 2.0 # The stretch scale parameter. (hyperparam)    
    
    zz = ((a - 1.0) * random.uniform(unif_key, shape=(n_active_chains,)) + 1)**2.0 / a
    factors = ((n_params - 1.0) * jnp.log(zz))
    r_idxs = random.randint(idx_key, shape=(n_active_chains,), minval=0, maxval=n_active_chains)
        
    proposal = inactive[r_idxs] - (inactive[r_idxs] - active) * zz[:, jnp.newaxis]

    return proposal, factors


# TODO: I can cache the output of this
def _get_nondiagonal_pairs(n):
    """Get the indices of a square matrix with size n, excluding the diagonal."""
    rows, cols = jnp.tril_indices(n, -1)  # -1 to exclude diagonal

    # Combine rows-cols and cols-rows pairs
    pairs = jnp.column_stack(
        [jnp.concatenate([rows, cols]), jnp.concatenate([cols, rows])]
    )

    return pairs

# HACK: arg should be n_active_chains, not hardcoded for my particular model
PAIRS = _get_nondiagonal_pairs(10) 

def de_move(rng_key, active, inactive):
    """emcee"""
    n_active_chains, n_params = active.shape
    pairs_key, gamma_key = random.split(rng_key)

    # <hyperparams>
    sigma = 1.0e-5 # sigma. The standard deviation of the Gaussian used to stretch the proposal vector.
    g0 = 2.38 / jnp.sqrt(2.0 * n_params) # gamma0. The mean stretch factor for the proposal vector.

    # Get the pair indices
    # (I cached it ahead of time)
    
    selected_pairs = random.choice(pairs_key, PAIRS, shape=(n_active_chains,))
    
    # Compute diff vectors
    diffs = jnp.diff(inactive[selected_pairs], axis=1).squeeze(axis=1) # get the pairwise difference of each vector 
    
    # Sample a gamma value for each walker following Nelson et al. (2013)
    gamma = dist.Normal(g0, g0*sigma).sample(gamma_key, sample_shape=(n_active_chains,1))
    
    # In this way, sigma is the standard deviation of the distribution of gamma,
    # instead of the standard deviation of the distribution of the proposal as proposed by Ter Braak (2006).
    # Otherwise, sigma should be tuned for each dimension, which confronts the idea of affine-invariance.
    
    proposal = inactive + gamma*diffs

    return proposal, jnp.zeros(n_active_chains)


def random_move(rng_key, active, inactive):
    """zeus"""
    
    
    pass


class EnsembleSampler(MCMCKernel):
    def __init__(self,
                 model=None,
                 num_chains=4,
                 potential_fn=None,
                 init_strategy=init_to_uniform
                 ):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        
        self._model = model
        self._potential_fn = potential_fn
        self._batch_potential_fn = None

        # --- other hyperparams go here
        self._num_chains = num_chains # must be an even number >= 2
        # ---        

        self._init_strategy = init_strategy
        self._postprocess_fn = None

        # TODO: allow users to select moves and the probability 
        # of using the move a la emcee: https://emcee.readthedocs.io/en/stable/tutorials/moves/
        
        
    @property
    def model(self):
        return self._model

    @property
    def sample_field(self):
        return "z"

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            new_params_info, potential_fn_gen, self._postprocess_fn, _, = initialize_model(
                rng_key,
                self._model,
                dynamic_args=True,
                init_strategy=self._init_strategy,
                model_args=model_args,
                model_kwargs=model_kwargs,
                validate_grad=False
            )
            new_init_params = new_params_info[0] 
            self._potential_fn = potential_fn_gen(*model_args, **model_kwargs)
            self._batch_potential_fn = vmap(self._potential_fn)

            if init_params is None:
                init_params = new_init_params        
        
        return init_params
            

    def init(self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}):
        rng_key, rng_key_init_model = random.split(rng_key)

        rng_key_init_model = random.split(rng_key_init_model, self._num_chains)
        
        init_params = self._init_state(rng_key_init_model, model_args, model_kwargs, init_params)
                
        if self._potential_fn and init_params is None:
            raise ValueError(
                "Valid value of `init_params` must be provided with" " `potential_fn`."
            )

        # TODO: currently we don't do anything internally with warmup arg
        
        return EnsembleSamplerState(init_params, rng_key)

    
    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    
    
    def sample(self, state, model_args, model_kwargs):
        z, rng_key = state

        # TODO: if random_split, randomize the chain indices

        # --- this can all be cached
        split_ind = self._num_chains // 2

        active_start_idx = [0, split_ind]
        active_stop_idx = [split_ind, self._num_chains]
        inactive_start_idx = [split_ind, 0]
        inactive_stop_idx = [self._num_chains, split_ind]
        # ---
        
        z_flat, unravel_fn = batch_ravel_pytree(z)
    
        for split in range(2):
            rng_key, proposal_key, accept_key = random.split(rng_key, 3)

            active_flat = z_flat[active_start_idx[split] : active_stop_idx[split]]
            inactive_flat = z_flat[inactive_start_idx[split] : inactive_stop_idx[split]]

            # --- make proposal (s = active, c = inactive)        
            #proposal_flat, factors = stretch_move(proposal_key, active_flat, inactive_flat)            
            proposal_flat, factors = de_move(proposal_key, active_flat, inactive_flat)            
            # ---
            
            # --- evaluate the proposal ---                
            log_accept_prob = (factors +
                               self._batch_potential_fn(unravel_fn(proposal_flat)) -
                               self._batch_potential_fn(unravel_fn(active_flat))
                               )

            accepted = dist.Uniform().sample(accept_key, (split_ind,)) > jnp.exp(log_accept_prob)            
            #accepted = jnp.ones_like(log_accept_prob)            
            
            z_updates = jnp.where(accepted[:, jnp.newaxis], proposal_flat, active_flat)
            
            z_flat = z_flat.at[active_start_idx[split]:active_stop_idx[split]].set(z_updates)
            
        
        return EnsembleSamplerState(unravel_fn(z_flat), rng_key)

