from abc import ABC, abstractmethod
from collections import namedtuple

import jax
from numpyro.infer.ensemble_util import batch_ravel_pytree, _get_nondiagonal_pairs
from jax import random, vmap
import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.util import identity, is_prng_key


EnsembleSamplerState = namedtuple(
    "EnsembleSamplerState",
    [
        "z",
        "inner_state",
        "rng_key"
     ])
"""
A :func:`~collections.namedtuple` consisting of the following fields:

 - **z** - Python collection representing values (unconstrained samples from
   the posterior) at latent sites.
 - **inner_state** - A namedtuple containing information needed to update half the ensemble.
 - **rng_key** - random number generator seed used for generating proposals, etc.
"""

AIESState = namedtuple(
    "AIESState", 
    [
        "i", 
        "accept_prob",
        "mean_accept_prob",
        "rng_key"
    ]
)
"""
A :func:`~collections.namedtuple` consisting of the following fields.

 - **i** - iteration. This is reset to 0 after warmup.
 - **accept_prob** - Acceptance probability of the proposal. Note that ``z``
   does not correspond to the proposal if it is rejected.
 - **mean_accept_prob** - Mean acceptance probability until current iteration
   during warmup adaptation or sampling (for diagnostics).
 - **rng_key** - random number generator seed used for generating proposals, etc.
"""

ESSState = namedtuple(
    "ESSState", 
    [
        "rng_key"
    ]
)
"""
A :func:`~collections.namedtuple` used as an inner state for Ensemble Sampler.
This consists of the following fields:

 - **rng_key** - random number generator seed used for generating proposals, etc.
"""



# ensemble slice sampler internal functions
def _ess(batch_log_density):
    # for easier batching
    batch_log_density = lambda x: batch_log_density(x)[:, jnp.newaxis]    
    
    # TODO what are these again?
    MAXSTEPS = 10_000
    MAXITER = 10_000 
    
    def step_out(rng_key, log_slice_height, active, directions):
        init_L_key, init_J_key = random.split(rng_key) 
        n_active_chains, n_params = active.shape

        iteration = 0
        n_expansions = 0
        # set initial interval boundaries
        L = -dist.Uniform().sample(init_L_key, sample_shape=(n_active_chains, 1))     
        R = L + 1.0    

        # stepping out
        J = jnp.floor(dist.Uniform(low=0, high=MAXSTEPS).sample(init_J_key, sample_shape=(n_active_chains, 1)))    
        K = (MAXSTEPS - 1) - J
        mask_J = jnp.full((n_active_chains, 1), True) # left stepping-out initialisation
        mask_K = jnp.full((n_active_chains, 1), True) # right stepping-out initialisation

        init_values = (n_expansions, L, R, J, K, mask_J, mask_K, iteration)

        def cond_fn(args):
            n_expansions, L, R, J, K, mask_J, mask_K, iteration = args

            return (jnp.count_nonzero(mask_J) + jnp.count_nonzero(mask_K) > 0) & (iteration < MAXITER) 

        def body_fn(args):
            n_expansions, L, R, J, K, mask_J, mask_K, iteration = args

            log_prob_L = batch_log_density(directions * L + active) # TODO: could make this into one if I wanted to
            log_prob_R = batch_log_density(directions * R + active)

            can_expand_L = log_prob_L > log_slice_height
            L = jnp.where(can_expand_L, L - 1, L)
            J = jnp.where(can_expand_L, J - 1, J)
            mask_J = jnp.where(can_expand_L, mask_J, False)

            can_expand_R = log_prob_R > log_slice_height
            R = jnp.where(can_expand_R, R + 1, R) 
            K = jnp.where(can_expand_R, K - 1, K) 
            mask_K = jnp.where(can_expand_R, mask_K, False) 

            iteration += 1
            n_expansions += jnp.count_nonzero(can_expand_L) + jnp.count_nonzero(can_expand_R) 

            return (n_expansions, L, R, J, K, mask_J, mask_K, iteration)

        # returns (n_expansions, L, R)
        n_expansions, L, R, J, K, mask_J, mask_K, iteration = jax.lax.while_loop(cond_fn, body_fn, init_values)

        return n_expansions, L, R
    
    def shrink(rng_key, log_slice_height, L, R, active, directions):
        n_active_chains, n_params = active.shape
        
        iteration = 0
        n_contractions = 0
        widths = jnp.zeros((n_active_chains, 1))
        proposed = jnp.zeros((n_active_chains, n_params))
        can_shrink = jnp.full((n_active_chains, 1), True) # this should be called can_shrink

        init_values = (rng_key, proposed, n_contractions, L, R, widths, can_shrink, iteration)

        def cond_fn(args):
            (rng_key, proposed, n_contractions, L, R, widths, can_shrink, iteration) = args

            return (jnp.count_nonzero(can_shrink) > 0) & (iteration < MAXITER) 

        def body_fn(args):
            rng_key, proposed, n_contractions, L, R, widths, can_shrink, iteration = args

            rng_key, _ = random.split(rng_key)

            widths = jnp.where(can_shrink, dist.Uniform(low=L, high=R).sample(rng_key), widths) 

            # compute new positions
            proposed = jnp.where(can_shrink, directions * widths + active, proposed)
            proposed_log_prob = batch_log_density(proposed)

            # shrink slices  
            can_shrink = proposed_log_prob < log_slice_height

            L_cond = can_shrink & (widths < 0.0)
            L = jnp.where(L_cond, widths, L)

            R_cond = can_shrink & (widths > 0.0)
            R = jnp.where(R_cond, widths, R)

            iteration += 1 
            n_contractions += jnp.count_nonzero(L_cond) + jnp.count_nonzero(R_cond) 

            return (rng_key, proposed, n_contractions, L, R, widths, can_shrink, iteration)

        rng_key, proposed, n_contractions, L, R, widths, can_shrink, iteration = jax.lax.while_loop(cond_fn, body_fn, init_values)
        
        return proposed, n_contractions 


    def sample_kernel(rng_key, active, inactive):
        n_active_chains, n_params = active.shape
        rng_key, dir_key, height_key, step_out_key, shrink_key = random.split(rng_key, 5)
        
        mu = 1.0
        
        directions = ESS.RandomMove(dir_key, inactive, mu)
        log_slice_height = batch_log_density(active) - dist.Exponential().sample(height_key, 
                                                                                  sample_shape=(n_active_chains, 1))
        
        n_expansions, L, R = step_out(step_out_key, log_slice_height, active, directions)
        proposal, n_contractions = shrink(shrink_key, log_slice_height, L, R, active, directions)

        # TODO: add tuning
        # this should be computed from the sum from both ones
        # n_expansions = jnp.max(jnp.array([1, n_expansions])) # This is to prevent the optimizer from getting stuck
        # mu *= 2.0 * n_expansions / (n_expansions + n_contractions)    

        return proposal

    return sample_kernel


class EnsembleSampler(MCMCKernel, ABC):
    def __init__(self,
                 model=None,
                 potential_fn=None,
                 randomize_split=True,
                 init_strategy=init_to_uniform
                 ):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        
        self._model = model
        self._potential_fn = potential_fn
        self._batch_log_density = None # unravel an (n_chains, n_params) Array into a pytree and
                                       # evaluate the log density at each chain

        # --- other hyperparams go here
        self._num_chains = None # must be an even number >= 2
        self._randomize_split = randomize_split # whether or not to permute the chain order at each iteration
        # ---        

        self._init_strategy = init_strategy
        self._postprocess_fn = None

        
    @property
    def model(self):
        return self._model

    @property
    def sample_field(self):
        return "z"

    @abstractmethod
    def init_inner_state(self, rng_key):
        """return inner_state"""
        raise NotImplementedError

    
    @abstractmethod
    def update_active_chains(self, active, inactive, inner_state):
        """return (updated active set of chains, updated inner state)"""
        raise NotImplementedError 


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
            
            _, unravel_fn = batch_ravel_pytree(new_init_params)            
            self._batch_log_density = lambda z: -vmap(self._potential_fn)(unravel_fn(z))
            
            if init_params is None:
                init_params = new_init_params        
        
        return init_params


    def init(self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}):
        assert not is_prng_key(rng_key), (
            "EnsembleSampler only supports chain_method='vectorized' or chain_method='parallel'."
        )
        assert rng_key.shape[0] % 2 == 0, ("Number of chains must be even.")

        self._num_chains = rng_key.shape[0]
                
        rng_key, rng_key_inner_state, rng_key_init_model = random.split(rng_key[0], 3)
        rng_key_init_model = random.split(rng_key_init_model, self._num_chains)
        
        init_params = self._init_state(rng_key_init_model, model_args, model_kwargs, init_params)
                
        if self._potential_fn and init_params is None:
            raise ValueError(
                "Valid value of `init_params` must be provided with" " `potential_fn`."
            )

        self._num_warmup = num_warmup
        
        return EnsembleSamplerState(init_params, self.init_inner_state(rng_key_inner_state), rng_key)

    
    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)
    
    
    def sample(self, state, model_args, model_kwargs):
        z, inner_state, rng_key = state
        rng_key, _ = random.split(rng_key)
        z_flat, unravel_fn = batch_ravel_pytree(z)
        
        # --- this can all be cached
        split_ind = self._num_chains // 2
        active_start_idx = [0, split_ind]
        active_stop_idx = [split_ind, self._num_chains]
        inactive_start_idx = [split_ind, 0]
        inactive_stop_idx = [self._num_chains, split_ind]
        # ---
    
        if self._randomize_split:
            z_flat = random.permutation(rng_key, z_flat, axis=0)    
    
        for split in range(2):
            active = z_flat[active_start_idx[split] : active_stop_idx[split]]
            inactive = z_flat[inactive_start_idx[split] : inactive_stop_idx[split]]
            
            z_updates, inner_state = self.update_active_chains(active, inactive, inner_state)
            
            z_flat = z_flat.at[active_start_idx[split]:active_stop_idx[split]].set(z_updates)
        
        return EnsembleSamplerState(unravel_fn(z_flat), inner_state, rng_key)

# affine-invariant ensemble sampler
class AIES(EnsembleSampler):
    # TODO: this doesn't work because state_method='vectorized' shuts off diagnostics_str
    def get_diagnostics_str(self, inner_state):
        return "acc. prob={:.2f}".format(inner_state.mean_accept_prob)
    
    
    def init_inner_state(self, rng_key):
        # TODO: allow users to specify moves and their corresponding probs
        # TODO: allow kwargs for moves
        # moves=[
        #     (emcee.moves.DEMove(), 0.8),
        #     (emcee.moves.DESnookerMove(), 0.2),
        # ]

        self.moves = [AIES.StretchMove, 
                      AIES.make_de_move(self._num_chains)
                      ]

        return AIESState(jnp.array(0.), 
                         jnp.array(0.),
                         jnp.array(0.),
                         rng_key)
    
    def update_active_chains(self, active, inactive, inner_state):
        i, _, mean_accept_prob, rng_key = inner_state
        rng_key, move_key, proposal_key, accept_key = random.split(rng_key, 4)

        move_i = random.choice(move_key, len(self.moves), p=jnp.array([0., 1.]))
        proposal, factors = jax.lax.switch(move_i, self.moves, proposal_key, active, inactive)         
            
        # --- evaluate the proposal ---                
        log_accept_prob = (factors +
                           self._batch_log_density(proposal) -
                           self._batch_log_density(active))

        accepted = dist.Uniform().sample(accept_key, (active.shape[0],)) < jnp.exp(log_accept_prob)                                    
        accept_prob = jnp.count_nonzero(accepted) / accepted.shape[0]        

        updated_active_chains = jnp.where(accepted[:, jnp.newaxis], proposal, active)

        itr = i + .5
        n = jnp.where(i < self._num_warmup, itr, itr - self._num_warmup)
        mean_accept_prob = mean_accept_prob + (accept_prob - mean_accept_prob) / n

        return updated_active_chains, AIESState(itr, accept_prob, mean_accept_prob, rng_key)
    
    
    @staticmethod
    def StretchMove(rng_key, active, inactive, a=2.0):
        """
        A `Goodman & Weare (2010)
        <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
        parallelization as described in `Foreman-Mackey et al. (2013)
        <https://arxiv.org/abs/1202.3665>`_.

        :param a: (optional)
            The stretch scale parameter. (default: ``2.0``)
        """
        n_active_chains, n_params = active.shape
        unif_key, idx_key = random.split(rng_key)

        zz = ((a - 1.0) * random.uniform(unif_key, shape=(n_active_chains,)) + 1) ** 2.0 / a
        factors = ((n_params - 1.0) * jnp.log(zz))
        r_idxs = random.randint(idx_key, shape=(n_active_chains,), minval=0, maxval=n_active_chains)

        proposal = inactive[r_idxs] - (inactive[r_idxs] - active) * zz[:, jnp.newaxis]

        return proposal, factors
    

    @staticmethod
    def make_de_move(n_chains):
        PAIRS = _get_nondiagonal_pairs(n_chains // 2)

        # TODO: this is broken
        def DEMove(rng_key, active, inactive, sigma=1.0e-5, g0=None):
            """A proposal using differential evolution.
            This `Differential evolution proposal
            <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ is
            implemented following `Nelson et al. (2013)
            <https://doi.org/10.1088/0067-0049/210/1/11>`_.
            Args:
                sigma (float): The standard deviation of the Gaussian used to stretch
                    the proposal vector.
                gamma0 (Optional[float]): The mean stretch factor for the proposal
                    vector. By default, it is `2.38 / sqrt(2*ndim)`
                    as recommended by the two references.
            """
            pairs_key, gamma_key = random.split(rng_key)
            n_active_chains, n_params = inactive.shape

            if not g0:
                g0 = 2.38 / jnp.sqrt(2.0 * n_params)

            selected_pairs = random.choice(pairs_key, PAIRS, shape=(n_active_chains,))

            # Compute diff vectors
            diffs = jnp.diff(inactive[selected_pairs], axis=1).squeeze(axis=1) # get the pairwise difference of each vector 

            # Sample a gamma value for each walker following Nelson et al. (2013)
            gamma = dist.Normal(g0, g0*sigma).sample(gamma_key, sample_shape=(n_active_chains, 1))

            # In this way, sigma is the standard deviation of the distribution of gamma,
            # instead of the standard deviation of the distribution of the proposal as proposed by Ter Braak (2006).
            # Otherwise, sigma should be tuned for each dimension, which confronts the idea of affine-invariance.

            proposal = inactive + gamma * diffs

            return proposal, jnp.zeros(n_active_chains)

        return DEMove
        
    
# ensemble slice sampler    
class ESS(EnsembleSampler):
    def init_inner_state(self, rng_key):
        # TODO
        pass


    def update_active_chains(self, active, inactive, inner_state):
        # TODO
        pass


    @staticmethod
    def RandomMove(rng_key, inactive, mu=1.0):
        """
        Generate direction vectors.
        Parameters
        ----------
            inactive: array
                Array of shape ``(nwalkers//2, ndim)`` with the walker positions of the complementary ensemble.
            mu : float
                The value of the scale factor ``mu``.

        Returns
        -------
            directions : array
                Array of direction vectors of shape ``(nwalkers//2, ndim)``.
        """
        directions = dist.Normal(loc=0, scale=1).sample(rng_key, sample_shape=inactive.shape)
        directions /= jnp.linalg.norm(directions, axis=0)
    
        return 2.0 * mu * directions