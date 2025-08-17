import numpy as np
import pandas as pd

import theano
import theano.tensor as tt
from theano import shared

import pymc3 as pm

from pymc3_hmm.distributions import SwitchingProcess, DiscreteMarkovChain
from pymc3_hmm.step_methods import FFBSStep, TransMatConjugateStep

def create_exponential_true(mu_1, mu_2):
    p_0_rv = np.r_[0.7, 0.3]
    p_1_rv = np.r_[0.2, 0.8]
    P_tt = tt.stack([p_0_rv, p_1_rv])
    P_rv = pm.Deterministic("P_t", tt.shape_padleft(P_tt)) #introduce a known transition matrix

    pi_0_tt = tt.as_tensor(np.r_[0.0, 1.0])  #the initial state S_0=1
    
    S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=1) #discrte state Markov chain

    Y_rv = SwitchingProcess(
        "Y_t",
        [pm.Exponential.dist(mu_1), pm.Exponential.dist(mu_2)],
        S_rv,
    ) #exponential distributions as switching emission distributions
    return Y_rv
    #this function is intended to get real observations

# function for Bayesian estimation for the special dim 1
def create_exponential_estimation_dim1(mu_1, mu_2, p_0_a, p_1_a, observed=None):
    p_0_rv = pm.Dirichlet("p_0", p_0_a) #first row in transition matrix, Dirichlet distribution prior
    p_1_rv = pm.Dirichlet("p_1", p_1_a) #second row in transition matrix, Dirichlet distribution prior

    P_tt = tt.stack([p_0_rv, p_1_rv]) 
    P_rv = pm.Deterministic("P_t", tt.shape_padleft(P_tt)) #transition matrix
    pi_0_tt = tt.as_tensor(np.r_[0.0, 1.0])  #the initial state S_0=1
    S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=1) #discrte state Markov chain
    
    Y_rv = SwitchingProcess(
        "Y_t",
        [pm.Exponential.dist(mu_1), pm.Exponential.dist(mu_2)],
        S_rv,
        observed=observed,
    ) 
    return Y_rv

# function for Bayesian estimation for dim>=2
def create_exponential_estimation(mu_1, mu_2, p_0_a, p_1_a, observed=None):
    p_0_rv = pm.Dirichlet("p_0", p_0_a) #first row in transition matrix, Dirichlet distribution prior
    p_1_rv = pm.Dirichlet("p_1", p_1_a) #second row in transition matrix, Dirichlet distribution prior

    P_tt = tt.stack([p_0_rv, p_1_rv]) 
    P_rv = pm.Deterministic("P_t", tt.shape_padleft(P_tt)) #transition matrix
    pi_0_tt = tt.as_tensor(np.r_[0.0, 1.0])  #the initial state S_0=1
    S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=np.shape(observed)[-1]) #discrte state Markov chain
    
    Y_rv = SwitchingProcess(
        "Y_t",
        [pm.Exponential.dist(mu_1), pm.Exponential.dist(mu_2)],
        S_rv,
        observed=observed,
    ) 
    return Y_rv

mu_1_true = 1
mu_2_true = 0.05
y_t = np.array([])
posterior_mu1 = np.array([])

for i in range(1000 ):
    if i<1:
        np.random.seed(i)
        with pm.Model() as sim_model:
            _ = create_exponential_true(mu_1_true, mu_2_true) #each time stage, 1 observation is observed

        y_t = np.append(y_t, pm.sample_prior_predictive(samples=1, model=sim_model)["Y_t"].squeeze())

        with pm.Model() as test_model:
            E_mu1, Var_mu = 1, 1
            E_mu2 = 0.1
            mu_1_rv = pm.Gamma("mu_1", E_mu1 ** 2 / Var_mu, E_mu1 / Var_mu)
            mu_2_rv = pm.Gamma("mu_2", E_mu2 ** 2 / Var_mu, E_mu2 / Var_mu) #gamma priors for mu_1 and mu_2

            _ = create_exponential_estimation_dim1(mu_1_rv, mu_2_rv, p_0_a=np.r_[1, 1], p_1_a=np.r_[1, 1], observed=y_t.reshape(-1,1)[0])

        with test_model:
            transmat_step = TransMatConjugateStep(test_model.P_t)
            states_step = FFBSStep([test_model.S_t])
            mu_step = pm.NUTS([test_model.mu_1, test_model.mu_2]) # MCMC sampling for states and all parameters

            posterior_trace = pm.sample(
            step=[transmat_step, states_step, mu_step],
            return_inferencedata=True,
            chains=1,
            progressbar=False,
            draws=3)
        posterior_mu1 = np.append(posterior_mu1, np.array(posterior_trace.posterior.mu_1))
   
    else:
        np.random.seed(i)
        with pm.Model() as sim_model:
            _ = create_exponential_true(mu_1_true, mu_2_true) #each time stage, 1 observation is observed

        y_t = np.append(y_t, pm.sample_prior_predictive(samples=1, model=sim_model)["Y_t"].squeeze())

        with pm.Model() as test_model:
            E_mu1, Var_mu = 1, 1
            E_mu2 = 0.1
            mu_1_rv = pm.Gamma("mu_1", E_mu1 ** 2 / Var_mu, E_mu1 / Var_mu)
            mu_2_rv = pm.Gamma("mu_2", E_mu2 ** 2 / Var_mu, E_mu2 / Var_mu) #gamma priors for mu_1 and mu_2

            _ = create_exponential_estimation(mu_1_rv, mu_2_rv, p_0_a=np.r_[1, 1], p_1_a=np.r_[1, 1], observed=y_t)

        with test_model:
            transmat_step = TransMatConjugateStep(test_model.P_t)  
            states_step = FFBSStep([test_model.S_t])
            mu_step = pm.NUTS([test_model.mu_1, test_model.mu_2]) # MCMC sampling for states and all parameters

            posterior_trace2 = pm.sample(
            step=[transmat_step, states_step, mu_step],
            return_inferencedata=True,
            chains=1,
            progressbar=False,
            draws=3)
        posterior_mu1 = np.append(posterior_mu1, np.array(posterior_trace2.posterior.mu_1))

print(posterior_mu1)