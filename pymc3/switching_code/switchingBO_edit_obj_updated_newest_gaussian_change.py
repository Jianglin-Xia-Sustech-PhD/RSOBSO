import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import poisson
#from hmmlearn import hmm
from switching_code.exponentialhmm import ExponentialHMM 
from hmmlearn.hmm import PoissonHMM
import pandas as pd
import scipy.stats as stats
import scipy
from pyDOE import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import pymc3 as pm 
import os
#from sklearn import decomposition
import warnings
warnings.filterwarnings("ignore") 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #find path from upper layer
import theano.tensor as tt
from pymc3_hmm.distributions import SwitchingProcess, DiscreteMarkovChain
from pymc3_hmm.step_methods import FFBSStep, TransMatConjugateStep
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize

####updated on Apr22, 2025 (deal with kde without log and no_RS codes)
####updated on Apr23, 2025 (deal with label switching for RS_plug_mean)
####updated on May6, 2025 (change objective function and candidate space)
####updated on May24, 2025 (change bayesian_hmm_posterior_gaussian_known_var startprob and tune 1000) and candidate space for syn

def inital_design(n_sample, seed, lower_bound, upper_bound):  
    # this function aims to use latin-hypercube design and 'maxmin' criterion to generate initial design points
    np.random.seed(seed)
    dimension_x = len(lower_bound)
    lhd= lhs(dimension_x,samples = n_sample,criterion = "maximin")
    D1 = np.zeros((n_sample, dimension_x))
    for i in range(dimension_x): 
        D1[:,i] = lhd[:,i]*(upper_bound[i]-lower_bound[i]) + lower_bound[i]
    return D1


def predicted_mean_std(X_, weights, gps):
    # this function aims to get the weighted sum of predicted mu and sigma generated from each gp in gps 
    mu_g_list = []
    sigma_g_list = []

    for gp in gps:
        mu_g,sigma_g = gp.predict(X_,return_std= True) 
        mu_g_list.append(mu_g)
        sigma_g_list.append(sigma_g)

    mu = 0
    sigma = 0
    for i in range(len(gps)):
        mu += mu_g_list[i] * weights[i]
        sigma += sigma_g_list[i]**2 * weights[i]**2
    sigma = np.sqrt(sigma)

    return mu, sigma

def predicted_mean_std_joint_model(X_,lambdas_, weights, gp):
    # in iterarion, (x1,x2,1/lambda0), (x1,x2,1/lambda1) used to gp prediction 
    weights = weights.reshape(1,-1)
    constant_lambda1 = np.full((X_.shape[0], 1), 1.0/lambdas_[0])
    constant_lambda2 = np.full((X_.shape[0], 1), 1.0/lambdas_[1])
    D1_lambda1 = np.concatenate((X_, constant_lambda1), axis=1)
    D1_lambda2 = np.concatenate((X_, constant_lambda2), axis=1)

    mu = np.zeros(X_.shape[0])
    var = np.zeros(X_.shape[0])
    for i in range(X_.shape[0]):
        D = np.concatenate((D1_lambda1[i:(i+1),:], D1_lambda2[i:(i+1),:]), axis=0)
        m, k = gp.predict(D,return_cov= True) 
        m = m.reshape(-1, 1)
        # print(m.shape, weights.shape, k.shape)
        # print(sum(m * np.transpose(weights)))
        mu[i] =  sum(m * np.transpose(weights))
        var[i] = np.matmul(np.matmul(weights, k), np.transpose(weights))
    sigma = np.sqrt(var)
    return mu, sigma

def predicted_mean_std_single_lambda(X_, lambda_val, gp):
    """
    Compute the predicted mean and standard deviation from a Gaussian Process (GP) 
    with a single lambda value.
    
    :param X_: Input data of shape (N, d)
    :param lambda_val: A single scalar lambda value
    :param gp: Trained Gaussian Process model
    :return: Tuple (mu, sigma), where:
        - mu: Predicted mean of shape (N,)
        - sigma: Predicted standard deviation of shape (N,)
    """
    # Expand X_ with 1/lambda
    lambda_inv = np.full((X_.shape[0], 1), 1.0 / lambda_val)
    D_lambda = np.concatenate((X_, lambda_inv), axis=1)  # Shape (N, d+1)

    # Ensure gp.predict supports return_std
    mu, sigma = gp.predict(D_lambda, return_std=True)  

    return mu, sigma

def predicted_mean_std_single_lambda_direct(X_, lambda_val, gp):
    """
    Compute the predicted mean and standard deviation from a Gaussian Process (GP) 
    using a single lambda value (no inverse).

    :param X_: Input data of shape (N, d)
    :param lambda_val: A single scalar lambda value
    :param gp: Trained Gaussian Process model
    :return: Tuple (mu, sigma), where:
        - mu: Predicted mean of shape (N,)
        - sigma: Predicted standard deviation of shape (N,)
    """
    # Append lambda directly instead of its inverse
    lambda_arr = np.full((X_.shape[0], 1), lambda_val)
    D_lambda = np.concatenate((X_, lambda_arr), axis=1)  # Shape (N, d+1)

    mu, sigma = gp.predict(D_lambda, return_std=True)

    return mu, sigma

def EGO(X_, mu_AEI,mu,sigma):
    # this function is the classical EGO algorithm, choose the next x from X_
    sigma = sigma.reshape(-1)
    mu_AEI = mu_AEI.reshape(-1)
    mu = mu.reshape(-1)
    mu_min = np.min(mu_AEI)
    EGO_crit = np.empty(X_.shape[0])
    sigma[sigma == 0 ] = np.finfo(float).eps        
    delta = mu_min - mu
    EGO_crit = delta* scipy.stats.norm.cdf(delta/sigma)+ sigma * scipy.stats.norm.pdf(delta/sigma) 
    index = np.argmax(EGO_crit)
    X_new = X_[index]

    return X_new

def switching_model_fit(X, n_components = 2, type = "exp"):
      scores = list()
      models = list()

      for idx in range(10):  # ten different random starting states

            # define our hidden Markov model
            if type == "exp":
                  model = ExponentialHMM(n_components=n_components, random_state=idx,
                                          n_iter=30)
            elif type == "poisson":
                  model = PoissonHMM(n_components=n_components, random_state=idx,
                                          n_iter=30)
            else:
                  raise NotImplementedError("Not implemented")
            
            model.fit(X)
            models.append(model)
            scores.append(model.score(X))
            print(f'Converged: {model.monitor_.converged}\t\t'
                  f'Score: {scores[-1]}')

      # get the best model
      model = models[np.argmax(scores)]
      print(f'The best model had a score of {max(scores)} and '
            f'{model.n_components} components')

      # use the Viterbi algorithm to predict the most likely sequence of states
      # given the model
    #   states = model.predict(X)

      return model

class portfolio_problem:
    
    def __init__(self, true_dist):
        
        self.dimension_x = 10
        self.dimension_p = 1
        self.lb = [0] *  self.dimension_x #[1, 2.26]
        self.ub = [1] * self.dimension_x #[2.25, 3.5]
        self.true_dist  = true_dist
        self.x_star = None 
        self.f_star = None 

    def __call__(self, xx) :
        
        value = self.evaluate_true(xx)
        
        return value
    
    def evaluate(self, xx, P, n_rep, method):
        s = xx[0]
        S = xx[1]
        value = cost_simulator(s,S,P, n_rep, method)[0]

        return value
    
    def evaluate_true(self, xx):

        s = xx[0]
        S = xx[1]
        value = inventory(s,S,self.true_dist.get("rate"))

        return value     

    def estimate_minimizer_minimum(self, n_sample = 1000):

        np.random.seed(None)
        lhd = lhs(self.dimension_x,samples = n_sample,criterion = "maximin")
        X_ = np.zeros((n_sample, self.dimension_x))
        for i in range(self.dimension_x): 
            X_[:,i] = lhd[:,i]*(self.ub[i]-self.lb[i]) + self.lb[i]
        Y_ = np.zeros(X_.shape[0])
        for i in range(X_.shape[0]):
            Y_[i] = self.evaluate_true(X_[i])
        self.f_star = min(Y_)
        self.x_star = X_[np.argmin(Y_)]

        return self.f_star,self.x_star 
    

class inventory_problem:
    
    def __init__(self, true_dist):
        
        self.dimension_x = 2
        self.dimension_p = 1
        self.lb = [1,70] #[1, 2.26]
        self.ub = [69, 250] #[2.25, 3.5]
        self.true_dist  = true_dist
        self.x_star = None #list([np.array([0.20169,0.150011,0.476874,0.275332,0.311652,0.6573]).reshape(1,-1)])
        self.f_star = None #-3.32237
        # self.mean = -0.26
        # self.std = 0.38
        # self.f_star = (self.f_star - self.mean)/self.std      
        # x_star = np.array([2.20849609,2.30601563])
        # f_star = inventory(2.20849609, 2.30601563,true_dist.get("lambda"))
        # print(x_star, f_star)

    def __call__(self, xx) :
        
        value = self.evaluate_true(xx)
        
        return value
    
    def evaluate(self, xx, P, n_rep, method):
        s = xx[0]
        S = xx[1]
        value = cost_simulator(s,S,P, n_rep, method)[0]

        return value
    
    def evaluate_true(self, xx):

        s = xx[0]
        S = xx[1]
        value = inventory(s,S,self.true_dist.get("rate"))

        return value     

    def estimate_minimizer_minimum(self, n_sample = 1000):

        np.random.seed(None)
        lhd = lhs(self.dimension_x,samples = n_sample,criterion = "maximin")
        X_ = np.zeros((n_sample, self.dimension_x))
        for i in range(self.dimension_x): 
            X_[:,i] = lhd[:,i]*(self.ub[i]-self.lb[i]) + self.lb[i]
        Y_ = np.zeros(X_.shape[0])
        for i in range(X_.shape[0]):
            Y_[i] = self.evaluate_true(X_[i])
        self.f_star = min(Y_)
        self.x_star = X_[np.argmin(Y_)]

        return self.f_star,self.x_star 

def cost_simulator(s,S,P,n_rep=10, method = 'NBRO',NOP=1000,b=100,h=1,c=1,K=100):
    #h holding cost per period per unit of inventory
    #b shortage cost per period per unit of inventory
    #c per-unit ordering cost
    #K setup cost for placing an order

    s = s
    S = S
    output=np.zeros(n_rep)

    for i in np.arange(n_rep):
        output[i]= costsim(NOP,P,b,h,c,K,s,S,method)

    cost_y= np.mean(output); # take the average over all replications (x-bar[n])
    cost_v= np.var(output)/n_rep; # noise withe rep replications
    
    return cost_y,cost_v

def costsim(NOP,P,b,h,c,K,s,S,method): ####controlled by 'method'
    ## types ####D: generate NOP samples by the fitted distribution (number of periods)
    burnin = 100

    if method == 'hist': # 'histogram'
        D = np.random.choice(P, size = NOP)
    elif method == 'bayesian_hist': #kde fit pdf
        D = kde_sample(P, size = NOP)
    elif method == 'exp': # "parametric"
        D = fit_and_generate_exponential(xi = P, size = NOP)
    elif method == 'posterior_mean_exp':
        D = posterior_mean_and_generate_exponential(xi = P, size = NOP)
    elif method == 'lognorm': # 
        D = fit_and_generate_log_normal(xi = P, size = NOP)
    elif method in ["switching","switching_joint"]:
        if P.get('dist') == "exp":
            rv = stats.expon(scale = 1.0/P.get("rate"))
            D = generate_sample_general(NOP,rv)
    else:
        raise KeyError('Wriong input!')

    if np.min(D) < 0:
        print('negative demand')

    cost = np.zeros(NOP)
    O = np.zeros(NOP) # Order
    I = np.zeros(NOP) # inventory on-hand at the end of period

    for t in np.arange(NOP):
    
        if t==0:
            I[t]= S-D[t] # inventory on-hand at the end of period
        else:
            I[t]= I[t-1]+O[t-1]- D[t]

        if I[t]> s:
            O[t]= 0
        else:
            O[t]= S-I[t]
        
        if O[t] > 0:
            cost[t]= c * O[t] + K + h * np.max((I[t],0))+ b * np.max((-I[t],0))
            
        else:
            cost[t]= h * np.max((I[t],0))+ b * np.max((-I[t],0)) #remove  c * O[t]
            
        #print(c * O[t], h * np.max(I[t],0), b * np.max((-I[t],0)))
    
    cost2 = cost[burnin:]
    
    Ecost = np.sum(cost2)/len(cost2)

    return Ecost

def inventory(s,S,lamda,b=100,h=1,c=1,K=100):
    s = s
    S = S
    output = c/lamda + (K+h*(s-1/lamda+0.5*lamda*(S**2- s**2))+ (h+b)/lamda*np.exp(-lamda*s))/(1+lamda*(S-s))
    return output

def fit_and_generate_exponential(xi, size = 1000):
    
    rv = fit_exponential(xi)
    xi = generate_sample_general(size, rv)

    return xi

def fit_and_generate_log_normal(xi, size = 1000):
    
    _, rv = fit_log_normal(xi)
    xi = generate_sample_general(size, rv)

    return xi

def fit_exponential(xi):
    _, scale = stats.expon.fit(xi, floc=0) ###exponential MLE fit parameters, _ loc params, no return
    rv = stats.expon(scale = scale)
    return rv

def fit_log_normal(xi):
    s, _, scale = stats.lognorm.fit(xi, floc=0)
    rv = stats.lognorm(s = s, scale = scale)
    return ([1], rv)

def generate_sample_general(xi_n, rv):
    xi  = rv.rvs(size=(xi_n))
    return xi

def posterior_mean_exponential_fast(observed, alpha_prior=1.0, beta_prior=0.1): 
    ##change prior distributions here
    """
    直接计算指数分布参数 λ 的后验均值，并返回基于后验均值的指数分布对象。

    参数:
    - observed: 观测数据 (numpy array)
    - alpha_prior: Gamma 先验的形状参数 (默认 1.0)
    - beta_prior: Gamma 先验的尺度参数 (默认 0.1)

    返回:
    - rv: 以后验均值 作为 lambda 估计的 scipy.stats.expon 分布对象
    """
    n = len(observed)  # 观测数据数量
    sum_x = np.sum(observed)  # 观测数据的总和
    
    # 计算 λ 的后验均值
    posterior_mean = (alpha_prior + n) / (beta_prior + sum_x)
    
    # 计算指数分布的 scale 参数
    scale = 1 / posterior_mean  # **修正这里**
    
    # 创建指数分布对象
    rv = stats.expon(scale=scale)
    return rv

def posterior_mean_and_generate_exponential(xi, size=1000): 
    """
    使用指数分布的后验均值生成新样本。

    参数:
    - xi: 原始观测数据
    - size: 要生成的样本数量
    - alpha_prior: Gamma 先验的 alpha 参数
    - beta_prior: Gamma 先验的 beta 参数

    返回:
    - xi: 生成的新样本
    """
    rv = posterior_mean_exponential_fast(xi)
    xi = generate_sample_general(size, rv)
    return xi

def kde_sample(xi, size=1000):  # newly added March 27, 2025
    """
    Fit a KDE to positive-valued data using log transformation and generate samples > 0.

    Parameters:
    - xi: Input data (numpy array, all values must be > 0)
    - size: Number of samples to generate (default: 1000)

    Returns:
    - samples: A (size,) numpy array of positive-valued random samples
    """
    xi = xi[xi > 0]  # Filter out non-positive values

    if len(xi) == 0:
        raise ValueError("No positive value in input data")

    log_xi = np.log(xi)  # Apply log transformation
    kde = stats.gaussian_kde(log_xi)  # Fit KDE on log-transformed data
    log_samples = kde.resample(size).reshape(-1)  # Sample in log-space
    samples = np.exp(log_samples)  # Transform back to original space
    return samples

#change prior gamma(1, 0.1) both, allow general number of regimes
def bayesian_hmm_posterior_general(n_draws, r_seed, obs, n_components, alpha=1, beta=0.1):
    """
    HMM posterior for n-regime exponential emission
    :param n_draws: the number of samples
    :param r_seed: random seed
    :param obs: observations
    :param n_components: number of hidden states
    :param alpha: shape parameter of the Gamma prior (default 1)
    :param beta: rate parameter of the Gamma prior (default 0.1)

    :return: posterior distribution of MCMC sampling
    """
    seq_len = len(obs)
    with pm.Model() as model:
        # Prior distributions for emission parameters (param name: lam_i+1)
        lam_rvs = [pm.Gamma(f"lam_{i+1}", alpha=alpha, beta=beta) for i in range(n_components)] 

        # Transition matrix priors
        trans_prior = np.ones(n_components) 
        p_rvs = [pm.Dirichlet(f"p_{i+1}", trans_prior) for i in range(n_components)]
        P_tt = tt.stack(p_rvs)
        P_rv = pm.Deterministic("P_t", tt.shape_padleft(P_tt))

        # Initial state probabilities (assuming first state is S_1=1)
        pi_0_tt = tt.as_tensor(np.r_[1, np.zeros(n_components - 1)]) 
        #S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=np.shape(obs)[-1])
        S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=seq_len)

        # Emission distribution (exponential with lam for each state)
        Y_t = SwitchingProcess("Y_t", [pm.Exponential.dist(lam=lam_rv) for lam_rv in lam_rvs],
                               S_rv, observed=obs)

        # MCMC sampling steps
        transmat_step = TransMatConjugateStep(model.P_t)
        states_step = FFBSStep([model.S_t])
        lam_step = pm.NUTS([model[f"lam_{i+1}"] for i in range(n_components)], target_accept=0.9)

        # MCMC sampling, default burn-in is 2000
        posterior_trace = pm.sample(
            step=[transmat_step, states_step, lam_step],
            chains=1,
            tune = 2000,
            return_inferencedata=True,
            progressbar=False, #no progress bar 
            draws=n_draws,
            random_seed=r_seed,
        )
    
    return posterior_trace

def mean_average(lists):
    """
    Calculate the average of a list of numbers.

    Parameters:
    - lists (list or iterable): A list or iterable of numeric values.

    Returns:
    - float: The average of the numbers in the list. Returns None if the list is empty.
    """
    if not lists:  # Check if the list is empty
        return None
    return sum(lists) / len(lists)

def sigma_average(lists):
    """
    Process a list of numbers by squaring each element, summing them,
    taking the square root of the sum, and then averaging the result.

    Parameters:
    - lists (list or iterable): A list or iterable of numeric values.

    Returns:
    - float: The processed result, or None if the list is empty.
    """
    if not lists:  # Check if the list is empty
        return None

    # Step 1: Square each element
    squared_elements = [x**2 for x in lists]

    # Step 2: Sum the squared elements
    sum_of_squares = sum(squared_elements)

    # Step 3: Take the square root of the sum
    sqrt_of_sum = np.sqrt(sum_of_squares)

    # Step 4: Compute the average
    average = sqrt_of_sum / len(lists)

    return average

def predicted_mean_std_joint_model_general(X_, lambdas_, weights, gp, n_components):
    """
    Calculate the predicted mean and standard deviation for the joint model, supporting any number of n_components states
    :param X_: Input X (N_samples, dimension_x)
    :param lambdas_: A list of lambdas corresponding to each state, shape: (n_components, 1)
    :param weights: State weights for the prediction time, shape: (1, n_components)
    :param gp: Trained Gaussian process model
    :param n_components: The number of states in the HMM
    :return: Predicted mean mu, predicted standard deviation sigma
    """
    weights = weights.reshape(1, -1)

    # Generate (X, 1/lambda) combinations for each state
    D_lambda_list = []
    for i in range(n_components):
        constant_lambda = np.full((X_.shape[0], 1), 1.0 / lambdas_[i])
        D_lambda = np.concatenate((X_, constant_lambda), axis=1)
        D_lambda_list.append(D_lambda)

    mu = np.zeros(X_.shape[0])
    var = np.zeros(X_.shape[0])

    for i in range(X_.shape[0]):  
        D = np.concatenate([D_lambda_list[j][i:(i+1), :] for j in range(n_components)], axis = 0)
        m, k = gp.predict(D, return_cov = True)
        m = m.reshape(-1, 1)

        # Calculate weighted mean and variance
        mu[i] = np.sum(m * np.transpose(weights))
        var[i] = np.matmul(np.matmul(weights, k), np.transpose(weights))

    sigma = np.sqrt(var)
    return mu, sigma

def update_data_switching_joint_general(D_new, D_old, Y_old, switching_distributions, n_rep, method, f, state_index):
    """
    Update data based on the chosen state (specified by state_index).
    
    :param D_new: new design points
    :param D_old: old design points
    :param Y_old: old observed values
    :param switching_distributions: list of distributions for each state
    :param n_rep: number of repetitions for each state
    :param method: evaluation method (used in `f.evaluate`)
    :param f: function for generating data
    :param state_index: the index of the chosen state for evaluation
    :return: updated design points and observations
    """
    
    # Get the distribution for the selected state
    selected_distribution = switching_distributions[state_index]

    # Evaluate the new observed values for the selected state
    Y_new = np.atleast_2d(f.evaluate(D_new, selected_distribution, n_rep, method))
    Y_update = np.concatenate((Y_old, Y_new), axis = 0)

    # Prepare the new design points and include the inverse of the rate (lambda)
    D_new = np.atleast_2d(D_new) 
    D_new1 = np.concatenate((D_new, 1.0/np.array(selected_distribution.get("rate")).reshape(1, -1)), axis=1)
    D_update = np.concatenate((D_old, D_new1), axis=0)

    return D_update, Y_update

def compute_hmm_weights_from_sample_idx(sample_index, obs, n_components, 
                                        posterior_p_list, 
                                        lambdas,
                                        startprob, hmm_class, n_iter=30):
    """
    Construct HMM from a posterior sample index and compute predicted state weights.

    Parameters:
    - sample_index: Index of the posterior sample.
    - obs: Observed data (1D array).
    - n_components: Number of hidden states.
    - posterior_p_list: Transition matrix samples from posterior.
    - lambdas: Emission parameters array of shape (n_components, 1).
    - startprob: Initial state probabilities.
    - hmm_class: HMM class to use.
    - n_iter: Max iterations for HMM (default: 30).

    Returns:
    - weights: Predicted state weights (1 x n_components).
    """
    model = hmm_class(n_components=n_components, random_state=sample_index, n_iter=n_iter)

    transmat = np.array([posterior_p_list[state][0][sample_index] for state in range(n_components)])

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.lambdas_ = lambdas

    predicted_proba = model.predict_proba(obs.reshape(-1, 1))[-1:]
    weights = np.matmul(predicted_proba, model.transmat_)

    return weights

def sample_posterior_exponential_conjugate(observed, num_samples=1000, alpha_prior=1, beta_prior=1):
    """
    Sample from the posterior distribution of the rate parameter λ of the exponential distribution 
    using its conjugate prior (Gamma distribution).

    Parameters:
    - observed: Observed data (numpy array)
    - num_samples: Number of posterior samples to generate (default: 1000)
    - alpha_prior: Shape parameter alpha of the Gamma prior (default: 1)
    - beta_prior: Rate parameter β of the Gamma prior (default: 1)

    Returns:
    - samples: λ samples drawn from the posterior Gamma(alpha + n, β + ∑X) (numpy array)
    """
    n = len(observed)  # Number of data points
    sum_x = np.sum(observed)  # Sum of the observed data

    # Parameters of the posterior distribution
    alpha_post = alpha_prior + n
    beta_post = beta_prior + sum_x

    # Draw samples from Gamma(α + n, β + ∑X)
    samples = stats.gamma.rvs(alpha_post, scale=1 / beta_post, size=num_samples)

    return samples



def GP_model(D,Y):
    

    kernel = ConstantKernel(1.0, (1e-3, 1e1)) * RBF(1e-3, (1e-2, 1e1)) + WhiteKernel(0.1**2)  #covariance kernel
    #kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 100.0),nu=1.5) #+  WhiteKernel(0.05)
    #kernel = 1.0 * Matern(nu=1.5)
    #gp = GaussianProcessRegressor(alpha=dy0/2, n_restarts_optimizer=10) #alpha controls the noise level
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,normalize_y = True) #alpha controls the noise level                                 
    gp.fit(D,Y)
    #print('f(x) kernel:',gp.kernel_)
    
    return gp

#def update_data(x_new,D_old,Y_old, P, n_rep, method,f):
    
    #Y_new = np.atleast_2d(f.evaluate(x_new, P, n_rep, method))
    #Y_update = np.concatenate((Y_old,Y_new),axis = 0)
    #D_new = np.atleast_2d(x_new) 
    #D_update = np.concatenate((D_old, D_new), axis = 0)

    #return D_update, Y_update
  
def update_data_switching(D_new,D_old,Y_old1, Y_old2, swtiching_distributions, n_rep, method,f):
    
    Y_new1 = np.atleast_2d(f.evaluate(D_new, swtiching_distributions[0], n_rep, method))
    Y_new2 = np.atleast_2d(f.evaluate(D_new, swtiching_distributions[1], n_rep, method))

    Y_update1 = np.concatenate((Y_old1,Y_new1),axis = 0)
    Y_update2 = np.concatenate((Y_old2,Y_new2),axis = 0)

    D_new = np.atleast_2d(D_new) 
    D_update = np.concatenate((D_old, D_new), axis = 0)

    return D_update, Y_update1, Y_update2


def update_data_switching_joint(D_new,D_old,Y_old, swtiching_distributions, n_rep, method,f):
    
    Y_new1 = np.atleast_2d(f.evaluate(D_new, swtiching_distributions[0], n_rep, method))
    Y_new2 = np.atleast_2d(f.evaluate(D_new, swtiching_distributions[1], n_rep, method))
    Y_update = np.concatenate((Y_old,Y_new1,Y_new2),axis = 0)

    D_new = np.atleast_2d(D_new) 
    D_new1 = np.concatenate((D_new, 1.0/np.array(swtiching_distributions[0].get("rate")).reshape(1,-1)),axis = 1)
    D_new2 = np.concatenate((D_new, 1.0/np.array(swtiching_distributions[1].get("rate")).reshape(1,-1)),axis = 1)
    D_update = np.concatenate((D_old, D_new1, D_new2), axis = 0)

    return D_update, Y_update

def update_data_switching_joint_single(D_new, D_old, Y_old, swtiching_distributions, n_rep, method, f):
    """
    Update the dataset with new observations for a single switching distribution.
    
    :param D_new: New input data (N, d)
    :param D_old: Existing input data (M, d+1)
    :param Y_old: Existing output data (M, 1)
    :param swtiching_distributions: A single distribution containing the "rate" key
    :param n_rep: Number of replications for function evaluation
    :param method: Evaluation method
    :param f: Function to evaluate
    :return: Updated (D_update, Y_update)
    """
    
    # Evaluate new observations
    Y_new = np.atleast_2d(f.evaluate(D_new, swtiching_distributions, n_rep, method))
    Y_update = np.concatenate((Y_old, Y_new), axis=0)

    # Expand D_new with 1/lambda
    D_new = np.atleast_2d(D_new)
    lambda_inv = 1.0 / np.array(swtiching_distributions.get("rate")).reshape(1, -1)
    D_new_expanded = np.concatenate((D_new, lambda_inv), axis=1)

    # Update dataset
    D_update = np.concatenate((D_old, D_new_expanded), axis=0)

    return D_update, Y_update

def update_data_switching_joint_single_gaussian(D_new, D_old, Y_old, switching_distributions, n_rep, method, f):
    """
    Update the dataset with new observations for a single Gaussian switching distribution.

    :param D_new: New input data (N, d)
    :param D_old: Existing input data (M, d+1)
    :param Y_old: Existing output data (M, 1)
    :param switching_distributions: A single distribution containing the "means" key
    :param n_rep: Number of replications for function evaluation
    :param method: Evaluation method
    :param f: Function to evaluate
    :return: Updated (D_update, Y_update)
    """

    # Evaluate new observations
    Y_new = np.atleast_2d(f.evaluate(D_new, switching_distributions, n_rep, method))
    Y_update = np.concatenate((Y_old, Y_new), axis=0)

    # Expand D_new with mean directly (no inverse)
    D_new = np.atleast_2d(D_new)
    lambda_val = np.array(switching_distributions.get("means")).reshape(1, -1)
    D_new_expanded = np.concatenate((D_new, lambda_val), axis=1)

    # Update dataset
    D_update = np.concatenate((D_old, D_new_expanded), axis=0)

    return D_update, Y_update

def output_Bayes(file_pre, minimizer, minimum_value, f_hat_x, TIME_RE, x_star, f_star, seed, n_sample, iteration, n_xi, S):

    # Prepare for the output of the results
    minimizer = np.array(minimizer) 
    minimum_value = np.array(minimum_value)
    f_hat_x =  np.array(f_hat_x).reshape(-1)
    x_star = np.array(x_star)
    f_star = np.array(f_star)
    xGAP1 = np.linalg.norm(minimizer - x_star,axis=1)
    GAP1 = f_hat_x - f_star
    CumGAP1 = np.cumsum(GAP1)

    # Results save
    TIME_RE = np.array(TIME_RE)
    results = pd.DataFrame(minimizer)
    results.columns = ['x_' + str(i) for i in range(minimizer.shape[1])]
    results['min_post_mean'] = minimum_value
    results['z_hat_x'] = f_hat_x
    results['z_star'] = f_star
    results['time'] = TIME_RE
    results['cum_time'] = np.cumsum(TIME_RE)
    results['n_xi'] = n_xi
    results['seed'] = seed
    results['iteration'] = np.arange(iteration)
    results['regime_status'] = S
    results['rand_xGAP1'] = xGAP1
    results['rand_GAP1'] = GAP1 
    results['cum_rand_GAP1'] = CumGAP1

    results_save(file_pre, results)

import os
def results_save(file_pre, results):

    file_name = file_pre +'_results.csv'
    if os.path.exists(file_name):
        results.to_csv(file_name, header = False, index = False, mode = 'a')
    else:
        results.to_csv(file_name, header = True, index = False)


def results_plot(GAP, n_sample, iteration):

    ### Plots
    SimRegret = list()
    for i in range(len(GAP)):
        SimRegret.append(np.min(GAP[:(i+1)]))
    SimRegret = np.array(SimRegret).reshape(-1,1)
    #AvgRegret = np.cumsum(GAP)/np.arange(1,(len(GAP)+1)).reshape(-1,1)
    CumRegret = np.cumsum(GAP)
    AvgRegret = np.cumsum(GAP)/np.arange(1,GAP.shape[0]+1)

    Budget = range(iteration) 
    plt.figure(figsize= (15,6))
    plt.subplot(2,2,1)
    plt.plot(Budget,CumRegret)
    plt.ylabel('CumRegret')
    plt.xlabel('Budget')   
    plt.axhline(0,color='black',ls='--')
    #plt.xlim(n_sample,N_total)
    #plt.ylim(-0.1,4)
    ####################################
    plt.subplot(2,2,2)
    plt.plot(Budget,AvgRegret)
    plt.ylabel('AvgRegret')
    plt.xlabel('Budget')   
    plt.axhline(0,color='black',ls='--')
    ####################################
    plt.subplot(2,2,3)
    plt.plot(Budget,SimRegret)
    plt.ylabel('SimRegret')
    plt.xlabel('Budget')   
    plt.axhline(0,color='black',ls='--')
    ####################################
    plt.subplot(2,2,4)
    plt.plot(Budget,GAP)
    plt.ylabel('GAP')
    plt.xlabel('Budget')   
    plt.axhline(0,color='black',ls='--')
    plt.show();

#def MAP_exponential(observed): #added single exponential distribution MAP
    
    #with pm.Model() as model:
        # 选择 Gamma 先验作为 λ 参数的先验分布
        #alpha_prior = 1  # Gamma 形状参数
        #beta_prior = 1   # Gamma 尺度参数
        #lam = pm.Gamma('lam', alpha=alpha_prior, beta=beta_prior)  # 直接定义 λ

        # 定义指数分布的似然，使用 observed 作为数据
        #likelihood = pm.Exponential('data', lam=lam, observed=observed)

        # 使用 find_MAP 来找到 MAP 估计
        #map_estimate = pm.find_MAP()

    # 返回估计的 λ 参数值对应的随机变量
    #lambda_map = map_estimate['lam'].item()
    #rv = stats.expon(scale = 1/lambda_map)
    #return rv

#def MAP_and_generate_exponential(xi, size = 1000): #added single exponential distribution generation
    
    #rv = MAP_exponential(xi)
    #xi = generate_sample_general(size, rv)

    #return xi

class synthetic_uni_problem: # added on Apr 17, 2025
    
    def __init__(self, true_dist):
        self.dimension_x = 1
        self.dimension_p = 1
        self.lb = [0]
        self.ub = [100] ##change on Jun5
        self.true_dist = true_dist  
        self.x_star = None
        self.f_star = None

    def __call__(self, xx):
        value = self.evaluate_true(xx)
        return value

    def evaluate(self, xx, P, n_rep, method, **kwargs):
        """
        xx: decision variable
        P: distribution info (could be samples or parameter dict)
        n_rep: number of simulation replications
        method: simulation method
        kwargs: extra keyword args passed to simulator
        """
        x = xx[0]
        value = synthetic_simulator1(x, P, n_rep, method, **kwargs)
        return value

    def evaluate_true(self, xx):
        """
        解析目标函数期望
        """
        x = xx[0]
        theta_c = self.true_dist.get('means')
        known_sigma = self.true_dist.get('sigmas')
        value = synthetic_true_objective1(x, theta_c, known_sigma)
        return value

    #def estimate_minimizer_minimum(self, n_sample=1000):
        # this function is same as inventory problem (LHS sampling & evaluate_true)
        #np.random.seed(None)
        #lhd = lhs(self.dimension_x, samples=n_sample, criterion="maximin")
        #X_ = np.zeros((n_sample, self.dimension_x))
        #for i in range(self.dimension_x):
            #X_[:, i] = lhd[:, i] * (self.ub[i] - self.lb[i]) + self.lb[i]
        #Y_ = np.zeros(X_.shape[0])
        #for i in range(X_.shape[0]):
            #Y_[i] = self.evaluate_true(X_[i])
        #self.f_star = min(Y_)
        #self.x_star = X_[np.argmin(Y_)]
        #return self.f_star, self.x_star
    
    def estimate_minimizer_minimum(self):
        def objective(x):
            return self.evaluate_true(x)

        lb_array = np.array(self.lb)
        ub_array = np.array(self.ub)
        bounds = list(zip(lb_array, ub_array))
        x0 = (lb_array + ub_array) / 2

        result = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')

        self.f_star = result.fun
        self.x_star = result.x

        return self.f_star, self.x_star

def synthetic_simulator1(x, P, n_rep, method, **kwargs):
    """
    A synthetic simulator that evaluates the expected objective value based on sampled input uncertainties.

    Parameters:
    - x: Decision variable (can be scalar or array-like depending on the objective function).
    - P: Distribution parameters or data, depending on the method.
         For example:
            - if method == 'bayesian_hist': P is observed historical data.
            - if method == 'switching' or 'switching_joint': P is a dictionary with keys 'means', 'sigmas', and 'dist'.
            - if method == 'posterior_mean_normal': P is a posterior object used to generate samples.
    - n_rep: Number of simulation replications
    - method: Sampling and evaluation method. Supported options:
        - 'bayesian_hist': KDE-based sampling from historical data.
        - 'switching', 'switching_joint': Sampling from Gaussian distribution with given parameters.
        - 'posterior_mean_normal': Sampling from posterior predictive distribution.

    Returns:
    - value: The average of evaluated objective values over sampled input uncertainties.
    """
    
    if method == 'bayesian_hist':  # KDE-based sampling from historical data
        xi_samples = kde_sample_general(P, size=n_rep, **kwargs)

    elif method in ["switching", "switching_joint"]:
        if P.get('dist') == "gaussian":
            rv = stats.norm(loc=P.get('means'), scale=P.get('sigmas')) 
            xi_samples = generate_sample_general(n_rep, rv)

    elif method == 'posterior_mean_normal':
        xi_samples = posterior_mean_and_generate_gaussian(xi=P, size=n_rep, **kwargs) 

    else:
        raise KeyError('Wrong input!')

    h_vals = synthetic_objective1(x, xi_samples)
    value = np.mean(h_vals)
    return value

def synthetic_objective1(x, xi):
    ##follow Enlu Zhou 2024 OR: Data-Driven Ranking and Selection Under IU, but add 10*xi to make the different optimal objective value (updated on May 6, 2025)
    return (x-xi)**2 + 10*xi 

def synthetic_true_objective1(x, theta_c, known_sigma):

    return (x-theta_c)**2 + known_sigma**2 + 10*theta_c

def kde_sample_general(xi, size=1000, seed=None): # newly added Apr 22, 2025
    """
    Fit a KDE to real-valued data and generate new samples.

    Parameters:
    - xi: Input data (numpy array, can be any real values)
    - size: Number of samples to generate (default: 1000)
    - seed: Random seed for reproducibility (default: None)

    Returns:
    - samples: A (size,) numpy array of random samples
    """
    if len(xi) == 0:
        raise ValueError("Input data is empty")
    
    # Set the random seed if provided
    if seed is not None:
        np.random.seed(seed)

    kde = stats.gaussian_kde(xi)
    samples = kde.resample(size).reshape(-1)
    return samples

def update_kde_and_plot(xi, save_path=None, seed=None): # newly added Apr 22, 2025
    """
    Fit a KDE to the data and plot it. Optionally save the plot to the specified path.

    Parameters:
    - xi: Input data (numpy array)
    - save_path: If provided, save the plot to this path
    - seed: If provided, set the random seed to ensure reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    kde = stats.gaussian_kde(xi)
    x_grid = np.linspace(min(xi) - 1, max(xi) + 1, 1000)
    kde_values = kde(x_grid)

    plt.figure(figsize=(8, 4))
    plt.plot(x_grid, kde_values, label='KDE fit', color='blue')
    plt.hist(xi, bins=30, density=True, alpha=0.4, label='Original data', color='gray')
    plt.title('KDE Fit and Original Data Histogram')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()

def update_data(x_new, D_old, Y_old, P, n_rep, method, f, **kwargs): ##updated in Apr22, 2025
    """
    Update the dataset with new input x_new and corresponding simulated output using function f.

    Parameters:
    - x_new: New decision variable
    - D_old: Previous decision variable data
    - Y_old: Previous objective values
    - P: Distribution parameters or data
    - n_rep: Number of simulation replications
    - method: Evaluation method
    - f: Object with .evaluate() method
    - **kwargs: Additional optional arguments (e.g., seed) for fixed fitted kde and ploting

    Returns:
    - D_update: Updated decision variable data
    - Y_update: Updated objective values
    """
    Y_new = np.atleast_2d(f.evaluate(x_new, P, n_rep, method, **kwargs))
    Y_update = np.concatenate((Y_old, Y_new), axis=0)
    D_new = np.atleast_2d(x_new)
    D_update = np.concatenate((D_old, D_new), axis=0)

    return D_update, Y_update


def posterior_mean_gaussian_fast(observed, mu_prior=5, sigma_prior=5, sigma_known=4):
    """
    Compute the posterior mean of the normal distribution when variance is known,
    and return a scipy.stats.norm distribution using the posterior mean as the mean.

    Parameters:
    - observed: Observed data (numpy array)
    - mu_prior: Prior mean of μ
    - sigma_prior: Prior standard deviation of μ (i.e., sqrt of prior variance)
    - sigma_known: Known standard deviation of the normal likelihood

    Returns:
    - rv: A scipy.stats.norm object using posterior mean as mean and known sigma
    """
    n = len(observed)
    sample_mean = np.mean(observed)
    
    # Convert standard deviations to variances
    var_prior = sigma_prior ** 2
    var_known = sigma_known ** 2

    # Compute posterior mean and variance (conjugate prior)
    posterior_variance = 1 / (n / var_known + 1 / var_prior)
    posterior_mean = posterior_variance * (n * sample_mean / var_known + mu_prior / var_prior)

    # Return distribution with posterior mean and known sigma
    rv = stats.norm(loc=posterior_mean, scale=sigma_known)
    return rv

def sample_posterior_normal_mean_conjugate(observed, num_samples=1000, mu_prior=5, sigma_prior=5, sigma_known=4):
    """
    Sample from the posterior distribution of the mean μ of a normal distribution with known variance,
    using the conjugate prior (normal prior).

    Parameters:
    - observed: Observed data (numpy array)
    - num_samples: Number of samples to draw from the posterior (default: 1000)
    - mu_prior: Prior mean of μ
    - sigma_prior: Prior standard deviation of μ
    - sigma_known: Known standard deviation of the likelihood (i.e., data distribution)

    Returns:
    - samples: μ samples drawn from the posterior distribution (numpy array)
    """
    n = len(observed)
    sample_mean = np.mean(observed)

    var_prior = sigma_prior ** 2
    var_known = sigma_known ** 2

    posterior_variance = 1 / (n / var_known + 1 / var_prior)
    posterior_mean = posterior_variance * (n * sample_mean / var_known + mu_prior / var_prior)

    posterior_std = np.sqrt(posterior_variance)
    
    # Draw samples from the posterior Normal distribution
    samples = np.random.normal(loc=posterior_mean, scale=posterior_std, size=num_samples)
    
    return samples

def posterior_mean_and_generate_gaussian(xi, size, mu_prior=5, sigma_prior=5, sigma_known=4):
    """
    Generate samples from a normal distribution where the mean is estimated by the posterior,
    and the variance is known.

    Parameters:
    - xi: Observed data (array)
    - size: Number of samples to generate
    - mu_prior: Prior mean of μ
    - sigma_prior: Prior std dev of μ
    - sigma_known: Known std dev of the normal distribution

    Returns:
    - xi_samples: Generated samples (numpy array)
    """
    rv = posterior_mean_gaussian_fast(xi, mu_prior, sigma_prior, sigma_known)
    xi_samples = generate_sample_general(size, rv)
    return xi_samples

def compute_posterior_mean_gaussian(observed, mu_prior=5, sigma_prior=5, sigma_known=4):
    """
    Compute the posterior mean of the normal distribution when variance is known.

    Parameters:
    - observed: Observed data (numpy array)
    - mu_prior: Prior mean of μ
    - sigma_prior: Prior standard deviation of μ
    - sigma_known: Known standard deviation of the normal likelihood

    Returns:
    - posterior_mean: The posterior mean of the distribution
    """
    n = len(observed)
    sample_mean = np.mean(observed)

    # Convert standard deviations to variances
    var_prior = sigma_prior ** 2
    var_known = sigma_known ** 2

    # Compute posterior mean (conjugate prior formula)
    posterior_variance = 1 / (n / var_known + 1 / var_prior)
    posterior_mean = posterior_variance * (n * sample_mean / var_known + mu_prior / var_prior)

    return posterior_mean

def switching_real_world_data(true_dist, timestep, random_state = 0): 
    ###add gaussian emission distributions

    if true_dist.get("emission") == "exp":
        # Build an HMM instance and set parameters
        model = ExponentialHMM(n_components=true_dist.get("n_components"), random_state = random_state)
        model.startprob_ = true_dist.get("startprob")
        model.transmat_ = true_dist.get("transmat")
        model.lambdas_ = true_dist.get("lambdas")

    elif true_dist.get("emission") == "gaussian":
        
        model = GaussianHMM(n_components=true_dist.get("n_components"), covariance_type="diag", 
                            random_state=random_state)
        model.startprob_ = true_dist.get("startprob")
        model.transmat_ = true_dist.get("transmat")
        model.means_ = true_dist.get("means")
        model.covars_ = true_dist.get("covars")
    
    else:
        raise NotImplementedError("Not implemented")


    # Generate samples
    X, Z = model.sample(timestep)

    return X, Z

def bayesian_hmm_posterior_gaussian_known_var(n_draws, r_seed, obs, n_components,
                                               mu_prior_mean=5, mu_prior_std=5,
                                               known_std=4, startprob=None):
    """
    HMM posterior for n-regime Gaussian emission with known variance.
    :param n_draws: number of MCMC draws
    :param r_seed: random seed
    :param obs: observations (1D)
    :param n_components: number of hidden states
    :param mu_prior_mean: prior mean of Gaussian prior for emission means
    :param mu_prior_std: prior std dev of Gaussian prior for emission means
    :param known_std: known standard deviation of Gaussian emissions
    :param startprob: initial state probability vector (1D array of size n_components)
    
    :return: posterior distribution of MCMC sampling
    """
    seq_len = len(obs)
    
    if startprob is None:
        raise ValueError("startprob must be explicitly provided. No default is allowed.")

    with pm.Model() as model:
        # Prior over means of Gaussian emissions
        mu_rvs = [pm.Normal(f"mu_{i+1}", mu=mu_prior_mean, sigma=mu_prior_std) for i in range(n_components)]

        # Transition matrix
        trans_prior = np.ones(n_components)
        p_rvs = [pm.Dirichlet(f"p_{i+1}", a=trans_prior) for i in range(n_components)]
        P_tt = tt.stack(p_rvs)
        P_rv = pm.Deterministic("P_t", tt.shape_padleft(P_tt))

        # Initial state probabilities
        pi_0_tt = tt.as_tensor(startprob)
        S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=seq_len)

        # Switching Gaussian emission process
        Y_t = SwitchingProcess("Y_t", 
            [pm.Normal.dist(mu=mu_rv, sigma=known_std) for mu_rv in mu_rvs], 
            S_rv, 
            observed=obs)

        # Sampling steps
        transmat_step = TransMatConjugateStep(model.P_t)
        states_step = FFBSStep([model.S_t])
        mu_step = pm.NUTS([model[f"mu_{i+1}"] for i in range(n_components)], target_accept=0.9)

        posterior_trace = pm.sample(
            step=[transmat_step, states_step, mu_step],
            chains=1,
            tune=1000, ##updated from 2000 to 1000, speed up
            draws=n_draws,
            random_seed=r_seed,
            return_inferencedata=True,
            progressbar=False
        )

    return posterior_trace

def compute_hmm_weights_from_sample_idx_gaussian(sample_index, obs, n_components, 
                                                 posterior_p_list, 
                                                 means, 
                                                 startprob, known_var=16):
    """
    Construct Gaussian HMM from a posterior sample index and compute predicted state weights.

    Parameters:
    - sample_index: Index of the posterior sample.
    - obs: Observed data (1D array).
    - n_components: Number of hidden states.
    - posterior_p_list: List of transition matrix samples from posterior, one per state.
    - means: Emission means parameters array.
    - startprob: Initial state probabilities.
    - known_var: Known variance (float, default: 16).

    Returns:
    - weights: Predicted state weights (1 x n_components).
    """
    model = GaussianHMM(n_components=n_components, covariance_type="diag", 
                        random_state=sample_index)

    transmat = np.array([posterior_p_list[state][0][sample_index] for state in range(n_components)])

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = np.full((n_components, 1), known_var)  # shape (n_components, 1)

    predicted_proba = model.predict_proba(obs.reshape(-1, 1))[-1:]
    weights = np.matmul(predicted_proba, model.transmat_)

    return weights

def predicted_mean_std_joint_model_general_lambda_direct(X_, lambdas_, weights, gp, n_components):
    """
    Calculate the predicted mean and standard deviation for the joint model, using lambda directly
    (instead of 1/lambda) as the regime feature.(no reciprocal of parameter since normal mean set >=1)

    :param X_: Input X (N_samples, dimension_x)
    :param lambdas_: A list or array of lambdas corresponding to each state, shape: (n_components, 1) or (n_components,)
    :param weights: State weights for the prediction time, shape: (1, n_components)
    :param gp: Trained Gaussian process model
    :param n_components: The number of states in the HMM
    :return: Predicted mean mu, predicted standard deviation sigma
    """
    weights = weights.reshape(1, -1)

    # Generate (X, lambda) combinations for each state
    D_lambda_list = []
    for i in range(n_components):
        constant_lambda = np.full((X_.shape[0], 1), lambdas_[i])  # ✅ use lambda directly
        D_lambda = np.concatenate((X_, constant_lambda), axis=1)
        D_lambda_list.append(D_lambda)

    mu = np.zeros(X_.shape[0])
    var = np.zeros(X_.shape[0])

    for i in range(X_.shape[0]):  
        D = np.concatenate([D_lambda_list[j][i:(i+1), :] for j in range(n_components)], axis=0)
        m, k = gp.predict(D, return_cov=True)
        m = m.reshape(-1, 1)

        # Calculate weighted mean and variance
        mu[i] = np.sum(m * np.transpose(weights))
        var[i] = np.matmul(np.matmul(weights, k), np.transpose(weights))

    sigma = np.sqrt(var)
    return mu, sigma

def update_data_switching_joint_general_gaussian(D_new, D_old, Y_old, switching_distributions, n_rep, method, f, state_index):#no reciprocal
    """
    Update data based on the chosen state (specified by state_index), using Gaussian mean as the regime feature. (no reciprocal of parameter since normal mean set >=1)
    :param D_new: new design points
    :param D_old: old design points
    :param Y_old: old observed values
    :param switching_distributions: list of distributions for each state
    :param n_rep: number of repetitions for each state
    :param method: evaluation method (used in `f.evaluate`)
    :param f: function for generating data
    :param state_index: the index of the chosen state for evaluation
    :return: updated design points and observations
    """

    # Get the distribution for the selected state
    selected_distribution = switching_distributions[state_index]

    # Evaluate the new observed values for the selected state
    Y_new = np.atleast_2d(f.evaluate(D_new, selected_distribution, n_rep, method))
    Y_update = np.concatenate((Y_old, Y_new), axis=0)

    # Prepare the new design points and include the mean as regime feature
    D_new = np.atleast_2d(D_new)
    D_new1 = np.concatenate((D_new, np.array(selected_distribution.get("means")).reshape(1, -1)), axis=1)
    D_update = np.concatenate((D_old, D_new1), axis=0)

    return D_update, Y_update


