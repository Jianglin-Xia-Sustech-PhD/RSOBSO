import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from hmmlearn import hmm
from switching_code.exponentialhmm import ExponentialHMM 
from hmmlearn.hmm import PoissonHMM
import pandas as pd
import scipy.stats as stats
import scipy
from pyDOE import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern,WhiteKernel,ConstantKernel
import pymc3 as pm 
import os
from sklearn import decomposition
import warnings
warnings.filterwarnings("ignore") 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #find path from upper layer

import patsy
import theano
import theano.tensor as tt
from theano import shared
from pymc3_hmm.distributions import SwitchingProcess, DiscreteMarkovChain
from pymc3_hmm.step_methods import FFBSStep, TransMatConjugateStep

####updated on May14, 2025, add **kwargs for logkde, adjust prior change for inventory class
####updated on May23, 2025, bayesian_hmm_posterior_general function, add startprob and decrease tune to 1000

def inital_design(n_sample, seed, lower_bound, upper_bound):  
    # this function aims to use latin-hypercube design and 'maxmin' criterion to generate initial design points
    np.random.seed(seed)
    dimension_x = len(lower_bound)
    lhd= lhs(dimension_x,samples = n_sample,criterion = "maximin")
    D1 = np.zeros((n_sample, dimension_x))
    for i in range(dimension_x): 
        D1[:,i] = lhd[:,i]*(upper_bound[i]-lower_bound[i]) + lower_bound[i]
    return D1

def generate_lambda(n_sample, seed, lambda_lower_bounds, lambda_upper_bounds):
    """
    生成多维参数 lambda
    
    参数：
    - n_sample: 样本数量
    - seed: 随机种子
    - lambda_lower_bounds: 参数每维的下界，数组或列表，长度为参数维度
    - lambda_upper_bounds: 参数每维的上界，数组或列表，长度为参数维度
    
    返回：
    - shape (n_sample, dimension) 的拉丁超立方采样点
    """
    np.random.seed(seed)
    dimension = len(lambda_lower_bounds)
    lhd = lhs(dimension, samples=n_sample, criterion="maximin")
    
    lambda_samples = np.zeros((n_sample, dimension))
    for i in range(dimension):
        lambda_samples[:, i] = lhd[:, i] * (lambda_upper_bounds[i] - lambda_lower_bounds[i]) + lambda_lower_bounds[i]
    
    return lambda_samples


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

def EGO_joint(X_lambda, mu_AEI, mu, sigma):
    """
    EGO acquisition function for joint input (X, lambda).

    Parameters:
    - X_lambda: (N_samples, d+1) array of candidate points (x concatenated with lambda)
    - mu_AEI: (N_samples,) Expected Improvement at each candidate
    - mu: (N_samples,) GP predicted mean at each candidate
    - sigma: (N_samples,) GP predicted std dev at each candidate

    Returns:
    - X_new: the selected (x, lambda) point maximizing EI
    """
    sigma = sigma.reshape(-1)
    mu_AEI = mu_AEI.reshape(-1)
    mu = mu.reshape(-1)

    mu_min = np.min(mu_AEI)  # best improvement so far

    # Avoid zero sigma for numerical stability
    sigma[sigma == 0] = np.finfo(float).eps

    delta = mu_min - mu

    EGO_crit = delta * scipy.stats.norm.cdf(delta / sigma) + sigma * scipy.stats.norm.pdf(delta / sigma)

    index = np.argmax(EGO_crit)
    X_new = X_lambda[index]

    return X_new


def switching_real_world_data(true_dist, timestep, random_state = 0):


    if true_dist.get("emission") == "exp":
        # Build an HMM instance and set parameters
        model = ExponentialHMM(n_components=true_dist.get("n_components"), random_state = random_state)

    elif true_dist.get("emission") == "poisson":
        
        model = PoissonHMM(n_components=true_dist.get("n_components"), random_state = random_state)
    
    else:
        raise NotImplementedError("Not implemented")

    # Instead of fitting it from the data, we directly set the estimated
    # parameters, the means and covariance of the components
    model.startprob_ = true_dist.get("startprob")
    model.transmat_ = true_dist.get("transmat")
    model.lambdas_ = true_dist.get("lambdas")

    # Generate samples
    X, Z = model.sample(timestep)

    return X, Z

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
    
    def evaluate(self, xx, P, n_rep, method, **kwargs):
        s = xx[0]
        S = xx[1]
        value = cost_simulator(s,S,P, n_rep, method, **kwargs)[0]

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

def cost_simulator(s,S,P,n_rep=10, method = 'NBRO',NOP=1000,b=100,h=1,c=1,K=100, **kwargs):
    #h holding cost per period per unit of inventory
    #b shortage cost per period per unit of inventory
    #c per-unit ordering cost
    #K setup cost for placing an order

    s = s
    S = S
    output=np.zeros(n_rep)

    for i in np.arange(n_rep):
        output[i]= costsim(NOP,P,b,h,c,K,s,S,method, **kwargs)

    cost_y= np.mean(output); # take the average over all replications (x-bar[n])
    cost_v= np.var(output)/n_rep; # noise withe rep replications
    
    return cost_y,cost_v

def costsim(NOP,P,b,h,c,K,s,S,method, **kwargs): ####controlled by 'method'
    ## types ####D: generate NOP samples by the fitted distribution (number of periods)
    burnin = 100

    if method == 'hist': # 'histogram'
        D = np.random.choice(P, size = NOP)
    elif method == 'bayesian_hist': #kde fit pdf
        D = kde_sample(P, size = NOP, **kwargs)
    elif method == 'exp': # "parametric"
        D = fit_and_generate_exponential(xi = P, size = NOP)
    elif method == 'MAP_exp': # exponential MAP bayesian
        D = MAP_and_generate_exponential(xi = P, size = NOP)
    elif method == 'posterior_mean_exp':
        D = posterior_mean_and_generate_exponential(xi = P, size = NOP, **kwargs)
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

def posterior_mean_exponential(observed, r_seed=None, draws=1000, chains=1):
    """
    使用 PyMC3 进行指数分布参数 λ 的贝叶斯推断，并返回基于后验均值的指数分布随机变量。

    参数:
    - observed: 观测数据 (numpy array)
    - r_seed: 随机种子 (默认 None)
    - draws: MCMC 采样次数 (默认 3000)
    - chains: MCMC 采样链数 (默认 1)

    返回:
    - rv: 以后验均值估计的 λ 作为尺度参数的 scipy.stats.expon 分布对象
    """
    with pm.Model() as model:
        # Gamma 先验分布参数
        alpha_prior = 1  # 形状参数
        beta_prior = 1   # 尺度参数
        
        # 定义 λ 的 Gamma 先验
        lam = pm.Gamma('lam', alpha=alpha_prior, beta=beta_prior)
        
        # 观测数据的似然函数 (指数分布)
        likelihood = pm.Exponential('data', lam=lam, observed=observed)

        # 进行 MCMC 采样
        trace = pm.sample(draws=draws, chains=chains, return_inferencedata=True,
                          progressbar=False, random_seed=r_seed)
    
    # 计算 λ 的后验均值
    posterior_mean = trace.posterior['lam'].mean().item()
    
    # 以后验均值 1/posterior_mean 作为尺度参数，返回指数分布随机变量
    rv = stats.expon(scale=1/posterior_mean)  
    return rv

def posterior_mean_exponential_fast(observed, alpha_prior=1, beta_prior=1):
    """
    直接计算指数分布参数 λ 的后验均值，并返回基于后验均值的指数分布对象。

    参数:
    - observed: 观测数据 (numpy array)
    - alpha_prior: Gamma 先验的形状参数 (默认 1)
    - beta_prior: Gamma 先验的尺度参数 (默认 1)

    返回:
    - rv: 以后验均值 作为 lambda 估计的 scipy.stats.expon 分布对象
    """
    n = len(observed)  # 观测数据数量
    sum_x = np.sum(observed)  # 观测数据的总和
    
    # 计算 λ 的后验均值
    posterior_mean = (alpha_prior + n) / (beta_prior + sum_x)
    
    # 计算指数分布的 scale 参数
    scale = 1 / posterior_mean  # 修正这里
    
    # 创建指数分布对象
    rv = stats.expon(scale=scale)
    return rv

def compute_posterior_mean_exponential(observed, alpha_prior=1, beta_prior=1):
    """
    计算指数分布参数 λ 的后验均值。

    参数:
    - observed: 观测数据 (numpy array)
    - alpha_prior: Gamma 先验的形状参数 (默认 1)
    - beta_prior: Gamma 先验的尺度参数 (默认 1)

    返回:
    - posterior_mean: 指数分布 λ 的后验均值
    """
    n = len(observed)         # 观测数据数量
    sum_x = np.sum(observed)  # 观测数据的总和

    # 计算 λ 的后验均值
    posterior_mean = (alpha_prior + n) / (beta_prior + sum_x)

    return posterior_mean

def posterior_mean_and_generate_exponential(xi, size = 1000, alpha_prior=1, beta_prior=1): 
    
    rv = posterior_mean_exponential_fast(xi, alpha_prior, beta_prior) ##fix, self compute
    xi = generate_sample_general(size, rv)

    return xi

def kde_sample(xi, size=1000, seed=None): # newly updated May 12, 2025
    """
    Fit a KDE to positive real-valued data and generate new positive samples.

    Parameters:
    - xi: Input data (numpy array, must be positive values > 0)
    - size: Number of samples to generate (default: 1000)
    - seed: Random seed for reproducibility (default: None)

    Returns:
    - samples: A (size,) numpy array of positive random samples
    """
    # Remove non-positive values
    xi = xi[xi > 0]

    if len(xi) == 0:
        raise ValueError("Input data must contain positive values")

    # Set the random seed if provided
    if seed is not None:
        np.random.seed(seed)

    log_xi = np.log(xi)  # Log-transform the data
    kde = stats.gaussian_kde(log_xi)
    log_samples = kde.resample(size).reshape(-1)
    samples = np.exp(log_samples)  # Exponentiate to get back to positive values
    return samples

def update_kde_positive_and_plot(xi, save_path=None, seed=None): 
    """
    Fit a KDE (in log-space) to positive data and plot the results.
    Optionally save the plot to the specified path.

    Parameters:
    - xi: Input data (numpy array, must be positive values > 0)
    - save_path: If provided, save the plot to this path
    - seed: If provided, set the random seed to ensure reproducibility
    """
    # Remove non-positive values
    xi = xi[xi > 0]

    if len(xi) == 0:
        raise ValueError("Input data must contain positive values.")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Log-transform
    log_xi = np.log(xi)
    kde = stats.gaussian_kde(log_xi)

    # Create grid for log-space and transform back
    log_x_grid = np.linspace(min(log_xi) - 1, max(log_xi) + 1, 1000)
    x_grid = np.exp(log_x_grid)
    kde_values = kde(log_x_grid) / x_grid  # Jacobian correction

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(x_grid, kde_values, label='KDE fit (positive data)', color='blue')
    plt.hist(xi, bins=30, density=True, alpha=0.4, label='Original data', color='gray')
    plt.title('KDE Fit and Original Positive Data Histogram')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
        plt.savefig(save_path)
    plt.close()

#change prior gamma(1, 0.1) both, allow general number of regimes
def bayesian_hmm_posterior_general(n_draws, r_seed, obs, n_components,
                                    alpha=1, beta=0.1, startprob=None):
    """
    HMM posterior for n-regime exponential emission

    :param n_draws: the number of samples
    :param r_seed: random seed
    :param obs: observations
    :param n_components: number of hidden states
    :param alpha: shape parameter of the Gamma prior (default 1)
    :param beta: rate parameter of the Gamma prior (default 0.1)
    :param startprob: initial state distribution
    :return: posterior distribution of MCMC sampling
    """
    seq_len = len(obs)

    if startprob is None:
        raise ValueError("startprob must be explicitly provided. No default is allowed.")

    with pm.Model() as model:
        # Emission parameters
        lam_rvs = [pm.Gamma(f"lam_{i+1}", alpha=alpha, beta=beta) for i in range(n_components)] 

        # Transition matrix
        trans_prior = np.ones(n_components)
        p_rvs = [pm.Dirichlet(f"p_{i+1}", trans_prior) for i in range(n_components)]
        P_tt = tt.stack(p_rvs)
        P_rv = pm.Deterministic("P_t", tt.shape_padleft(P_tt))

        # Initial state distribution
        pi_0_tt = tt.as_tensor(startprob)
        S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=seq_len)

        # Emissions
        Y_t = SwitchingProcess("Y_t", [pm.Exponential.dist(lam=lam_rv) for lam_rv in lam_rvs],
                               S_rv, observed=obs)

        # Sampling steps
        transmat_step = TransMatConjugateStep(model.P_t)
        states_step = FFBSStep([model.S_t])
        lam_step = pm.NUTS([model[f"lam_{i+1}"] for i in range(n_components)], target_accept=0.9)

        # MCMC sampling with fixed tune=1000
        posterior_trace = pm.sample(
            step=[transmat_step, states_step, lam_step],
            chains=1,
            tune=1000,
            draws=n_draws,
            return_inferencedata=True,
            progressbar=False,
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

def compute_diag_kernel(kernel_func, X1, X2):
    # X1, X2 shape: (N, d+1)
    # 返回对应点对点的核值向量，长度 N
    return np.array([kernel_func(X1[i:i+1], X2[i:i+1])[0,0] for i in range(X1.shape[0])])

def predicted_mc_mean_vectorized(X_, lambdas_all, weights_all, gp):
    """
    Compute Monte Carlo averaged posterior mean of GP at input X_.

    Parameters:
    - X_: (N_samples, d) array of input design points
    - lambdas_all: (N_MC, n_components) array of posterior lambda samples
    - weights_all: (N_MC, n_components) array of posterior weights samples
    - gp: trained GaussianProcessRegressor

    Returns:
    - mu: (N_samples,) posterior mean at each input x
    """

    N_samples, d = X_.shape
    N_MC, n_components = lambdas_all.shape

    # Step 1: Flatten 1/lambda samples
    lambdas_inv_flat = 1.0 / lambdas_all.reshape(-1)  # shape: (N_MC * n_components,)

    # Step 2: Tile for each x
    lambdas_inv_tiled = np.tile(lambdas_inv_flat, N_samples).reshape(-1, 1)  # (N_samples * N_MC * n, 1)

    # Step 3: Repeat X_
    X_repeat = np.repeat(X_, N_MC * n_components, axis=0)  # (N_samples * N_MC * n, d)

    # Step 4: Final GP input
    X_lambda_all = np.hstack((X_repeat, lambdas_inv_tiled))  # (N_samples * N_MC * n, d+1)

    # Step 5: Predict GP mean
    m_all = gp.predict(X_lambda_all).reshape(N_samples, N_MC, n_components).transpose(1, 2, 0)  # (N_MC, n, N_samples)

    # Step 6: Weighted mean
    weighted_means = np.sum(weights_all[:, :, None] * m_all, axis=1)  # (N_MC, N_samples)
    mu = np.mean(weighted_means, axis=0)  # (N_samples,)

    return mu

def predicted_mc_mean_single_regime(X_, lambdas_all, gp):
    """
    Compute Monte Carlo averaged posterior mean of GP at input X_, 
    under the assumption of only one regime (no mixture).

    Parameters:
    - X_: (N_samples, d) array of input design points
    - lambdas_all: (N_MC,) array of posterior lambda samples
    - gp: trained GaussianProcessRegressor

    Returns:
    - mu: (N_samples,) posterior mean at each input x
    """

    N_samples, d = X_.shape
    N_MC = lambdas_all.shape[0]

    # Step 1: 1/lambda
    lambdas_inv = 1.0 / lambdas_all  # (N_MC,)

    # Step 2: Repeat X_ for each lambda sample
    X_repeat = np.repeat(X_, N_MC, axis=0)  # (N_samples * N_MC, d)

    # Step 3: Tile lambdas for each x
    lambdas_tiled = np.tile(lambdas_inv, (N_samples, 1)).reshape(-1, 1)  # (N_samples * N_MC, 1)

    # Step 4: Final input to GP
    X_lambda_all = np.hstack((X_repeat, lambdas_tiled))  # (N_samples * N_MC, d+1)

    # Step 5: GP mean prediction
    m_all = gp.predict(X_lambda_all).reshape(N_samples, N_MC)  # (N_samples, N_MC)

    # Step 6: Average over MC
    mu = np.mean(m_all, axis=1)  # (N_samples,)

    return mu

def predicted_mean_single(X_, lambdas_, weights_, gp):
    """
    Compute the posterior mean of GP using a single set of lambda and weights.

    Parameters:
    - X_: (N_samples, d) input design points
    - lambdas_: (n_components,) array of lambda values (interpolated)
    - weights_: (n_components,) array of weights (interpolated)
    - gp: trained GaussianProcessRegressor

    Returns:
    - mu: (N_samples,) posterior mean
    """
    N_samples, d = X_.shape
    n_components = lambdas_.shape[0]

    # 1. Construct GP inputs (x, 1/lambda)
    X_list = []
    for i in range(n_components):
        lambda_inv = 1.0 / lambdas_[i]
        x_aug = np.hstack([X_, np.full((N_samples, 1), lambda_inv)])  # (N_samples, d+1)
        X_list.append(x_aug)

    # 2. Stack all: (N_samples * n_components, d+1)
    X_all = np.vstack(X_list)

    # 3. Predict GP: (N_samples * n_components,)
    m_all = gp.predict(X_all).reshape(n_components, N_samples)

    # 4. Weighted mean: (N_samples,)
    mu = np.dot(weights_, m_all)  # (N_samples,)

    return mu


def predicted_mc_mean_and_approx_var_vectorized(X_, lambda_, lambdas_all, weights_all, gp, n_rep):

    N_samples, d = X_.shape
    N_MC, n_components = lambdas_all.shape

    ### --- Construct input (x, 1/lambda) combinations ---
    # Step 1: Prepare all (1/lambda) as flat vector: shape (N_MC * n_components,)
    lambdas_inv_flat = 1.0 / lambdas_all.reshape(-1) 

    # Step 2: Tile this to match X_: final shape (N_samples * N_MC * n_components, 1)
    lambdas_inv_tiled = np.tile(lambdas_inv_flat, N_samples).reshape(-1, 1)

    # Step 3: Repeat X_ to align: shape (N_samples * N_MC * n_components, d)
    X_repeat = np.repeat(X_, N_MC * n_components, axis=0)

    # Step 4: Final GP input: shape (N_samples * N_MC * n_components, d + 1)
    X_lambda_all = np.hstack((X_repeat, lambdas_inv_tiled))

    ### --- GP prediction ---
    m_all = gp.predict(X_lambda_all).reshape(N_samples, N_MC, n_components).transpose(1, 2, 0)  # (N_MC, n, N_samples)

    # Weighted mean over MC samples
    weighted_means = np.sum(weights_all[:, :, None] * m_all, axis=1)  # shape: (N_MC, N_samples)
    mu = np.mean(weighted_means, axis=0)  # shape: (N_samples,)

    ### --- Approximate variance (squared weighted kernel sum) ---
    tilde_sigma_sq = np.zeros(N_samples)

    for idx in range(N_samples):
        # For each x
        x = X_[idx:idx+1, :]  # shape: (1, d)
        lambda_j = lambda_[idx]

        # 1. 构造分母输入
        x_lambda_j = np.hstack([x, 1.0 / lambda_j.reshape(-1,1)])  # (1, d+1)
        # 2. 构造分子输入
        # (N_MC * n_components, d+1)
        lambdas_inv_all = 1.0 / lambdas_all.reshape(-1, 1)  
        x_rep = np.tile(x, (N_MC * n_components, 1))
        x_lambdas_all = np.hstack([x_rep, lambdas_inv_all])

        # 3. 计算分母k((x_j, lambda_j), (x_j, lambda_j)) + noise
        k_diag = gp.kernel_(x_lambda_j, x_lambda_j)  # (1,1)
        sigma_epsilon_sq = gp.kernel_.k2.noise_level
        denom = np.sqrt(k_diag[0,0] + sigma_epsilon_sq / n_rep)  # scalar

        # 4. 计算分子k((x_j, lambda_l^{(t,i)}), (x_j, lambda_j)) 对所有 i,l
        x_lambda_j_expanded = np.repeat(x_lambda_j, N_MC * n_components, axis=0)  # (N_MC * n, d+1)
        k_vals = compute_diag_kernel(gp.kernel_, x_lambdas_all, x_lambda_j_expanded)

        # 5. 计算加权核和
        k_vals = k_vals.reshape(N_MC, n_components)
        weighted_k = np.sum(weights_all * k_vals / denom, axis=1)  # (N_MC,)

        tilde_sigma_sq[idx] = (np.mean(weighted_k)) ** 2

    return mu, np.sqrt(tilde_sigma_sq)

def predicted_mc_mean_and_approx_var_single_regime_v2(X_, lambda_point_est, lambdas_mcmc, gp, n_rep):
    """
    Compute posterior mean and approximate variance under single regime (no weights),
    based exactly on the given formulas.

    Parameters:
    - X_: (N_samples, d) array of x
    - lambda_point_est: (N_samples,) array of point estimates (e.g., MAP or posterior mean)
    - lambdas_mcmc: (N_MC,) array of posterior samples of lambda
    - gp: trained GaussianProcessRegressor (on input (x, 1/lambda))
    - n_rep: number of simulation replications at each x

    Returns:
    - mu: (N_samples,) GP posterior mean (MC average)
    - approx_std: (N_samples,) approximate posterior std
    """
    N_samples, d = X_.shape
    N_MC = lambdas_mcmc.shape[0]

    ### Compute posterior mean: mu_n(x)
    lambdas_inv = 1.0 / lambdas_mcmc  # (N_MC,)
    X_repeat = np.repeat(X_, N_MC, axis=0)  # (N_samples * N_MC, d)
    lambdas_tiled = np.tile(lambdas_inv.reshape(-1, 1), (N_samples, 1))  # (N_samples * N_MC, 1)
    X_lambda_all = np.hstack([X_repeat, lambdas_tiled])  # (N_samples * N_MC, d+1)

    # GP predict mean
    m_all = gp.predict(X_lambda_all).reshape(N_MC, N_samples).T  # (N_samples, N_MC)
    mu = np.mean(m_all, axis=1)  # (N_samples,)

    ### Compute approximate variance
    approx_std = np.zeros(N_samples)
    kernel = gp.kernel_
    sigma_eps_sq = gp.kernel_.k2.noise_level  # assumes WhiteKernel as .k2

    for idx in range(N_samples):
        x = X_[idx:idx+1, :]  # (1, d)
        lambda_hat = float(lambda_point_est[idx])
        lambda_hat_inv = 1.0 / lambda_hat

        # Denominator: sqrt(k((x, lambda_hat), (x, lambda_hat)) + sigma^2 / n_rep)
        x_lambda_hat = np.hstack([x, [[lambda_hat_inv]]])  # (1, d+1)
        k_xx = kernel(x_lambda_hat, x_lambda_hat)[0, 0]
        denom = np.sqrt(k_xx + sigma_eps_sq / n_rep)

        # Numerator: average of k((x, lambda^i), (x, lambda_hat))
        lambdas_inv_all = 1.0 / lambdas_mcmc.reshape(-1, 1)  # (N_MC, 1)
        x_rep = np.repeat(x, N_MC, axis=0)  # (N_MC, d)
        x_lambda_i = np.hstack([x_rep, lambdas_inv_all])  # (N_MC, d+1)
        x_lambda_hat_rep = np.repeat(x_lambda_hat, N_MC, axis=0)  # (N_MC, d+1)
        k_vals = compute_diag_kernel(kernel, x_lambda_i, x_lambda_hat_rep)  # (N_MC,)
        approx_std[idx] = (np.mean(k_vals / denom))  # std = sqrt(var), but already squared outside

    return mu, approx_std


def predicted_mean_and_approx_var_given_weights(X_, lambda_, lambdas, weights, gp, n_rep):
    """
    Compute posterior mean and approximate variance of GP at input X_,
    using a single set of lambda and weights (e.g., from interpolation).

    Parameters:
    - X_: (N_samples, d) array of input design points
    - lambda_: (N_samples,) array of corresponding lambda values at each x
    - lambdas: (n_components,)  — interpolated emission parameters
    - weights: (n_components,)  — interpolated weights
    - gp: trained GaussianProcessRegressor
    - n_rep: number of replications

    Returns:
    - mu: (N_samples,) posterior mean at each input x
    - tilde_sigma: (N_samples,) posterior std (approximate)
    """

    N_samples, d = X_.shape
    n_components = len(lambdas)

    # Step 1: Construct (x, 1/lambda_k) pairs for all k
    lambdas_inv = 1.0 / lambdas  # shape: (n_components,)
    lambdas_inv_tiled = np.tile(lambdas_inv, (N_samples, 1)).T  # (n_components, N_samples)
    X_repeat = np.repeat(X_[np.newaxis, :, :], n_components, axis=0)  # (n_components, N_samples, d)
    X_lambdas = np.concatenate([X_repeat, lambdas_inv_tiled[..., np.newaxis]], axis=2)  # (n_components, N_samples, d+1)
    X_lambdas = X_lambdas.reshape(n_components * N_samples, d + 1)

    # Step 2: Predict GP mean at each (x, 1/lambda_k)
    m_all = gp.predict(X_lambdas).reshape(n_components, N_samples)  # (n_components, N_samples)

    # Step 3: Weighted mean over components
    mu = np.dot(weights, m_all)  # (N_samples,)

    # Step 4: Approximate variance
    tilde_sigma = np.zeros(N_samples)
    sigma_epsilon_sq = gp.kernel_.k2.noise_level

    for idx in range(N_samples):
        x = X_[idx:idx+1, :]          # (1, d)
        lambda_j = lambda_[idx]       # scalar
        x_lambda_j = np.hstack([x, [[1.0 / lambda_j]]])  # (1, d+1)

        # Numerator kernel terms: k((x, 1/lambda_k), (x, 1/lambda_j)) for each k
        x_rep = np.repeat(x, n_components, axis=0)
        lambda_k_inv = 1.0 / lambdas.reshape(-1, 1)
        x_lambdas_all = np.hstack([x_rep, lambda_k_inv])
        x_lambda_j_expanded = np.repeat(x_lambda_j, n_components, axis=0)

        k_vals = compute_diag_kernel(gp.kernel_, x_lambdas_all, x_lambda_j_expanded)  # (n_components,)

        # Denominator
        k_diag = gp.kernel_(x_lambda_j, x_lambda_j)[0, 0]
        denom = np.sqrt(k_diag + sigma_epsilon_sq / n_rep)

        weighted_k = np.sum(weights * k_vals / denom)
        tilde_sigma[idx] = weighted_k  # not squared

    return mu, tilde_sigma

#def generate_EGO_log_df_RS_BRO(seed, t, ego_iter, lambda_new, lambdas_all, weights_all):
    #N_MC, n_components = lambdas_all.shape
    #records = []

    #for i in range(N_MC):
        #record = {
            #'seed': seed,
            #'timestep': t,
            #'ego_iter': ego_iter,
            #'EGO_lambda': 1.0 / lambda_new.item(),
            #'n_mc_idx': i,
        #}

        # 添加 lambdas1, lambdas2, ...
        #for j in range(n_components):
            #record[f'lambda{j+1}'] = lambdas_all[i, j]

        # 添加 weights1, weights2, ...
        #for j in range(n_components):
            #record[f'weight{j+1}'] = weights_all[i, j]

        #records.append(record)

    #return pd.DataFrame(records)

def generate_EGO_log_df_RS_BRO(seed, t, ego_iter, x_new, lambda_new, lambdas_all, weights_all):
    """
    构建列顺序为：meta info + x + lambda + all lambdas + all weights
    """
    N_MC, n_components = lambdas_all.shape
    records = []

    x_new = x_new.flatten()  # 假设 x_new 是 shape (1, d)

    for i in range(N_MC):
        record = {
            'seed': seed,
            'timestep': t,
            'ego_iter': ego_iter,
            'EGO_lambda': 1.0 / lambda_new.item(),
            'n_mc_idx': i,
        }

        # 添加 x1, x2, ...
        for j in range(len(x_new)):
            record[f'x{j+1}'] = x_new[j]

        # 添加 lambdas1, lambdas2, ...
        for j in range(n_components):
            record[f'lambda{j+1}'] = lambdas_all[i, j]

        # 添加 weights1, weights2, ...
        for j in range(n_components):
            record[f'weight{j+1}'] = weights_all[i, j]

        records.append(record)

    return pd.DataFrame(records)

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

def update_data_switching_EGO_joint(D_lambda_new, D_old, Y_old, n_rep, method, f, shape_x):

    D_new = D_lambda_new[:shape_x]
    lambda_new = D_lambda_new[shape_x:] ###here is 1/lambda
    distribution_new = {'dist': "exp", "rate": 1/lambda_new}

    Y_new = np.atleast_2d(f.evaluate(D_new, distribution_new, n_rep, method))
    Y_update = np.concatenate((Y_old, Y_new), axis = 0)

    D_new1 = np.atleast_2d(D_lambda_new)
    D_update = np.concatenate((D_old, D_new1), axis=0)

    return D_update, Y_update


def compute_hmm_weights_from_sample_idx(sample_index, obs, n_components, 
                                        posterior_p_list, 
                                        lambdas,
                                        startprob, hmm_class):
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

    Returns:
    - weights: Predicted state weights (1 x n_components).
    """
    model = hmm_class(n_components=n_components, random_state=sample_index)

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

def update_data(x_new, D_old, Y_old, P, n_rep, method, f, **kwargs): ##updated in 12 May, 2025
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
    - **kwargs: Additional optional arguments (e.g., seed) for fixed fitted logkde and ploting

    Returns:
    - D_update: Updated decision variable data
    - Y_update: Updated objective values
    """
    Y_new = np.atleast_2d(f.evaluate(x_new, P, n_rep, method, **kwargs))
    Y_update = np.concatenate((Y_old, Y_new), axis=0)
    D_new = np.atleast_2d(x_new)
    D_update = np.concatenate((D_old, D_new), axis=0)

    return D_update, Y_update

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


def output(file_pre, minimizer,minimum_value, f_hat_x, TIME_RE, x_star, f_star, seed, n_sample, iteration, n_xi, S):

    # Prepare for the output of the results
    minimizer = np.array(minimizer) 
    minimum_value = np.array(minimum_value)
    f_hat_x =  np.array(f_hat_x).reshape(-1)
    x_star = np.array(x_star)
    f_star = np.array(f_star)
    xGAP = np.linalg.norm(minimizer - x_star,axis=1)
    GAP = f_hat_x - f_star
    print(x_star.shape,f_star.shape, 
          minimizer.shape, f_hat_x.shape, 
          xGAP.shape, GAP.shape )

    # Results save
    TIME_RE = np.array(TIME_RE)
    results = pd.DataFrame(minimizer)
    results.columns = ['x_' + str(i) for i in range(minimizer.shape[1])]
    results['f_hat_x'] = f_hat_x
    results['f_star'] = f_star
    results['minimum_value'] = minimum_value
    results['iteration'] = np.arange(iteration)
    results['time'] = TIME_RE
    results['seed'] = seed
    results['n_xi'] = n_xi
    results['GAP'] = GAP
    results['xGAP'] = xGAP
    results['regime_status'] = S

    results_save(file_pre, results)
    #results_plot(f_hat_x, n_sample, iteration)

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

def MAP_exponential(observed): #added single exponential distribution MAP
    """
    使用 PyMC3 进行 MAP 估计，返回 λ (rate parameter) 的估计值对应的随机变量

    参数:
    - observed: 观测数据 (numpy array)

    返回:
    - rv: MAP 估计的 λ 参数值对应的随机变量
    """
    with pm.Model() as model:
        # 选择 Gamma 先验作为 λ 参数的先验分布
        alpha_prior = 1  # Gamma 形状参数
        beta_prior = 1   # Gamma 尺度参数
        lam = pm.Gamma('lam', alpha=alpha_prior, beta=beta_prior)  # 直接定义 λ

        # 定义指数分布的似然，使用 observed 作为数据
        likelihood = pm.Exponential('data', lam=lam, observed=observed)

        # 使用 find_MAP 来找到 MAP 估计
        map_estimate = pm.find_MAP()

    # 返回估计的 λ 参数值对应的随机变量
    lambda_map = map_estimate['lam'].item()
    rv = stats.expon(scale = 1/lambda_map)
    return rv

def MAP_and_generate_exponential(xi, size = 1000): #added single exponential distribution generation
    
    rv = MAP_exponential(xi)
    xi = generate_sample_general(size, rv)

    return xi
