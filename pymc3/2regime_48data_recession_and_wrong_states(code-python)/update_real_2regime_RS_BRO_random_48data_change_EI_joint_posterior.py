### real-world experiment
### 2-regime inventory problem, exponential demand
### the regime data are from real-world data
### initial 48 month data given
### additional 24 months (time stages)
### no need to make it in the increasing order
### BRO framework
### EGO lambda: average weight for regime choice, random sampling for parameters in each regime

#### restart in 7/7/2025, correct the variance of weighted average GP model
#### EI: select the joint (x, \lambda) referred from Pearce and Branke (WSC 2017)

# potential enhancement way
#### try to improve this method: LHS sampling for (x, \lambda) pair
#### increase the number of candidate points (x, \lambda), 100 is not enough 

from hmmlearn import hmm
import numpy as np
import pandas as pd
from pyDOE import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time
pd.options.display.float_format = '{:.2f}'.format
from switching_code.exponentialhmm import ExponentialHMM
from switching_code.switchingBO import *
 
def SwitchingBRO_streaming(seed):

    print(method)
    print('number of states', n_components)
    startprob = np.r_[1, np.zeros(n_components - 1)] # initial probability (1, 0) here

    # read real regime data and generate corresponding random xi here
    df0 = pd.read_csv('/home/admin1/HEBO/MCMC/pymc3-hmm-main/RSDR_CVaR_dataset/BB_state.csv')
    #df0 = pd.read_csv('/root/rs_bro92/RSDR_CVaR_dataset/BB_state.csv')
    df0['state_name'] = df0['state'].map({1: 'bull', 2: 'bear'})
    df0['state'] = df0['state'].replace({1: 0, 2: 1})
    # given + upcoming: 200401-200712, 200801-200912 (48+24=72)
    df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20100101)].reset_index(drop=True)
    #df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20080301)].reset_index(drop=True)
    S = df_used['state'].to_numpy() # real state data

    # random seed setting to generate xi
    rng = np.random.default_rng(seed=seed)
    xi = np.empty(len(S))
    # regime 0: rate 1/20, regime 1: rate 1 
    for i, s in enumerate(S):
        scale = 20 if s == 0 else 1 
        xi[i] = stats.expon(scale=scale).rvs(random_state=rng)

    # t = 0, given initial observations, get n_initial sets of posterior samples, n_observations = n_xi + (t)
    initial_posterior_0 = bayesian_hmm_posterior_general(n_initial, seed, xi[:n_xi], n_components, alpha=lam_prior_alpha, beta=lam_prior_beta, startprob=startprob) #add startprob

    # take n_components lam_i 
    initial_lam_list = [np.array(initial_posterior_0.posterior[f"lam_{i+1}"]) for i in range(n_components)]

    # initial experiment design, get n_initial design points x
    D1 = inital_design(n_initial, seed, lower_bound, upper_bound)

    # concatenate 1/lambda 
    D_update_list = [np.concatenate((D1, 1/(lam.T)), axis = 1) for lam in initial_lam_list]

    # combine data from all states
    D_update = np.concatenate(D_update_list, axis = 0)
    X_update = np.unique(D_update[:, :dimension_x], axis = 0) 

    # initialize a list
    Y_list_all = [list() for _ in range(n_components)]

    for sample_idx in range(n_initial):
        switching_distributions_estimated = [{'dist': "exp", "rate": initial_lam_list[state][0][sample_idx]} for state in range(n_components)]
        xx = D1[sample_idx, :dimension_x]  
        for state1 in range(n_components):
            Pi = switching_distributions_estimated[state1]
            Y_list_all[state1].append(f.evaluate(xx, Pi, n_rep, method))

    Y_all = [np.array(Y_list).reshape(-1, 1) for Y_list in Y_list_all]
    Y_update = np.concatenate(Y_all, axis = 0) 
    # here, I get the initial GP model Z(x,lambda), the validated GP model
    gp = GP_model(D_update, Y_update)

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    f_hat_x_list = list()
    x_star_list = list()
    f_star_list = list()
    TIME_RE = list()

    # change codes here
    X_generate = inital_design(n_candidate, None, lower_bound, upper_bound) # select n_candidate x first
    # I need to generate posterior lambda to combine with x (candidates of lambda)

    X_ = np.repeat(X_generate, n_components, axis=0)

    EI_lam_posterior = bayesian_hmm_posterior_general((n_iteration+1) * n_candidate, None, xi[:n_xi], n_components, alpha=lam_prior_alpha, beta=lam_prior_beta, startprob=startprob)

    EI_lam_posterior_list = [np.array(EI_lam_posterior.posterior[f"lam_{state+1}"]) for state in range(n_components)]

    lam_samples = np.array(EI_lam_posterior_list).squeeze(1) 

    start0 = 0 * n_candidate
    end0 = 1 * n_candidate
    lambda_ = lam_samples.T[start0:end0].reshape(-1, 1)

    lambda_inv = 1.0 / lambda_ ###reciprocal (x, 1/lambda)

    X_lambda = np.hstack((X_, lambda_inv))

    # generate MCMC sample (N_MC*(1+n_iteration)), no random sampling, use posterior sample mean
    sample_count = n_MCMC * (1 + n_iteration)
    seed_mcmc = seed + 1 #seed for mcmc sampling, avoid sampling repetition
    MCMC_posterior_0 = bayesian_hmm_posterior_general(sample_count, seed_mcmc, xi[:n_xi], n_components, alpha=lam_prior_alpha, beta=lam_prior_beta, startprob=startprob) 

    MCMC_posterior_lam_list = [np.array(MCMC_posterior_0.posterior[f"lam_{state+1}"]) for state in range(n_components)]
    MCMC_posterior_p_list = [np.array(MCMC_posterior_0.posterior[f"p_{state+1}"]) for state in range(n_components)]

    weights_all = []
    for sample_idx in range(n_MCMC):
        lambdas = np.array([[MCMC_posterior_lam_list[state][0][sample_idx]] for state in range(n_components)])
        weights = compute_hmm_weights_from_sample_idx(sample_idx, xi[:n_xi], n_components, MCMC_posterior_p_list, lambdas, startprob, ExponentialHMM)
        weights_all.append(weights)
    
    weights_all = np.vstack(weights_all)  # shape: (n_MCMC, n_components)

    # lambdas_all: (n_MCMC, n_components)
    lambdas_all = np.array([[MCMC_posterior_lam_list[state][0][sample_idx] 
                            for state in range(n_components)] 
                            for sample_idx in range(n_MCMC)]) # shape: (n_MCMC, n_components)
    
    MCMC_mean_mu_evaluated = predicted_mc_mean_vectorized(X_update, lambdas_all, weights_all, gp)
    MCMC_mean_mu_g, MCMC_mean_sigma_g = predicted_mc_mean_and_approx_var_vectorized(X_, lambda_, lambdas_all, weights_all, gp, n_rep)

    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()

        for ego_iter in range(n_iteration):
            print("iteration", ego_iter)

            # 1. Algorithm for joint (x, \lambda) 
            D_lambda_new = EGO_joint(X_lambda, MCMC_mean_mu_evaluated, MCMC_mean_mu_g, MCMC_mean_sigma_g)

            #d = X_.shape[1]          # X 的维度
            x_new = D_lambda_new[:dimension_x].reshape(1, -1)
            lambda_new = D_lambda_new[dimension_x:].reshape(1, -1) # 取后 p 列，是新的 1/lambda

            #log_df = generate_EGO_log_df_RS_BRO(seed, t, ego_iter, lambda_new, lambdas_all, weights_all)
            log_df = generate_EGO_log_df_RS_BRO(seed, t, ego_iter, x_new, lambda_new, lambdas_all, weights_all)
            
            # 保存为 CSV 文件
            log_df.to_csv('log_inventory_2reg_RS_BRO_change_EI_pos.csv', mode='a', header=not os.path.exists('log_inventory_2reg_RS_BRO_change_EI_pos.csv'), index=False)

            D_update, Y_update = update_data_switching_EGO_joint(D_lambda_new, D_update, Y_update, n_rep, method, f, dimension_x)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0) 

            # update GP model and make prediction
            gp = GP_model(D_update, Y_update) 

            # change codes here
            X_generate = inital_design(n_candidate, None, lower_bound, upper_bound) # select n_candidate x first
            # I need to generate posterior lambda to combine with x (candidates of lambda)

            X_ = np.repeat(X_generate, n_components, axis=0)

            start0 = (ego_iter + 1) * n_candidate
            end0 = (ego_iter + 2) * n_candidate
            lambda_ = lam_samples.T[start0:end0].reshape(-1, 1)

            lambda_inv = 1.0 / lambda_ ###reciprocal (x, 1/lambda)

            X_lambda = np.hstack((X_, lambda_inv))

            weights_all = []
            for mcmc_idx in range(n_MCMC):

                sample_idx = (ego_iter + 1) * n_MCMC + mcmc_idx
                lambdas = np.array([[MCMC_posterior_lam_list[state][0][sample_idx]] for state in range(n_components)])
                weights = compute_hmm_weights_from_sample_idx(sample_idx, xi[:n_xi+t], n_components, MCMC_posterior_p_list, lambdas, startprob, ExponentialHMM)
                weights_all.append(weights)
            
            weights_all = np.vstack(weights_all)  # shape: (n_MCMC, n_components)
            # lambdas_all: (n_MCMC, n_components)
            lambdas_all = np.array([[MCMC_posterior_lam_list[state][0][(ego_iter + 1) * n_MCMC + mcmc_idx] 
                            for state in range(n_components)] 
                            for mcmc_idx in range(n_MCMC)]) # shape: (n_MCMC, n_components)
            
            MCMC_mean_mu_evaluated = predicted_mc_mean_vectorized(X_update, lambdas_all, weights_all, gp)
            MCMC_mean_mu_g, MCMC_mean_sigma_g = predicted_mc_mean_and_approx_var_vectorized(X_, lambda_, lambdas_all, weights_all, gp, n_rep)
            
        # 4. Update x^* (tomorrow best design point)
        S_i = S[n_xi + t]
        f_star = true_minimium[S_i]
        x_star = true_minimizer[S_i]
        # my prediction x^ for tomorrow 
        hat_x = X_update[np.argmin(MCMC_mean_mu_evaluated)]
        f_hat_x = f_true[S_i].evaluate_true(hat_x)

        minimizer_list.append(hat_x)
        minimum_value_list.append(min(MCMC_mean_mu_evaluated))
        f_hat_x_list.append(f_hat_x)
        x_star_list.append(x_star)
        f_star_list.append(f_star)

        # change codes here
        X_generate = inital_design(n_candidate, None, lower_bound, upper_bound) # select n_candidate x first
        # I need to generate posterior lambda to combine with x (candidates of lambda)

        X_ = np.repeat(X_generate, n_components, axis=0)

        EI_lam_posterior = bayesian_hmm_posterior_general((n_iteration+1) * n_candidate, None, xi[:n_xi+t+1], n_components, alpha=lam_prior_alpha, beta=lam_prior_beta, startprob=startprob)

        EI_lam_posterior_list = [np.array(EI_lam_posterior.posterior[f"lam_{state+1}"]) for state in range(n_components)]

        lam_samples = np.array(EI_lam_posterior_list).squeeze(1) 
        
        # first n_candidate lam samples
        start0 = 0 * n_candidate
        end0 = 1 * n_candidate
        lambda_ = lam_samples.T[start0:end0].reshape(-1, 1)

        lambda_inv = 1.0 / lambda_ ###reciprocal (x, 1/lambda)

        X_lambda = np.hstack((X_, lambda_inv))
        
        # input data updated, generate new MCMC sample (N_MC*(1+n_iteration))
        MCMC_posterior_0 = bayesian_hmm_posterior_general(sample_count, seed_mcmc+t+1, xi[:n_xi+t+1], n_components, alpha=lam_prior_alpha, beta=lam_prior_beta, startprob=startprob)

        MCMC_posterior_lam_list = [np.array(MCMC_posterior_0.posterior[f"lam_{state+1}"]) for state in range(n_components)]
        MCMC_posterior_p_list = [np.array(MCMC_posterior_0.posterior[f"p_{state+1}"]) for state in range(n_components)]

        weights_all = []
        for sample_idx in range(n_MCMC):
            lambdas = np.array([[MCMC_posterior_lam_list[state][0][sample_idx]] for state in range(n_components)])
            weights = compute_hmm_weights_from_sample_idx(sample_idx, xi[:n_xi+t+1], n_components, MCMC_posterior_p_list, lambdas, startprob, ExponentialHMM)
            weights_all.append(weights)
        
        weights_all = np.vstack(weights_all)  # shape: (n_MCMC, n_components)
        # lambdas_all: (n_MCMC, n_components)
        lambdas_all = np.array([[MCMC_posterior_lam_list[state][0][sample_idx] 
                                for state in range(n_components)] 
                                for sample_idx in range(n_MCMC)]) # shape: (n_MCMC, n_components)
        
        MCMC_mean_mu_evaluated = predicted_mc_mean_vectorized(X_update, lambdas_all, weights_all, gp)
        MCMC_mean_mu_g, MCMC_mean_sigma_g = predicted_mc_mean_and_approx_var_vectorized(X_, lambda_, lambdas_all, weights_all, gp, n_rep)
        
        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)
                
    output_Bayes(file_pre, minimizer_list, minimum_value_list, f_hat_x_list, TIME_RE, x_star_list, f_star_list, seed, n_initial, n_timestep, n_xi, S[n_xi:])

def Experiment_RSBRO(seed):
    if method == 'switching_joint':
        SwitchingBRO_streaming(seed)

import argparse
parser = argparse.ArgumentParser(description='SwitchingBRO-algo')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument('-t','--time', type=str, help = 'Date of the experiments, e.g. 20241121',default = '20250320')
parser.add_argument('-method','--method', type=str, help = 'switching/hist/exp/lognorm',default = "switching_joint")
parser.add_argument('-problem','--problem', type=str, help = 'Function',default = 'inventory')
parser.add_argument('-smacro','--start_num',type=int, help = 'Start number of macroreplication', default = 24)
parser.add_argument('-macro','--repet_num', type=int, help = 'Number of macroreplication',default = 50)
parser.add_argument('-n_xi','--n_xi', type=int, help = 'Number of observations of the random variable',default = 100)
parser.add_argument('-n_initial','--n_initial',type=int, help = 'number of initial samples', default = 20)
parser.add_argument('-n_i','--n_iteration', type=int, help = 'Number of iteration',default = 15)
parser.add_argument('-n_timestep','--n_timestep', type=int, help = 'Number of timestep',default = 25)
parser.add_argument('-n_rep','--n_replication', type=int, help = 'Number of replications at each point',default = 2)
parser.add_argument('-n_candidate','--n_candidate', type=int, help = 'Number of candidate points for each iteration',default = 100)
parser.add_argument('-window','--window', type=str, help = 'Number of moving window',default = "all")
parser.add_argument('-n_MCMC','--n_MCMC', type=int, help = 'Number of MCMC samples',default = 100)
parser.add_argument('-n_components','--n_components', type=int, help = 'Number of stages', default = 2)
parser.add_argument('-lam_prior_alpha', '--lam_prior_alpha', type=float, help='Prior alpha of lam', default=1.0)
parser.add_argument('-lam_prior_beta', '--lam_prior_beta', type=float, help='Prior beta of lam', default=1.0)

args = parser.parse_args()
cmd = ['-t','40000101-RS-BRO-random-change-EIjoint-pos-2regime-real-48data-seed-60-69', 
       '-method', 'switching_joint',
       '-problem', 'inventory', 
       '-smacro', '60',###60
       '-macro','70',###70 
       '-n_xi', '48',###48 (change)
       '-n_timestep', '24',###24 (change)
       '-n_initial', '20', ###20
       '-n_i', '30',###30  
       '-n_rep', '10',###10  
       '-n_candidate','250',###250 (250*2=500)
       '-window', 'all',
       '-n_MCMC', '100',##100
       '-n_components', '2', ###2
       '-lam_prior_alpha', '1.0', ###1.0
       '-lam_prior_beta', '1.0' ###1.0
       ]
args = parser.parse_args(cmd)
print(args)

time_run = args.time 
method = args.method 
problem = args.problem 

n_xi = args.n_xi
n_initial = args.n_initial # number of initial samples
n_iteration = args.n_iteration # Iteration
n_timestep = args.n_timestep
n_rep = args.n_replication  #number of replication on each point
n_candidate = args.n_candidate # Each iteration, number of the candidate points are regenerated+
window = args.window
n_MCMC = args.n_MCMC
n_components = args.n_components #number of hidden states
lam_prior_alpha = args.lam_prior_alpha # prior distribution alpha for lam
lam_prior_beta = args.lam_prior_beta # prior distribution beta for lam

true_lambdas = np.array([[1/20],[1]]) #increasing order

switching_distributions = []
for s in range(n_components):
    switching_distributions.append({
        'dist': "exp",
        "rate": true_lambdas[s]
    })

f1 = inventory_problem(switching_distributions[0])
f2 = inventory_problem(switching_distributions[1])

f1.x_star = np.array([63.8,127.0]) #lambda: 1/20
f2.x_star = np.array([1,70]) #lambda: 1

f1.f_star = 147.0
f2.f_star = 38.0

f_true = [f1,f2]
f = f1
true_minimizer = [f1.x_star, f2.x_star]
true_minimium = [f1.f_star, f2.f_star]
print(true_minimizer)
print(true_minimium)

dimension_x = f.dimension_x
lower_bound = [1,70] #### same setting as wsc
upper_bound = [69, 250]

#### added bounds for input parameters
lambda_lower_bounds = [0.01]
lambda_upper_bounds = [2] ######need attention

### combined bounds (added new)
combined_lower_bounds = [1, 70, 0.01]
combined_upper_bounds = [69, 250, 2]

output_dir = "outputs/restart"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if "switching" in method:
    file_pre = '_'.join([output_dir,time_run, problem, method, "RS"])
else:
    file_pre = '_'.join([output_dir,time_run, problem, method, "no_RS"])

print(file_pre)

# single core version
if __name__ == '__main__':

    import logging
    logging.getLogger("theano").setLevel(logging.ERROR)
    logging.getLogger("theano.tensor.blas").setLevel(logging.ERROR)
    logging.getLogger("pymc").setLevel(logging.ERROR)

    start_num = args.start_num
    repet_num = args.repet_num

    starttime = time.time()

    for seed in range(start_num, repet_num):
        t0 = time.time()
        print('current experiment macro-replication', seed)
        Experiment_RSBRO(seed)
        print('That took {:.2f} seconds'.format(time.time() - t0))
    
    print('That took {:.2f} seconds'.format(time.time() - starttime))