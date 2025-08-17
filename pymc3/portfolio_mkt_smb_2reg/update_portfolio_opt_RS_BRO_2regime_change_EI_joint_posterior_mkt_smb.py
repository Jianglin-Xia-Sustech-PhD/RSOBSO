from hmmlearn import hmm
import numpy as np
from pyDOE import *
import random
import warnings
warnings.filterwarnings("ignore")
import time
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
from switching_code.switchingBO_edit_obj_updated_newest_gaussian_portfolio import *

def SwitchingBRO_streaming(seed):

    print(method)
    print('number of states', n_components)
    startprob = np.ones(n_components) / n_components # initial probability updated
    print('start_prob', startprob)

    df0 = pd.read_csv('/root/rs_bro92/RSDR_CVaR_dataset/3_MKT2_Monthly_scaled.csv')
    # given + upcoming: 200401-200712, 200801-200912 (48+24=72)
    df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20100101)].reset_index(drop=True)
    #df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20080301)].reset_index(drop=True)
    returns = df_used[["Mkt-RF", "SMB"]].to_numpy() # 2 selected assets

    # t = 0, given initial observations, get n_initial sets of posterior samples, n_observations = n_xi + (t)
    initial_posterior_0 = bayesian_hmm_posterior_gaussian_2d_uniform_std(n_initial, seed, returns[:n_xi], n_components,mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)

    regime_params_all = {}

    for i in range(1, n_components+1):
        mu1 = initial_posterior_0.posterior[f"mu1_{i}"].values.flatten()
        sigma1 = initial_posterior_0.posterior[f"sigma1_{i}"].values.flatten()
        mu2 = initial_posterior_0.posterior[f"mu2_{i}"].values.flatten()
        sigma2 = initial_posterior_0.posterior[f"sigma2_{i}"].values.flatten()

        regime_params_all[i] = np.column_stack([mu1, sigma1, mu2, sigma2])
    
    # initial experiment design, get n_initial design points x
    D1 = initial_design_sum1(n_initial, dimension_x, seed = seed)

    D_update_all = {i: np.concatenate((D1, regime_params_all[i]), axis=1)
    for i in regime_params_all}

    D_update = np.concatenate(list(D_update_all.values()), axis=0)
    X_update = np.unique(D_update[:, :dimension_x], axis = 0)

    Y_list1 = []
    Y_list2 = []
    for sample_idx in range(n_initial):
        xx = D1[sample_idx, :dimension_x]  
        P1 = regime_params_all[1][sample_idx]
        P2 = regime_params_all[2][sample_idx]
        Y_list1.append(f.evaluate(xx, P1, n_rep, seed))
        Y_list2.append(f.evaluate(xx, P2, n_rep, seed))

    Y_update = np.concatenate([Y_list1, Y_list2], axis=0).reshape(-1,1)

    # here, I get the initial GP model Z(x,lambda), the validated GP model
    gp = GP_model(D_update, Y_update)

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    r_tomorrow_list = list()
    TIME_RE = list()

    X_generate = initial_design_sum1(n_candidate, dimension_x, seed = None)
    X_ = np.repeat(X_generate, n_components, axis=0)

    EI_posterior_0 = bayesian_hmm_posterior_gaussian_2d_uniform_std((n_iteration+1) * n_candidate, None, returns[:n_xi], n_components, mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)

    regime_params_list = [
    np.column_stack([
        EI_posterior_0.posterior[f"mu1_{i}"].values.flatten(),
        EI_posterior_0.posterior[f"sigma1_{i}"].values.flatten(),
        EI_posterior_0.posterior[f"mu2_{i}"].values.flatten(),
        EI_posterior_0.posterior[f"sigma2_{i}"].values.flatten()
    ])
    for i in range(1, n_components + 1)]

    lambda_samples = np.stack(regime_params_list, axis=1).reshape(-1, 4)

    start0 = 0 * n_candidate * n_components
    end0 = 1 * n_candidate * n_components
    lambda_ = lambda_samples[start0:end0]

    X_lambda = np.hstack((X_, lambda_))

    # generate MCMC sample (N_MC*(1+n_iteration)), no random sampling, use posterior sample mean
    sample_count = n_MCMC * (1 + n_iteration)
    seed_mcmc = seed + 1
    MCMC_posterior_0 = bayesian_hmm_posterior_gaussian_2d_uniform_std(sample_count, seed_mcmc, returns[:n_xi], n_components,mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)

    regime_params_all = {}

    for i in range(1, n_components + 1):
        mu1 = MCMC_posterior_0.posterior[f"mu1_{i}"].values.flatten()
        sigma1 = MCMC_posterior_0.posterior[f"sigma1_{i}"].values.flatten()
        mu2 = MCMC_posterior_0.posterior[f"mu2_{i}"].values.flatten()
        sigma2 = MCMC_posterior_0.posterior[f"sigma2_{i}"].values.flatten()

        regime_params_all[i] = np.column_stack([mu1, sigma1, mu2, sigma2])
    
    posterior_p_all = {}

    for i in range(1, n_components + 1):
        p = MCMC_posterior_0.posterior[f"p_{i}"].values  # shape: (chain, draw, n_components)
        posterior_p_all[i] = p.reshape(-1, p.shape[-1])  # flatten chain and draw
    
    weights_all = []
    for sample_idx in range(n_MCMC):
        weights = compute_hmm_weights_from_sample_idx_2d_gaussian(sample_idx, returns[:n_xi], n_components,
        posterior_p_all, regime_params_all, startprob)
        weights_all.append(weights)
    
    weights_all = np.vstack(weights_all) # shape: (n_MCMC, n_components)

    # lambdas_all: (n_MCMC, n_components, 4)
    lambdas_all = np.array([
    [regime_params_all[i][sample_idx] for i in range(1, n_components + 1)]
    for sample_idx in range(n_MCMC)])  # shape: (n_MCMC, n_components, 4)

    MCMC_mean_mu_evaluated = predicted_mc_mean_vectorized_multidim(X_update, lambdas_all, weights_all, gp)
    MCMC_mean_mu_g, MCMC_mean_sigma_g = predicted_mc_mean_and_approx_var_vectorized_multidim(X_, lambda_, lambdas_all, weights_all, gp, n_rep)

    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()

        for ego_iter in range(n_iteration):
            print("iteration", ego_iter)
            #1. Algorithm for x 
            D_lambda_new = EGO_joint(X_lambda, MCMC_mean_mu_evaluated, MCMC_mean_mu_g, MCMC_mean_sigma_g)

            x_new = D_lambda_new[:dimension_x].reshape(1, -1)
            lambda_new = D_lambda_new[dimension_x:].reshape(1, -1) # 取后 p 列，是新的 lambda

            log_df = generate_EGO_log_df_RS_BRO_multidim(seed, t, ego_iter, x_new, lambda_new, lambdas_all, weights_all)

            log_df.to_csv('log_portfo_2regime_48data_RS_BRO_EIjoint_mkt_smb.csv', mode='a', header=not os.path.exists('log_portfo_2regime_48data_RS_BRO_EIjoint_mkt_smb.csv'), index=False)

            D_update, Y_update = update_data_switching_EGO_joint_multidim(D_lambda_new, D_update, Y_update, n_rep, f, dimension_x)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0) 

            # update GP model and make prediction
            gp = GP_model(D_update, Y_update) 

            X_generate = initial_design_sum1(n_candidate, dimension_x, seed = None)
            X_ = np.repeat(X_generate, n_components, axis=0)

            start0 = (ego_iter + 1) * n_candidate * n_components
            end0 = (ego_iter + 2) * n_candidate * n_components
            lambda_ = lambda_samples[start0:end0]
            X_lambda = np.hstack((X_, lambda_))

            weights_all = []
            for mcmc_idx in range(n_MCMC):

                sample_idx = (ego_iter + 1) * n_MCMC + mcmc_idx
                weights = compute_hmm_weights_from_sample_idx_2d_gaussian(sample_idx, returns[:n_xi+t], n_components,
                posterior_p_all, regime_params_all, startprob)
                weights_all.append(weights)
            
            weights_all = np.vstack(weights_all) # shape: (n_MCMC, n_components)

            # lambdas_all: (n_MCMC, n_components, 4)
            lambdas_all = np.array([
            [regime_params_all[i][(ego_iter + 1) * n_MCMC + mcmc_idx] for i in range(1, n_components + 1)]
            for mcmc_idx in range(n_MCMC)])  # shape: (n_MCMC, n_components, 4)

            MCMC_mean_mu_evaluated = predicted_mc_mean_vectorized_multidim(X_update, lambdas_all, weights_all, gp)
            MCMC_mean_mu_g, MCMC_mean_sigma_g = predicted_mc_mean_and_approx_var_vectorized_multidim(X_, lambda_, lambdas_all, weights_all, gp, n_rep)

        # 4. Update x^* (tomorrow best portfolio weight)
        hat_x = X_update[np.argmin(MCMC_mean_mu_evaluated)]
        returns_tomorrow = returns[n_xi+t]
        portfolio_return_tomorrow = np.dot(hat_x, returns_tomorrow)
        minimizer_list.append(hat_x)
        minimum_value_list.append(min(MCMC_mean_mu_evaluated))
        r_tomorrow_list.append(portfolio_return_tomorrow)

        X_generate = initial_design_sum1(n_candidate, dimension_x, seed = None)
        X_ = np.repeat(X_generate, n_components, axis=0)

        EI_posterior_0 = bayesian_hmm_posterior_gaussian_2d_uniform_std((n_iteration+1) * n_candidate, None, returns[:n_xi+t+1], n_components, mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)

        regime_params_list = [np.column_stack([
        EI_posterior_0.posterior[f"mu1_{i}"].values.flatten(),
        EI_posterior_0.posterior[f"sigma1_{i}"].values.flatten(),
        EI_posterior_0.posterior[f"mu2_{i}"].values.flatten(),
        EI_posterior_0.posterior[f"sigma2_{i}"].values.flatten()])
        for i in range(1, n_components + 1)]

        lambda_samples = np.stack(regime_params_list, axis=1).reshape(-1, 4)

        start0 = 0 * n_candidate * n_components
        end0 = 1 * n_candidate * n_components
        lambda_ = lambda_samples[start0:end0]

        X_lambda = np.hstack((X_, lambda_))

        # input data updated, generate new MCMC sample (N_MC*(1+n_iteration))
        MCMC_posterior_0 = bayesian_hmm_posterior_gaussian_2d_uniform_std(sample_count, seed_mcmc+t+1, returns[:n_xi+t+1], n_components,mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)

        regime_params_all = {}

        for i in range(1, n_components + 1):
            mu1 = MCMC_posterior_0.posterior[f"mu1_{i}"].values.flatten()
            sigma1 = MCMC_posterior_0.posterior[f"sigma1_{i}"].values.flatten()
            mu2 = MCMC_posterior_0.posterior[f"mu2_{i}"].values.flatten()
            sigma2 = MCMC_posterior_0.posterior[f"sigma2_{i}"].values.flatten()

            regime_params_all[i] = np.column_stack([mu1, sigma1, mu2, sigma2])
        
        posterior_p_all = {}

        for i in range(1, n_components + 1):
            p = MCMC_posterior_0.posterior[f"p_{i}"].values  # shape: (chain, draw, n_components)
            posterior_p_all[i] = p.reshape(-1, p.shape[-1])  # flatten chain and draw
        
        weights_all = []
        for sample_idx in range(n_MCMC):
            weights = compute_hmm_weights_from_sample_idx_2d_gaussian(sample_idx, returns[:n_xi+t+1], n_components,
            posterior_p_all, regime_params_all, startprob)
            weights_all.append(weights)
    
        weights_all = np.vstack(weights_all) # shape: (n_MCMC, n_components)

        # lambdas_all: (n_MCMC, n_components, 4)
        lambdas_all = np.array([
        [regime_params_all[i][sample_idx] for i in range(1, n_components + 1)]
        for sample_idx in range(n_MCMC)])  # shape: (n_MCMC, n_components, 4)

        MCMC_mean_mu_evaluated = predicted_mc_mean_vectorized_multidim(X_update, lambdas_all, weights_all, gp)
        MCMC_mean_mu_g, MCMC_mean_sigma_g = predicted_mc_mean_and_approx_var_vectorized_multidim(X_, lambda_, lambdas_all, weights_all, gp, n_rep)

        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)

    output_portfolio(file_pre, minimizer_list, minimum_value_list, r_tomorrow_list, TIME_RE, seed, n_initial, n_timestep, n_xi)

def Experiment_RSBRO(seed):
    if method == 'switching_joint':
        SwitchingBRO_streaming(seed)

import argparse
parser = argparse.ArgumentParser(description='Online-SwitchingBRO-algo')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument('-t','--time', type=str, help = 'Date of the experiments, e.g. 20241121',default = '20250320')
parser.add_argument('-method','--method', type=str, help = 'switching_joint/bayesian_hist/posterior_mean_plug',default = "switching_joint")
parser.add_argument('-problem','--problem', type=str, help = 'Function',default = 'inventory')
parser.add_argument('-smacro','--start_num',type=int, help = 'Start number of macroreplication', default = 40)
parser.add_argument('-macro','--repet_num', type=int, help = 'Number of macroreplication',default = 70)
parser.add_argument('-n_xi','--n_xi', type=int, help = 'Number of observations of the random variable',default = 100)
parser.add_argument('-n_initial','--n_initial',type=int, help = 'number of initial samples', default = 20)
parser.add_argument('-n_i','--n_iteration', type=int, help = 'Number of iteration',default = 15)
parser.add_argument('-n_timestep','--n_timestep', type=int, help = 'Number of timestep',default = 25)
parser.add_argument('-n_rep','--n_replication', type=int, help = 'Number of replications at each point',default = 2)
parser.add_argument('-n_candidate','--n_candidate', type=int, help = 'Number of candidate points for each iteration',default = 100)
parser.add_argument('-window','--window', type=str, help = 'Number of moving window',default = "all")
parser.add_argument('-n_MCMC','--n_MCMC', type=int, help = 'Number of MCMC samples',default = 100)
parser.add_argument('-n_components','--n_components', type=int, help = 'Number of stages', default = 2)
parser.add_argument('-mu_prior_lower', '--mu_prior_lower', type=float, help='Uniform prior lower of mu', default=5.0)
parser.add_argument('-mu_prior_upper', '--mu_prior_upper', type=float, help='Uniform prior upper of mu', default=5.0)
parser.add_argument('-sigma_prior_lower', '--sigma_prior_lower', type=float, help='Uniform prior lower of sigma', default=5.0)
parser.add_argument('-sigma_prior_upper', '--sigma_prior_upper', type=float, help='Uniform prior upper of sigma', default=5.0)

args = parser.parse_args()
cmd = ['-t','p0101-RS-BRO-EI-jo-posterior-portfolio-mkt-smb-2regime-48data-seed40-44', 
       '-method', 'switching_joint',
       '-problem', 'portfolio', 
       '-smacro', '40',###40
       '-macro','45',###45 
       '-n_xi', '48',###48 (change)
       '-n_timestep', '24',###24
       '-n_initial', '20', ###20
       '-n_i', '30',###30
       '-n_rep', '1000',###1000 (more) 
       '-n_candidate','250',###250 (250*2 = 500)
       '-window', 'all',
       '-n_MCMC', '100',##100
       '-n_components', '2',###2 
       '-mu_prior_lower', '0.0', ###0.0 
       '-mu_prior_upper', '50.0', ###50.0
       '-sigma_prior_lower', '0.1', ###0.1 
       '-sigma_prior_upper', '20.0', ###20.0 
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
mu_prior_lower = args.mu_prior_lower # prior distribution mean for mu
mu_prior_upper = args.mu_prior_upper # prior distribution stdev for mu
sigma_prior_lower = args.sigma_prior_lower # prior distribution mean for sigma
sigma_prior_upper = args.sigma_prior_upper # prior distribution stdev for sigma

f = portfolio_problem_2stocks()
dimension_x = 2

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

