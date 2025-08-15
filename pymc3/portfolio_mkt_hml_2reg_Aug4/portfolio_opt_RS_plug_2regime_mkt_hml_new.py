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

    df0 = pd.read_csv('/root/rs_bro92/RSDR_CVaR_dataset/3_MKT2_Monthly_scaled_mkt_hml.csv')
    # given + upcoming: 200401-200712, 200801-200912 (48+24=72)
    df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20100101)].reset_index(drop=True)
    #df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20080301)].reset_index(drop=True)
    returns = df_used[["Mkt-RF", "HML"]].to_numpy() # 2 selected assets

    # t = 0, given initial observations, get n_initial sets of posterior samples, n_observations = n_xi + (t)
    switching_posterior = bayesian_hmm_posterior_gaussian_2d_uniform_std(mean_sample_count, seed, returns[:n_xi], n_components,mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)

    regime_params_all = {}
    for i in range(1, n_components+1):
        mu1 = switching_posterior.posterior[f"mu1_{i}"].values.flatten()
        sigma1 = switching_posterior.posterior[f"sigma1_{i}"].values.flatten()
        mu2 = switching_posterior.posterior[f"mu2_{i}"].values.flatten()
        sigma2 = switching_posterior.posterior[f"sigma2_{i}"].values.flatten()

        regime_params_all[i] = np.column_stack([mu1, sigma1, mu2, sigma2])
    
    regime_params_all_mean = {} #########
    for i in range(1, n_components + 1):
        regime_params_all_mean[i] = regime_params_all[i].mean(axis=0)  # shape: (4,)
    
    regime_params_all_mean_array = np.vstack([
    regime_params_all_mean[i] for i in range(1, n_components + 1)])
    
    posterior_p_all = {}
    for i in range(1, n_components + 1):
        p = switching_posterior.posterior[f"p_{i}"].values  # shape: (chain, draw, n_components)
        posterior_p_all[i] = p.reshape(-1, p.shape[-1])  # flatten chain and draw
    
    posterior_p_all_mean_normalized = {} ##########

    for i in range(1, n_components + 1):
        p_mean = posterior_p_all[i].mean(axis=0)
        p_mean_normalized = p_mean / p_mean.sum()
        posterior_p_all_mean_normalized[i] = p_mean_normalized  
    
    # weights corresponding to posterior mean
    model = GaussianHMM(n_components=n_components, covariance_type="diag", random_state=seed)
    model.startprob_ = startprob
    model.transmat_ = np.vstack([posterior_p_all_mean_normalized[i] for i in range(1, n_components + 1)])
    means = np.vstack([
    [regime_params_all_mean[i][0], regime_params_all_mean[i][2]]  # mu1, mu2
    for i in range(1, n_components + 1)])
    model.means_ = means
    covars = np.vstack([
    [regime_params_all_mean[i][1]**2, regime_params_all_mean[i][3]**2]  # sigma1^2, sigma2^2
    for i in range(1, n_components + 1)])
    model.covars_ = covars
    predicted_proba = model.predict_proba(returns[:n_xi])[-1:]
    weights = np.matmul(predicted_proba, model.transmat_) 

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    r_tomorrow_list = list()
    TIME_RE = list()
    
    # initial experiment design, get n_initial design points x
    D1 = initial_design_sum1(n_initial, dimension_x, seed = seed)
    Y_list1 = []
    Y_list2 = []
    for sample_idx in range(n_initial):
        xx = D1[sample_idx, :dimension_x]  
        P1 = regime_params_all_mean[1]
        P2 = regime_params_all_mean[2]
        Y_list1.append(f.evaluate(xx, P1, n_rep, seed))
        Y_list2.append(f.evaluate(xx, P2, n_rep, seed))

    Y_update = np.concatenate([Y_list1, Y_list2], axis=0).reshape(-1,1)

    D_update_list = []

    for i in range(1, n_components + 1):
        param = regime_params_all_mean[i].reshape(1, -1)  # shape: (1, dimension_p)
        param_repeated = np.repeat(param, D1.shape[0], axis=0)  # shape: (n_initial, dimension_p)
        D_update_0 = np.concatenate((D1, param_repeated), axis=1)  # shape: (n_initial, dimension_x + dimension_p)
        D_update_list.append(D_update_0)
    
    D_update = np.concatenate(D_update_list, axis=0)

    # here, I get the initial GP model Z(x,lambda), the validated GP model
    gp = GP_model(D_update, Y_update)

    # Get the unique X values (no duplicates)
    X_update = np.unique(D_update[:, :dimension_x], axis=0)
    
    X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)

    mu_evaluated, _ = predicted_mean_std_joint_model_gaussian_params(X_update, regime_params_all_mean_array, weights, gp, n_components)
    mu_g, sigma_g = predicted_mean_std_joint_model_gaussian_params(X_, regime_params_all_mean_array, weights, gp, n_components)

    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()

        for ego_iter in range(n_iteration):
            print("iteration", ego_iter)
            #1. Algorithm for x 
            D_new = EGO(X_, mu_evaluated, mu_g, sigma_g)

            u = np.random.uniform(0, 1)

            # Compute cumulative probabilities for selecting each state
            cumulative_probs = np.cumsum(weights) 

            # Determine which lambda to evaluate, find first index where u ≤ cumulative_prob
            selected_state = np.searchsorted(cumulative_probs, u)

            # record related information to csv file
            record = {
                        'seed': seed,
                        'timestep': t,
                        'ego_iter': ego_iter}
            
            for idx, val in enumerate(weights.flatten()):
                record[f'weights_{idx}'] = val
            
            for state_idx, lam in regime_params_all_mean.items():
                for dim_idx, param in enumerate(lam):
                    record[f'posterior_random_lam_{state_idx}_dim{dim_idx+1}'] = param

            for dim_idx, param in enumerate(regime_params_all_mean[selected_state + 1]):
                record[f'selected_lambda_dim{dim_idx+1}'] = param

            df = pd.DataFrame([record])
            df.to_csv('log_portfo_2regime_48data_RS_plug_mkt_hml.csv', mode='a', header=not os.path.exists('log_portfo_2regime_48data_RS_plug_mkt_hml.csv'), index=False)

            D_update, Y_update = update_data_switching_joint_general_gaussian_portfolio(D_new, D_update, Y_update, regime_params_all_mean, n_rep, f, selected_state+1)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0) 

            # update GP model and make prediction
            gp = GP_model(D_update, Y_update) 

            X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)

            mu_evaluated, _ = predicted_mean_std_joint_model_gaussian_params(X_update, regime_params_all_mean_array, weights, gp, n_components)
            mu_g, sigma_g = predicted_mean_std_joint_model_gaussian_params(X_, regime_params_all_mean_array, weights, gp, n_components)

        # 4. Update x^* (tomorrow best portfolio weight)
        hat_x = X_update[np.argmin(mu_evaluated)]
        returns_tomorrow = returns[n_xi+t]
        portfolio_return_tomorrow = np.dot(hat_x, returns_tomorrow)
        minimizer_list.append(hat_x)
        minimum_value_list.append(min(mu_evaluated))
        r_tomorrow_list.append(portfolio_return_tomorrow)

        # input data updated, generate new MCMC sample (N_MC*(1+n_iteration))
        switching_posterior = bayesian_hmm_posterior_gaussian_2d_uniform_std(mean_sample_count, seed, returns[:n_xi+t+1], n_components,mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)
        
        regime_params_all = {}
        for i in range(1, n_components+1):
            mu1 = switching_posterior.posterior[f"mu1_{i}"].values.flatten()
            sigma1 = switching_posterior.posterior[f"sigma1_{i}"].values.flatten()
            mu2 = switching_posterior.posterior[f"mu2_{i}"].values.flatten()
            sigma2 = switching_posterior.posterior[f"sigma2_{i}"].values.flatten()

            regime_params_all[i] = np.column_stack([mu1, sigma1, mu2, sigma2])
        
        regime_params_all_mean = {} #########
        for i in range(1, n_components + 1):
            regime_params_all_mean[i] = regime_params_all[i].mean(axis=0)  # shape: (4,)
        
        regime_params_all_mean_array = np.vstack([regime_params_all_mean[i] for i in range(1, n_components + 1)])
        
        posterior_p_all = {}
        for i in range(1, n_components + 1):
            p = switching_posterior.posterior[f"p_{i}"].values  # shape: (chain, draw, n_components)
            posterior_p_all[i] = p.reshape(-1, p.shape[-1])  # flatten chain and draw
        
        posterior_p_all_mean_normalized = {} ##########

        for i in range(1, n_components + 1):
            p_mean = posterior_p_all[i].mean(axis=0)
            p_mean_normalized = p_mean / p_mean.sum()
            posterior_p_all_mean_normalized[i] = p_mean_normalized
        
        # weights corresponding to posterior mean
        model = GaussianHMM(n_components=n_components, covariance_type="diag", random_state=seed)
        model.startprob_ = startprob
        model.transmat_ = np.vstack([posterior_p_all_mean_normalized[i] for i in range(1, n_components + 1)])
        means = np.vstack([
        [regime_params_all_mean[i][0], regime_params_all_mean[i][2]]  # mu1, mu2
        for i in range(1, n_components + 1)])
        model.means_ = means
        covars = np.vstack([
        [regime_params_all_mean[i][1]**2, regime_params_all_mean[i][3]**2]  # sigma1^2, sigma2^2
        for i in range(1, n_components + 1)])
        model.covars_ = covars
        predicted_proba = model.predict_proba(returns[:n_xi+t+1])[-1:]
        weights = np.matmul(predicted_proba, model.transmat_) 

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
parser.add_argument('-mean_sample_count','--mean_sample_count', type=int, help = 'number of posterior samples to calculate posterior mean (follow from BRO: N_mc*ego_steps)', default = 3000)

args = parser.parse_args()
cmd = ['-t','p0102-RS-plug-portfolio-2regime-mkt-hml-new-48data-seed40-44', 
       '-method', 'switching_joint',
       '-problem', 'portfolio', 
       '-smacro', '40',###40
       '-macro','45',###45 
       '-n_xi', '48',###48 (change)
       '-n_timestep', '24',###24
       '-n_initial', '20', ###20
       '-n_i', '30',###30
       '-n_rep', '1000',###1000 (more) 
       '-n_candidate','100',###100
       '-window', 'all',
       '-n_MCMC', '100',##100
       '-n_components', '2',###2 
       '-mu_prior_lower', '0.0', ###0.0 (change)
       '-mu_prior_upper', '20.0', ###20.0 (change)
       '-sigma_prior_lower', '0.1', ###0.1 
       '-sigma_prior_upper', '10.0', ###10.0
       '-mean_sample_count', '1500' ###1500
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
mean_sample_count = args.mean_sample_count 

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

