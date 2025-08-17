### real-world experiment
### 2-regime inventory problem, exponential demand
### the regime data are from real-world data
### initial 48 month data given
### additional 24 months (time stages)
### no need to make it in the increasing order
### BRO framework
### EGO lambda: average weight for regime choice, random sampling for parameters in each regime
### infer the number of states through HDPHMM
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
from hdphmm.online_hdphmm import *
 
def SwitchingBRO_streaming_unknown(seed):

    print(method)
    N_max = 10 # upper bound of initial number of regimes
    print('N_max', N_max)

    # read real regime data and generate corresponding random xi here
    df0 = pd.read_csv('/root/rs_bro92/RSDR_CVaR_dataset/BB_state.csv')
    df0['state_name'] = df0['state'].map({1: 'bull', 2: 'bear'})
    df0['state'] = df0['state'].replace({1: 0, 2: 1})
    # given + upcoming: 200401-200712, 200801-200912 (48+24=72)
    df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20100101)].reset_index(drop=True)
    #df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20080601)].reset_index(drop=True)
    S = df_used['state'].to_numpy() # real state data

    # random seed setting to generate xi
    rng = np.random.default_rng(seed=seed)
    xi = np.empty(len(S))
    # regime 0: rate 1/20, regime 1: rate 1 
    for i, s in enumerate(S):
        scale = 20 if s == 0 else 1 
        xi[i] = stats.expon(scale=scale).rvs(random_state=rng)
    
    # given initial data, infer n_components instead of given it directly
    state_counts = infer_state_count(xi[:n_xi], N_max, burn_in = 100, total_samples = 100)
    # get max number of states
    S_max = np.max(state_counts)
    # get the inferred number of states
    n_components = infer_n_components(state_counts) ##### key update
    print('initial inferred number of states', n_components)
    
    startprob = np.r_[1, np.zeros(n_components - 1)] # initial probability
    print('initial start_prob', startprob)
    
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
    N_max_list = list()
    S_max_list = list()
    state_infer_list = list() ###add

    X_ = inital_design(n_candidate, None, lower_bound, upper_bound) 

    # generate MCMC sample (N_MC*(1+n_iteration)), no random sampling, use posterior sample mean
    sample_count = n_MCMC * (1 + n_iteration)
    seed_mcmc = seed + 1 #seed for mcmc sampling, avoid sampling repetition
    MCMC_posterior_0 = bayesian_hmm_posterior_general(sample_count, seed_mcmc, xi[:n_xi], n_components, alpha=lam_prior_alpha, beta=lam_prior_beta, startprob=startprob) #add startprob

    MCMC_posterior_lam_list = [np.array(MCMC_posterior_0.posterior[f"lam_{state+1}"]) for state in range(n_components)]
    MCMC_posterior_p_list = [np.array(MCMC_posterior_0.posterior[f"p_{state+1}"]) for state in range(n_components)]

    # take the first N_MC for MCMC mean
    mu_evaluated_list = list()
    mu_g_list = list()
    sigma_g_list = list()
    time_weight_list = [[] for _ in range(n_components)]
    posterior_lam_list = [[] for _ in range(n_components)] 
    
    for mcmc_idx in range(n_MCMC):
        sample_idx = mcmc_idx
        lambdas = np.array([[MCMC_posterior_lam_list[state][0][sample_idx]] for state in range(n_components)])

        weights = compute_hmm_weights_from_sample_idx(sample_idx, xi[:n_xi], n_components, MCMC_posterior_p_list, lambdas, startprob, ExponentialHMM)

        mu_evaluated, _ = predicted_mean_std_joint_model_general(X_update, lambdas, weights, gp, n_components)
        mu_g, sigma_g = predicted_mean_std_joint_model_general(X_, lambdas, weights, gp, n_components) 

        mu_evaluated_list.append(mu_evaluated)
        mu_g_list.append(mu_g)
        sigma_g_list.append(sigma_g)

        for state_idx in range(n_components):
            time_weight_list[state_idx].append(weights[0][state_idx])
            posterior_lam_list[state_idx].append(MCMC_posterior_lam_list[state_idx][0][sample_idx])

    MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list) 
    MCMC_mean_mu_g = mean_average(mu_g_list) 
    MCMC_mean_sigma_g = mean_average(sigma_g_list) 

    # we directly sample randomly
    posterior_random_lam_list = [np.random.choice(posterior_lam_list[state]) for state in range(n_components)] 
    # normalize weights
    time_avg_weight_list = [mean_average(time_weight_list[state]) for state in range(n_components)]
    time_avg_total_weight = sum(time_avg_weight_list)
    time_prob_weight_list = [time_avg_weight_list[state] / time_avg_total_weight for state in range(n_components)]

    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()

        for ego_iter in range(n_iteration):
            print("iteration", ego_iter)

            # 1. Algorithm for x 
            D_new = EGO(X_, MCMC_mean_mu_evaluated, MCMC_mean_mu_g, MCMC_mean_sigma_g)

            # 2. Algorithm for lambda (random sampling)
            switching_distributions_estimated_updated = []

            for state_idx in range(n_components):
                distribution_updated = {'dist': "exp", "rate": posterior_random_lam_list[state_idx]} 

                switching_distributions_estimated_updated.append(distribution_updated)

            # introduce the random number u from U[0, 1) to guide which lambda to evaluate
            u = np.random.uniform(0, 1)

            # Compute cumulative probabilities for selecting each state
            cumulative_probs = np.cumsum(time_prob_weight_list) 

            # Determine which lambda to evaluate, find first index where u ≤ cumulative_prob
            selected_state = np.searchsorted(cumulative_probs, u)  
            # record related information to csv file
            record = {
                        'seed': seed,
                        'timestep': t,
                        'ego_iter': ego_iter,
                        'regime_num': n_components
                    }

            for idx, val in enumerate(time_prob_weight_list):
                record[f'time_prob_weight_{idx}'] = val

            for idx, val in enumerate(posterior_random_lam_list):
                record[f'posterior_random_lam_{idx}'] = val

            record['selected_lambda'] = posterior_random_lam_list[selected_state]

            df = pd.DataFrame([record])
            df.to_csv('log_real_inventory_2regime_48data_unknown_RS_BRO_random.csv', mode='a', header=not os.path.exists('log_real_inventory_2regime_48data_unknown_RS_BRO_random.csv'), index=False)

            D_update, Y_update = update_data_switching_joint_general(D_new, D_update, Y_update, switching_distributions_estimated_updated, n_rep, method, f, selected_state)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0) 

            # update GP model and make prediction
            gp = GP_model(D_update, Y_update) 

            X_ = inital_design(n_candidate, None, lower_bound, upper_bound)

            # need to calculate the weight here again
            mu_evaluated_list = list()
            mu_g_list = list()
            sigma_g_list = list()
            time_weight_list = [[] for _ in range(n_components)]
            posterior_lam_list = [[] for _ in range(n_components)]
    
            # generate N_MCMC samples to update the GP model Z_n
            for mcmc_idx in range(n_MCMC):

                sample_idx = (ego_iter + 1) * n_MCMC + mcmc_idx
                lambdas = np.array([[MCMC_posterior_lam_list[state][0][sample_idx]] for state in range(n_components)])
                weights = compute_hmm_weights_from_sample_idx(sample_idx, xi[:n_xi + t], n_components, MCMC_posterior_p_list, lambdas, startprob, ExponentialHMM)
                
                mu_evaluated, _ = predicted_mean_std_joint_model_general(X_update, lambdas, weights, gp, n_components)
                mu_g, sigma_g = predicted_mean_std_joint_model_general(X_, lambdas, weights, gp, n_components) 

                mu_evaluated_list.append(mu_evaluated)
                mu_g_list.append(mu_g)
                sigma_g_list.append(sigma_g)

                for state_idx in range(n_components):
                    time_weight_list[state_idx].append(weights[0][state_idx]) #added
                    posterior_lam_list[state_idx].append(MCMC_posterior_lam_list[state_idx][0][sample_idx])
            
            MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list)
            MCMC_mean_mu_g = mean_average(mu_g_list)
            MCMC_mean_sigma_g = sigma_average(sigma_g_list)

            # we directly sample randomly
            posterior_random_lam_list = [np.random.choice(posterior_lam_list[state]) for state in range(n_components)]
        
            # normalize average weights
            time_avg_weight_list = [mean_average(time_weight_list[state]) for state in range(n_components)]
            time_avg_total_weight = sum(time_avg_weight_list)
            time_prob_weight_list = [time_avg_weight_list[state] / time_avg_total_weight for state in range(n_components)]

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

        N_max_list.append(N_max)
        S_max_list.append(S_max)
        state_infer_list.append(n_components) 
         
        X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
        # input data updated, then update the number of states here
        N_max = S_max + 1
        # given initial data, infer n_components instead of given it directly
        state_counts = infer_state_count(xi[:n_xi+t+1], N_max, burn_in = 100, total_samples = 100)
        # get max number of states
        S_max = np.max(state_counts)
        # get the inferred number of states
        n_components = infer_n_components(state_counts) ##### key update
        print('inferred number of components', n_components)

        startprob = np.r_[1, np.zeros(n_components - 1)]
        print('updated start probability', startprob)

        # input data updated, generate new MCMC sample (N_MC*(1+n_iteration))
        MCMC_posterior_0 = bayesian_hmm_posterior_general(sample_count, seed_mcmc+t+1, xi[:n_xi+t+1], n_components, alpha=lam_prior_alpha, beta=lam_prior_beta, startprob=startprob) #add startprob

        MCMC_posterior_lam_list = [np.array(MCMC_posterior_0.posterior[f"lam_{state+1}"]) for state in range(n_components)]
        MCMC_posterior_p_list = [np.array(MCMC_posterior_0.posterior[f"p_{state+1}"]) for state in range(n_components)]

        # take the first N_MC for MCMC mean
        mu_evaluated_list = list()
        mu_g_list = list()
        sigma_g_list = list()
        time_weight_list = [[] for _ in range(n_components)]
        posterior_lam_list = [[] for _ in range(n_components)]

        for mcmc_idx in range(n_MCMC):
            sample_idx = mcmc_idx
            lambdas = np.array([[MCMC_posterior_lam_list[state][0][sample_idx]] for state in range(n_components)])
            weights = compute_hmm_weights_from_sample_idx(sample_idx, xi[:n_xi+t+1], n_components, MCMC_posterior_p_list, lambdas, startprob, ExponentialHMM)

            mu_evaluated, _ = predicted_mean_std_joint_model_general(X_update, lambdas, weights, gp, n_components)
            mu_g, sigma_g = predicted_mean_std_joint_model_general(X_, lambdas, weights, gp, n_components)

            mu_evaluated_list.append(mu_evaluated)
            mu_g_list.append(mu_g)
            sigma_g_list.append(sigma_g)

            for state_idx in range(n_components):
                time_weight_list[state_idx].append(weights[0][state_idx])
                posterior_lam_list[state_idx].append(MCMC_posterior_lam_list[state_idx][0][sample_idx]) 

        MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list)
        MCMC_mean_mu_g = mean_average(mu_g_list)
        MCMC_mean_sigma_g = sigma_average(sigma_g_list)
        
        # random sampling for each state
        posterior_random_lam_list = [np.random.choice(posterior_lam_list[state]) for state in range(n_components)]

        # update and normalize weights
        time_avg_weight_list = [mean_average(time_weight_list[state]) for state in range(n_components)]
        time_avg_total_weight = sum(time_avg_weight_list)
        time_prob_weight_list = [time_avg_weight_list[state] / time_avg_total_weight for state in range(n_components)]
        
        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)
                
    output_HDPHMM(file_pre, minimizer_list, minimum_value_list, f_hat_x_list, TIME_RE, x_star_list, f_star_list, seed, n_initial, n_timestep, n_xi, S[n_xi:], N_max_list, S_max_list, state_infer_list)

def Experiment_RSBRO(seed):
    if method == 'switching_joint':
        SwitchingBRO_streaming_unknown(seed)

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
parser.add_argument('-lam_prior_alpha', '--lam_prior_alpha', type=float, help='Prior alpha of lam', default=1.0)
parser.add_argument('-lam_prior_beta', '--lam_prior_beta', type=float, help='Prior beta of lam', default=1.0)

args = parser.parse_args()
cmd = ['-t','20270901-RS-BRO-random-2regime-real-48data-unknown-regime-seed-40-44', 
       '-method', 'switching_joint',
       '-problem', 'inventory', 
       '-smacro', '40',###60
       '-macro','45',###70 
       '-n_xi', '48',###48 (change)
       '-n_timestep', '24',###24 (change)
       '-n_initial', '20', ###20
       '-n_i', '30',###30  
       '-n_rep', '10',###10  
       '-n_candidate','100',###100 
       '-window', 'all',
       '-n_MCMC', '100',##100
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
lam_prior_alpha = args.lam_prior_alpha # prior distribution alpha for lam
lam_prior_beta = args.lam_prior_beta # prior distribution beta for lam

true_lambdas = np.array([[1/20],[1]]) #increasing order
true_n_components = 2 ###real

switching_distributions = []
for s in range(true_n_components):
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


