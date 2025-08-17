### real-world experiment
### 2-regime inventory problem, exponential demand
### the regime data are from real-world data
### initial 48 month data given
### additional 24 months (time stages)
### no need to make it in the increasing order
### consider regime switching
### plug-in: posterior sample mean
# %%
from hmmlearn import hmm
import numpy as np
import pandas as pd
from pyDOE import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time
import os
pd.options.display.float_format = '{:.2f}'.format
from switching_code.exponentialhmm import ExponentialHMM
from switching_code.switchingBO import *

def SwitchingPlug_streaming(seed):
    # main function of bayesian plug-in posterior mean switching EI algorithm
    print(method) ##'switching_joint' 
    print('number of states', n_components)
    startprob = np.r_[1, np.zeros(n_components - 1)] # initial probability
    
    # read real regime data and generate corresponding random xi here
    df0 = pd.read_csv('/home/admin1/HEBO/MCMC/pymc3-hmm-main/RSDR_CVaR_dataset/BB_state.csv')
    df0['state_name'] = df0['state'].map({1: 'bull', 2: 'bear'})
    df0['state'] = df0['state'].replace({1: 0, 2: 1})
    # given + upcoming: 200401-200712, 200801-200912 (48+24=72)
    df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20100101)].reset_index(drop=True)
    S = df_used['state'].to_numpy() # real state data

    # random seed setting to generate xi
    rng = np.random.default_rng(seed=seed)
    xi = np.empty(len(S))
    # regime 0: rate 1/20, regime 1: rate 1 
    for i, s in enumerate(S):
        scale = 20 if s == 0 else 1 
        xi[i] = stats.expon(scale=scale).rvs(random_state=rng)

    # posterior sampling
    switching_posterior = bayesian_hmm_posterior_general(mean_sample_count, seed, xi[:n_xi], n_components, alpha=lam_prior_alpha, beta=lam_prior_beta)

    switching_posterior_lam_list = [np.array(switching_posterior.posterior[f"lam_{state+1}"]) for state in range(n_components)]
    switching_posterior_p_list = [np.array(switching_posterior.posterior[f"p_{state+1}"]) for state in range(n_components)]

    # calculate posterior mean of lambdas and transition probabilities 
    posterior_mean_lam = np.array([np.mean(switching_posterior_lam_list[state]) for state in range(n_components)]).reshape(n_components, 1)  # shape: (n_components, 1)

    posterior_mean_p = np.array([np.mean(switching_posterior_p_list[state], axis=1).squeeze()  for state in range(n_components)])  # shape: (n_components, n_components)

    # Normalize each row to sum to 1 
    posterior_mean_p = posterior_mean_p / posterior_mean_p.sum(axis=1, keepdims=True)
    
    model = ExponentialHMM(n_components=n_components, random_state=seed, n_iter=30)
    model.startprob_ = startprob  
    model.transmat_ = posterior_mean_p
    model.lambdas_ = posterior_mean_lam
    predicted_proba = model.predict_proba(xi[:n_xi].reshape(-1,1))[-1:]
    weights = np.matmul(predicted_proba, model.transmat_) #weights corresponding to posterior mean  

    swtiching_distributions_estimated = [{'dist': "exp", 'rate': posterior_mean_lam[state]}
    for state in range(n_components)]

    # Prepare for iteration 
    minimizer_list = list() # x-hat
    minimum_value_list = list() # min posterior mean
    f_hat_x_list = list() # z-hat
    x_star_list = list() # x_St_star
    f_star_list = list() # z_St_star
    TIME_RE = list() 

    # initial experiment design for switching algorithm
    D1 = inital_design(n_initial, seed, lower_bound, upper_bound)

    # Initialize list to store Y values for all components
    Y_list_all = []

    # Loop over each component to evaluate the function for each distribution
    for k in range(n_components):
        Y_list = []
        for i in range(D1.shape[0]):
            xx = D1[i, :dimension_x]
            P = swtiching_distributions_estimated[k]
            Y_list.append(f.evaluate(xx, P, n_rep, method))
        Y_array = np.array(Y_list).reshape(-1, 1)
        Y_list_all.append(Y_array)

    # Fit to data using Maximum Likelihood Estimation of the parameters for each component
    D_list_all = []
    for k in range(n_components):
        constant_lambda_k = np.full((D1.shape[0], 1), 1.0 / posterior_mean_lam[k])
        D_lambda_k = np.concatenate((D1, constant_lambda_k), axis=1)
        D_list_all.append(D_lambda_k)

    # Concatenate all D values and Y values
    D_update = np.concatenate(D_list_all, axis=0)
    Y_update = np.concatenate(Y_list_all, axis=0)

    # Get the unique X values (no duplicates)
    X_update = np.unique(D_update[:, :dimension_x], axis=0)

    gp = GP_model(D_update, Y_update)   
    X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
    mu_g, sigma_g = predicted_mean_std_joint_model_general(X_, posterior_mean_lam, weights, gp, n_components)
    mu_evaluated, _ = predicted_mean_std_joint_model_general(X_update, posterior_mean_lam, weights, gp, n_components)

    for t in range(n_timestep):

        print("timestep", t)   
        start = time.time()
        
        for i in range(n_iteration):
            print("iteration",i)

            # 1. Algorithm for x 
            D_new = EGO(X_, mu_evaluated, mu_g, sigma_g)

            # 2. Add selected samples, here we only evaluate x and one lambda
            # introduce the random number u from U[0, 1) to guide which lambda to evaluate
            u = np.random.uniform(0, 1)

            # Compute cumulative probabilities for selecting each state
            cumulative_probs = np.cumsum(weights)

            # Determine which state to evaluate, find the first index where u ≤ cumulative_prob
            selected_state = np.searchsorted(cumulative_probs, u)

            # record related information to csv file
            record = {
                        'seed': seed,
                        'timestep': t,
                        'ego_iter': i,
                    }

            for idx, val in enumerate(weights.flatten()):
                record[f'weights_{idx}'] = val

            for idx, val in enumerate(posterior_mean_lam.flatten()):
                record[f'posterior_mean_lam_{idx}'] = val

            record['selected_lambda'] = posterior_mean_lam[selected_state].item()

            df = pd.DataFrame([record])
            df.to_csv('log_real_inventory_RS_plug_mean.csv', mode='a', header=not os.path.exists('log_real_inventory_RS_plug_mean.csv'), index=False)

            D_update, Y_update = update_data_switching_joint_general(D_new, D_update, Y_update, 
            swtiching_distributions_estimated, n_rep, method, f, selected_state)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0)

            # 3. Update GP model and make prediction 
            gp = GP_model(D_update, Y_update)  
            X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
            mu_g, sigma_g = predicted_mean_std_joint_model_general(X_, posterior_mean_lam, weights, gp, n_components)
            mu_evaluated, _ = predicted_mean_std_joint_model_general(X_update, posterior_mean_lam, weights, gp, n_components)
        
        # 4. Update x^* 
        S_i = S[n_xi + t]
        f_star = true_minimium[S_i]
        x_star = true_minimizer[S_i]

        hat_x = X_update[np.argmin(mu_evaluated)]
        f_hat_x = f_true[S_i].evaluate_true(hat_x)

        # append data into list
        minimizer_list.append(hat_x)
        minimum_value_list.append(min(mu_evaluated))
        f_hat_x_list.append(f_hat_x)
        x_star_list.append(x_star)
        f_star_list.append(f_star)

        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)

        # update data, posterior sampling
        switching_posterior = bayesian_hmm_posterior_general(mean_sample_count, seed, xi[:n_xi+t+1], n_components, alpha=lam_prior_alpha, beta=lam_prior_beta)

        switching_posterior_lam_list = [np.array(switching_posterior.posterior[f"lam_{state+1}"]) for state in range(n_components)]
        switching_posterior_p_list = [np.array(switching_posterior.posterior[f"p_{state+1}"]) for state in range(n_components)]

        # calculate posterior mean of lambdas and transition probabilities 
        posterior_mean_lam = np.array([np.mean(switching_posterior_lam_list[state]) for state in range(n_components)]).reshape(n_components, 1)  # shape: (n_components, 1)

        posterior_mean_p = np.array([np.mean(switching_posterior_p_list[state], axis=1).squeeze()  for state in range(n_components)])  # shape: (n_components, n_components)

        # Normalize each row to sum to 1 
        posterior_mean_p = posterior_mean_p / posterior_mean_p.sum(axis=1, keepdims=True)
        
        model = ExponentialHMM(n_components = n_components, random_state = seed, n_iter = 30)
        model.startprob_ = startprob  
        model.transmat_ = posterior_mean_p
        model.lambdas_ = posterior_mean_lam
        predicted_proba = model.predict_proba(xi[:n_xi+t+1].reshape(-1,1))[-1:]
        weights = np.matmul(predicted_proba, model.transmat_) #weights corresponding to posterior mean  

        swtiching_distributions_estimated = [{'dist': "exp", 'rate': posterior_mean_lam[state]}
        for state in range(n_components)]

    output_Bayes(file_pre, minimizer_list, minimum_value_list, f_hat_x_list, TIME_RE, x_star_list, f_star_list, seed, n_initial, n_timestep, n_xi, S[n_xi:])

# %%
def Experiment_RS_plug(seed):
    if method == 'switching_joint':
        SwitchingPlug_streaming(seed) 

# %%
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
parser.add_argument('-n_components','--n_components', type=int, help = 'Number of stages', default = 2)
parser.add_argument('-lam_prior_alpha', '--lam_prior_alpha', type=float, help='Prior alpha of lam', default=1.0)
parser.add_argument('-lam_prior_beta', '--lam_prior_beta', type=float, help='Prior beta of lam', default=1.0)
parser.add_argument('-mean_sample_count','--mean_sample_count', type=int, help = 'number of posterior samples to calculate posterior mean (follow from BRO: N_mc*ego_steps)', default = 3000)

args = parser.parse_args()
cmd = ['-t','20250511-RS-plug-mean-2regime-real-48data-seed-60-69', 
       '-method', 'switching_joint',
       '-problem', 'inventory', 
       '-smacro', '60',###60
       '-macro','70',###70 
       '-n_xi', '48',###48 (change)
       '-n_timestep', '24',###24
       '-n_initial', '20', ###20
       '-n_i', '30',###30  
       '-n_rep', '10',###10  
       '-n_candidate','100',###100 
       '-window', 'all',
       '-n_components', '2', ###2
       '-lam_prior_alpha', '1.0', ###1.0
       '-lam_prior_beta', '1.0', ###1.0
       '-mean_sample_count', '3000' ###3000
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
n_components = args.n_components #number of hidden states
lam_prior_alpha = args.lam_prior_alpha # prior distribution alpha for lam
lam_prior_beta = args.lam_prior_beta # prior distribution beta for lam
mean_sample_count = args.mean_sample_count # number of posterior samples to calculate posterior mean

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
        Experiment_RS_plug(seed)
        print('That took {:.2f} seconds'.format(time.time() - t0))
    
    print('That took {:.2f} seconds'.format(time.time() - starttime))
