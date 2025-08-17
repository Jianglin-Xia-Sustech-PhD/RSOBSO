### 4-regime synthetic test problem1, exponential \xi, initial 100 input data
### consider regime switching
### BRO framework
### no need to make it in the increasing order
### EGO lambda: average weight to decide regime, random sampling to choose which parameter in the chosen regime
from switching_code.switchingBO_exp_syn import *  
from switching_code.exponentialhmm import ExponentialHMM
pd.options.display.float_format = '{:.2f}'.format
import time
 
def SwitchingBRO_streaming(seed):

    print(method)
    print('number of states', n_components)
    startprob = np.ones(n_components) / n_components # initial probability updated
    print('start_prob', startprob)

    # real world data generation (fix the regime, get random xi data)
    _, S = switching_real_world_data(switching_true_dist, n_xi + n_timestep, random_state = 24)
    rng = np.random.default_rng(seed=seed)
    xi = np.empty(len(S))
    scale_map = {0: 30, 1: 20, 2: 10, 3: 1}
    
    for i, s in enumerate(S):
        xi[i] = stats.expon(scale=scale_map[s]).rvs(random_state=rng)

    # t = 0, given initial observations, get n_initial sets of posterior samples, n_observations = n_xi + (t)
    initial_posterior_0 = bayesian_hmm_posterior_general(n_initial, seed, xi[:n_xi], n_components, alpha=lam_prior_alpha, beta=lam_prior_beta, startprob=startprob) #add startprob

    # take n_components lam_i 
    initial_lam_list = [np.array(initial_posterior_0.posterior[f"lam_{i+1}"]) for i in range(n_components)]

    # initial experiment design, get n_initial design points x
    D1 = inital_design(n_initial, seed, lower_bound, upper_bound)

    # concatenate 1/lambda 
    D_update_list = [np.concatenate((D1, 1/(lam.T)), axis = 1) for lam in initial_lam_list] ###still inverse

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

    X_generate = inital_design(n_candidate, None, lower_bound, upper_bound) # select n_candidate x first
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
    MCMC_posterior_0 = bayesian_hmm_posterior_general(sample_count, seed_mcmc, xi[:n_xi], n_components, alpha=lam_prior_alpha, beta=lam_prior_beta, startprob=startprob) #add startprob

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

            # 1. Algorithm for x 
            D_lambda_new = EGO_joint(X_lambda, MCMC_mean_mu_evaluated, MCMC_mean_mu_g, MCMC_mean_sigma_g)

            x_new = D_lambda_new[:dimension_x].reshape(1, -1)
            lambda_new = D_lambda_new[dimension_x:].reshape(1, -1) # 取后 p 列，是新的 1/lambda

            log_df = generate_EGO_log_df_RS_BRO(seed, t, ego_iter, x_new, lambda_new, lambdas_all, weights_all)

            # 保存为 CSV 文件
            log_df.to_csv('log_exp_syn_4reg_RS_BRO_change_EI_pos_125.csv', mode='a', header=not os.path.exists('log_exp_syn_4reg_RS_BRO_change_EI_pos_125.csv'), index=False)

            D_update, Y_update = update_data_switching_EGO_joint(D_lambda_new, D_update, Y_update, n_rep, method, f, dimension_x)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0) 

            # update GP model and make prediction
            gp = GP_model(D_update, Y_update) 

            X_generate = inital_design(n_candidate, None, lower_bound, upper_bound) # select n_candidate x

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
cmd = ['-t','80000101-exp-syn2-RS-BRO-EI-joint-pos-4regime-100data-seed-40-49', 
       '-method', 'switching_joint',
       '-problem', 'syn2', 
       '-smacro', '40',###60
       '-macro','50',###70 
       '-n_xi', '100',###100
       '-n_timestep', '25',###25 
       '-n_initial', '20', ###20
       '-n_i', '30',###30 
       '-n_rep', '100',###100  
       '-n_candidate','125',###125(change) 
       '-window', 'all',
       '-n_MCMC', '100',##100
       '-n_components', '4', ###4
       '-lam_prior_alpha', '1.0', ###1.0
       '-lam_prior_beta', '0.1' ###0.1
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

switching_true_dist = {
    'startprob': np.array([0, 0, 0, 1]), 
    "transmat": np.array([
    [0.70, 0.10, 0.10, 0.10],
    [0.10, 0.70, 0.10, 0.10],
    [0.10, 0.10, 0.70, 0.10],
    [0.05, 0.05, 0.10, 0.80]
    ]),
    "lambdas": np.array([[1/30], [1/20], [1/10], [1]]),  
    "emission": "exp",
    "n_components": 4  # 4 regimes
}

switching_distributions = []
for s in range(n_components):
    switching_distributions.append({
        'dist': switching_true_dist.get("emission"),
        "rate": switching_true_dist.get("lambdas")[s]
    })

f1 = synthetic_uni_problem(switching_distributions[0]) 
f2 = synthetic_uni_problem(switching_distributions[1]) 
f3 = synthetic_uni_problem(switching_distributions[2]) 
f4 = synthetic_uni_problem(switching_distributions[3]) 

# simple function, directly give optimal solutions and values
f1.x_star = np.array([30])
f2.x_star = np.array([20]) 
f3.x_star = np.array([10])
f4.x_star = np.array([1]) 

f1.f_star = 1200
f2.f_star = 600
f3.f_star = 200
f4.f_star = 11

f_true = [f1,f2,f3,f4]
f = f1
true_minimizer = [f1.x_star, f2.x_star, f3.x_star, f4.x_star]
true_minimium = [f1.f_star, f2.f_star, f3.f_star, f4.f_star]
print(true_minimizer)
print(true_minimium)

dimension_x = f.dimension_x
lower_bound = f.lb
upper_bound = f.ub
print(lower_bound, upper_bound)

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


