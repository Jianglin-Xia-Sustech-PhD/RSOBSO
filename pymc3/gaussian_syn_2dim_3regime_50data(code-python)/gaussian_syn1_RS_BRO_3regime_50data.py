### 4-regime synthetic test problem1, gaussian \xi, initial 100 input data
### consider regime switching
### BRO framework
### no need to make it in the increasing order
### EGO lambda: average weight to decide regime, random sampling to choose which parameter in the chosen regime 

import time
from switching_code.switchingBO_edit_obj_updated_newest_gaussian_2dim import * #####newest edited on Jun13
pd.options.display.float_format = '{:.2f}'.format

def SwitchingBRO_streaming(seed):

    print(method)
    print('number of states', n_components)
    startprob = np.ones(n_components) / n_components # initial probability updated
    print('start_prob', startprob)

    # real world data generation (fix the regime, get random xi data)
    _, S = switching_real_world_data(switching_true_dist, n_xi + n_timestep, random_state = 40) #fixed state
    rng = np.random.default_rng(seed=seed)
    xi = np.empty(len(S))
    mean_map = {0: 2, 1: 4, 2: 10}
    for i, s in enumerate(S):
        xi[i] = stats.norm(loc=mean_map[s], scale=known_sigma).rvs(random_state=rng)

    # t = 0, given initial observations, get n_initial sets of posterior samples, n_observations = n_xi + (t)
    initial_posterior_0 = bayesian_hmm_posterior_gaussian_known_var_uniform(n_initial, seed, xi[:n_xi], n_components, mu_prior_lower, mu_prior_upper, known_sigma, startprob=startprob) #uniform prior

    # take n_components lam_i (no need to consider the order)
    initial_lam_list = [np.array(initial_posterior_0.posterior[f"mu_{i+1}"]) for i in range(n_components)]

    # initial experiment design, get n_initial design points x
    D1 = inital_design(n_initial, seed, lower_bound, upper_bound)

    # concatenate lambda (no reciporal)
    D_update_list = [np.concatenate((D1, (lam.T)), axis = 1) for lam in initial_lam_list]

    # combine data from all states
    D_update = np.concatenate(D_update_list, axis = 0)
    X_update = np.unique(D_update[:, :dimension_x], axis = 0)

    Y_list_all = [list() for _ in range(n_components)]

    for sample_idx in range(n_initial):
        switching_distributions_estimated = [{'dist': "gaussian", "sigmas": known_sigma, "means": initial_lam_list[state][0][sample_idx]} for state in range(n_components)]
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

    X_ = inital_design(n_candidate, None, lower_bound, upper_bound)

    # generate MCMC sample (N_MC*(1+n_iteration)), no random sampling, use posterior sample mean
    sample_count = n_MCMC * (1 + n_iteration)
    seed_mcmc = seed + 1 #seed for mcmc sampling, avoid sampling repetition
    MCMC_posterior_0 = bayesian_hmm_posterior_gaussian_known_var_uniform(sample_count, seed_mcmc, xi[:n_xi], n_components, mu_prior_lower, mu_prior_upper, known_sigma, startprob=startprob) #uniform prior

    MCMC_posterior_lam_list = [np.array(MCMC_posterior_0.posterior[f"mu_{state+1}"]) for state in range(n_components)]
    MCMC_posterior_p_list = [np.array(MCMC_posterior_0.posterior[f"p_{state+1}"]) for state in range(n_components)]
    
    # take the first N_MC for MCMC mean
    mu_evaluated_list = list()
    mu_g_list = list()
    sigma_g_list = list()
    time_weight_list = [[] for _ in range(n_components)]
    posterior_lam_list = [[] for _ in range(n_components)] # to save all the posterior samples in this list

    for mcmc_idx in range(n_MCMC):
        sample_idx = mcmc_idx
        lambdas = np.array([[MCMC_posterior_lam_list[state][0][sample_idx]] for state in range(n_components)])

        weights = compute_hmm_weights_from_sample_idx_gaussian(sample_idx, xi[:n_xi], n_components, MCMC_posterior_p_list, lambdas, startprob, known_var=known_sigma**2)

        mu_evaluated, _ = predicted_mean_std_joint_model_general_lambda_direct(X_update, lambdas, weights, gp, n_components) #no reciprocal
        mu_g, sigma_g = predicted_mean_std_joint_model_general_lambda_direct(X_, lambdas, weights, gp, n_components)

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
 
    # normalize the average weights
    time_avg_weight_list = [mean_average(time_weight_list[state]) for state in range(n_components)]
    time_avg_total_weight = sum(time_avg_weight_list)
    time_prob_weight_list = [time_avg_weight_list[state] / time_avg_total_weight for state in range(n_components)]

    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()

        for ego_iter in range(n_iteration):
            print("iteration", ego_iter)

            #1. Algorithm for x 
            D_new = EGO(X_, MCMC_mean_mu_evaluated, MCMC_mean_mu_g, MCMC_mean_sigma_g)

            # 2. Algorithm for lambda (posterior random sampling)
            switching_distributions_estimated_updated = []

            for state_idx in range(n_components):
                distribution_updated = {'dist': "gaussian", "sigmas": known_sigma, "means": posterior_random_lam_list[state_idx]}  # posterior random sampling

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
                    }

            for idx, val in enumerate(time_prob_weight_list):
                record[f'time_prob_weight_{idx}'] = val

            for idx, val in enumerate(posterior_random_lam_list):
                record[f'posterior_random_lam_{idx}'] = val

            record['selected_lambda'] = posterior_random_lam_list[selected_state]

            df = pd.DataFrame([record])
            df.to_csv('log_gaussian_syn2_3regime_50data_RS_BRO_random.csv', mode='a', header=not os.path.exists('log_gaussian_syn2_3regime_50data_RS_BRO_random.csv'), index=False)

            D_update, Y_update = update_data_switching_joint_general_gaussian(D_new, D_update, Y_update, switching_distributions_estimated_updated, n_rep, method, f, selected_state)

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
                weights = compute_hmm_weights_from_sample_idx_gaussian(sample_idx, xi[:n_xi + t], n_components, MCMC_posterior_p_list, lambdas, startprob, known_var=known_sigma**2)

                mu_evaluated, _ = predicted_mean_std_joint_model_general_lambda_direct(X_update, lambdas, weights, gp, n_components) #no reciprocal
                mu_g, sigma_g = predicted_mean_std_joint_model_general_lambda_direct(X_, lambdas, weights, gp, n_components)

                mu_evaluated_list.append(mu_evaluated)
                mu_g_list.append(mu_g)
                sigma_g_list.append(sigma_g)

                for state_idx in range(n_components):
                    time_weight_list[state_idx].append(weights[0][state_idx]) #added
                    posterior_lam_list[state_idx].append(MCMC_posterior_lam_list[state_idx][0][sample_idx]) 

            MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list)
            MCMC_mean_mu_g = mean_average(mu_g_list)
            MCMC_mean_sigma_g = sigma_average(sigma_g_list)

            #instead of computing average for each state, we directly sample randomly
            posterior_random_lam_list = [np.random.choice(posterior_lam_list[state]) for state in range(n_components)]
        
            # update and normalize weights
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

        X_ = inital_design(n_candidate, None, lower_bound, upper_bound)

        # input data updated, generate new MCMC sample (N_MC*(1+n_iteration))
        MCMC_posterior_0 = bayesian_hmm_posterior_gaussian_known_var_uniform(sample_count, seed_mcmc+t+1, xi[:n_xi+t+1], n_components, mu_prior_lower, mu_prior_upper, known_sigma, startprob=startprob)

        MCMC_posterior_lam_list = [np.array(MCMC_posterior_0.posterior[f"mu_{state+1}"]) for state in range(n_components)]
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
            weights = compute_hmm_weights_from_sample_idx_gaussian(sample_idx, xi[:n_xi+t+1], n_components, MCMC_posterior_p_list, lambdas, startprob, known_var=known_sigma**2)

            mu_evaluated, _ = predicted_mean_std_joint_model_general_lambda_direct(X_update, lambdas, weights, gp, n_components) #no reciprocal
            mu_g, sigma_g = predicted_mean_std_joint_model_general_lambda_direct(X_, lambdas, weights, gp, n_components)

            mu_evaluated_list.append(mu_evaluated)
            mu_g_list.append(mu_g)
            sigma_g_list.append(sigma_g)

            for state_idx in range(n_components):
                time_weight_list[state_idx].append(weights[0][state_idx])
                posterior_lam_list[state_idx].append(MCMC_posterior_lam_list[state_idx][0][sample_idx])

        MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list) 
        MCMC_mean_mu_g = mean_average(mu_g_list) 
        MCMC_mean_sigma_g = mean_average(sigma_g_list)

        # random sampling for each state
        posterior_random_lam_list = [np.random.choice(posterior_lam_list[state]) for state in range(n_components)]

        # update and normalize average weights
        time_avg_weight_list = [mean_average(time_weight_list[state]) for state in range(n_components)]
        time_avg_total_weight = sum(time_avg_weight_list)
        time_prob_weight_list = [time_avg_weight_list[state] / time_avg_total_weight for state in range(n_components)]

        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)

    output_Bayes(file_pre, minimizer_list, minimum_value_list, f_hat_x_list, TIME_RE, x_star_list, f_star_list, seed, n_initial, n_timestep, n_xi, S[n_xi:])

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
parser.add_argument('-known_sigma', '--known_sigma', type=float, help='Known standard deviation (sigma)', default=4.0)
parser.add_argument('-mu_prior_lower', '--mu_prior_lower', type=float, help='Uniform prior lower of mu', default=5.0)
parser.add_argument('-mu_prior_upper', '--mu_prior_upper', type=float, help='Uniform prior upper of mu', default=5.0)

args = parser.parse_args()
cmd = ['-t','20260601-RS-BRO-gaus-syn2-3regime-50data-seed-60-69', 
       '-method', 'switching_joint',
       '-problem', 'synthetic2', 
       '-smacro', '60',###40
       '-macro','70',###70 
       '-n_xi', '50',###50 (change)
       '-n_timestep', '25',###25
       '-n_initial', '20', ###20
       '-n_i', '30',###30
       '-n_rep', '100',###100 (more) 
       '-n_candidate','100',###100
       '-window', 'all',
       '-n_MCMC', '100',##100
       '-n_components', '3',###3 (change)
       '-known_sigma', '3.0', ###3.0(change)
       '-mu_prior_lower', '0.0', ###0.0 (change)
       '-mu_prior_upper', '50.0' ###50.0 (change)
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
known_sigma = args.known_sigma # known stdev sigma of gaussian distributions
mu_prior_lower = args.mu_prior_lower # prior distribution mean for mu
mu_prior_upper = args.mu_prior_upper # prior distribution stdev for mu

switching_true_dist = {
    'startprob': np.array([0, 0, 1]), 
    "transmat": np.array([
    [0.7, 0.15, 0.15],
    [0.15, 0.7, 0.15],
    [0.10, 0.10, 0.8]
    ]),
    "means": np.array([[2], [4], [10]]), 
    "covars": np.full((n_components, 1), known_sigma**2), #known and constant variance
    "emission": "gaussian",
    "n_components": 3
    }

switching_distributions = []
for s in range(n_components):
    switching_distributions.append({
        'dist': switching_true_dist.get("emission"),
        'sigmas': known_sigma, # constant, given
        "means": switching_true_dist.get("means")[s]
    })

f1 = synthetic_quad_problem(switching_distributions[0]) 
f2 = synthetic_quad_problem(switching_distributions[1]) 
f3 = synthetic_quad_problem(switching_distributions[2]) 

# simple function, directly give optimal solutions and values
f1.x_star = np.array([6, 12])
f2.x_star = np.array([2, 4]) 
f3.x_star = np.array([-10, -20])

f1.f_star = (6 - 10)**2 + (12 - 20)**2 + 2 * (4*6 + 8*12)
f2.f_star = (2 - 10)**2 + (4 - 20)**2 + 4 * (4*2 + 8*4)
f3.f_star = (-10 - 10)**2 + (-20 - 20)**2 + 10 * (4*(-10) + 8*(-20))

f_true = [f1,f2,f3]
f = f1
true_minimizer = [f1.x_star, f2.x_star, f3.x_star]
true_minimium = [f1.f_star, f2.f_star, f3.f_star]
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