### 4-regime synthetic test problem1, gaussian \xi, initial 100 input data
### no need to make it in the increasing order
### plug-in posterior mean framework

import time
from switching_code.switchingBO_edit_obj_updated_newest_gaussian_2dim import * #####newest edited on Jun13
pd.options.display.float_format = '{:.2f}'.format

def SwitchingPlug_streaming(seed):

    print(method) ##'switching_joint' 
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

    # posterior sampling process
    switching_posterior = bayesian_hmm_posterior_gaussian_known_var_uniform(mean_sample_count, seed, xi[:n_xi], n_components, mu_prior_lower, mu_prior_upper, known_sigma, startprob=startprob)
    
    switching_posterior_lam_list = [np.array(switching_posterior.posterior[f"mu_{state+1}"]) for state in range(n_components)]
    switching_posterior_p_list = [np.array(switching_posterior.posterior[f"p_{state+1}"]) for state in range(n_components)]

    # calculate posterior mean of lambdas and transition probabilities 
    posterior_mean_lam = np.array([np.mean(switching_posterior_lam_list[state]) for state in range(n_components)]).reshape(n_components, 1)  # shape: (n_components, 1)

    posterior_mean_p = np.array([np.mean(switching_posterior_p_list[state], axis=1).squeeze()  for state in range(n_components)])  # shape: (n_components, n_components)

    # Normalize each row to sum to 1 
    posterior_mean_p = posterior_mean_p / posterior_mean_p.sum(axis=1, keepdims=True)

    # weights corresponding to posterior mean 
    model = GaussianHMM(n_components=n_components, covariance_type="diag", random_state=seed)
    model.startprob_ = startprob  
    model.transmat_ = posterior_mean_p
    model.means_ = posterior_mean_lam
    model.covars_ = np.full((n_components, 1), known_sigma**2)  # shape (n_components, 1)
    predicted_proba = model.predict_proba(xi[:n_xi].reshape(-1,1))[-1:]
    weights = np.matmul(predicted_proba, model.transmat_) #weights corresponding to posterior mean 

    swtiching_distributions_estimated = [{'dist': "gaussian", "sigmas": known_sigma, "means": posterior_mean_lam[state]} for state in range(n_components)] 

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
    
    # data linked to posterior mean parameter, get joint (x, \lambda) as design points 
    # no reciprocal 
    D_list_all = []
    for k in range(n_components):
        constant_lambda_k = np.full((D1.shape[0], 1), posterior_mean_lam[k])
        D_lambda_k = np.concatenate((D1, constant_lambda_k), axis=1)
        D_list_all.append(D_lambda_k)
    
    # Concatenate all D values and Y values
    D_update = np.concatenate(D_list_all, axis=0)
    Y_update = np.concatenate(Y_list_all, axis=0)

    # Get the unique X values (no duplicates)
    X_update = np.unique(D_update[:, :dimension_x], axis=0)

    gp = GP_model(D_update, Y_update)   
    X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
    #no reciprocal
    mu_g, sigma_g = predicted_mean_std_joint_model_general_lambda_direct(X_, posterior_mean_lam, weights, gp, n_components)
    mu_evaluated, _ = predicted_mean_std_joint_model_general_lambda_direct(X_update, posterior_mean_lam, weights, gp, n_components) 

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
            df.to_csv('log_gaussian_syn2_3regime_50data_RS_plug_mean.csv', mode='a', header=not os.path.exists('log_gaussian_syn2_3regime_50data_RS_plug_mean.csv'), index=False)

            D_update, Y_update = update_data_switching_joint_general_gaussian(D_new, D_update, Y_update, 
            swtiching_distributions_estimated, n_rep, method, f, selected_state)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0)

            # update GP model and make prediction
            gp = GP_model(D_update, Y_update)
            X_ = inital_design(n_candidate, None, lower_bound, upper_bound)

            # no reciprocal
            mu_g, sigma_g = predicted_mean_std_joint_model_general_lambda_direct(X_, posterior_mean_lam, weights, gp, n_components)
            mu_evaluated, _ = predicted_mean_std_joint_model_general_lambda_direct(X_update, posterior_mean_lam, weights, gp, n_components)
        
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

        # update data, posterior sampling
        switching_posterior=bayesian_hmm_posterior_gaussian_known_var_uniform(mean_sample_count, seed, xi[:n_xi+t+1], n_components, mu_prior_lower, mu_prior_upper, known_sigma, startprob=startprob)

        switching_posterior_lam_list = [np.array(switching_posterior.posterior[f"mu_{state+1}"]) for state in range(n_components)]
        switching_posterior_p_list = [np.array(switching_posterior.posterior[f"p_{state+1}"]) for state in range(n_components)]

        # calculate posterior mean of lambdas and transition probabilities (sorted order)
        posterior_mean_lam = np.array([np.mean(switching_posterior_lam_list[state]) for state in range(n_components)]).reshape(n_components, 1)  # shape: (n_components, 1)

        posterior_mean_p = np.array([np.mean(switching_posterior_p_list[state], axis=1).squeeze()  for state in range(n_components)])  # shape: (n_components, n_components)

        # Normalize each row to sum to 1 
        posterior_mean_p = posterior_mean_p / posterior_mean_p.sum(axis=1, keepdims=True)

        #weights corresponding to posterior mean 
        model = GaussianHMM(n_components=n_components, covariance_type="diag", random_state=seed)
        model.startprob_ = startprob  
        model.transmat_ = posterior_mean_p
        model.means_ = posterior_mean_lam
        model.covars_ = np.full((n_components, 1), known_sigma**2)  # shape (n_components, 1)
        predicted_proba = model.predict_proba(xi[:n_xi+t+1].reshape(-1,1))[-1:]
        weights = np.matmul(predicted_proba, model.transmat_) #weights corresponding to posterior mean 

        swtiching_distributions_estimated = [{'dist': "gaussian", "sigmas": known_sigma, "means": posterior_mean_lam[state]} for state in range(n_components)] 

        # Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)

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
parser.add_argument('-known_sigma', '--known_sigma', type=float, help='Known standard deviation (sigma)', default=4.0)
parser.add_argument('-mu_prior_lower', '--mu_prior_lower', type=float, help='Uniform prior lower of mu', default=5.0)
parser.add_argument('-mu_prior_upper', '--mu_prior_upper', type=float, help='Uniform prior upper of mu', default=5.0)
parser.add_argument('-mean_sample_count','--mean_sample_count', type=int, help = 'number of posterior samples to calculate posterior mean (follow from BRO: N_mc*ego_steps)', default = 3000)

args = parser.parse_args()
cmd = ['-t','20260602-RS-plug-gaus-syn2-3regime-50data-seed-60-69', 
       '-method', 'switching_joint',
       '-problem', 'synthetic4', 
       '-smacro', '60',###60
       '-macro','70',###70 
       '-n_xi', '50',###50 (change)
       '-n_timestep', '25',###25
       '-n_initial', '20', ###20
       '-n_i', '30',###30
       '-n_rep', '100',###100  
       '-n_candidate','100',###100 
       '-window', 'all',
       '-n_components', '3', ###3
       '-known_sigma', '3.0', ###3.0
       '-mu_prior_lower', '0.0', ###0.0
       '-mu_prior_upper', '50.0', ###50.0
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
n_components = args.n_components #number of hidden states
known_sigma = args.known_sigma # known stdev sigma of gaussian distributions
mu_prior_lower = args.mu_prior_lower # prior distribution mean for mu
mu_prior_upper = args.mu_prior_upper # prior distribution stdev for mu
mean_sample_count = args.mean_sample_count # number of posterior samples to calculate posterior mean

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
        Experiment_RS_plug(seed)
        print('That took {:.2f} seconds'.format(time.time() - t0))
    
    print('That took {:.2f} seconds'.format(time.time() - starttime))

