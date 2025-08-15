### 4-regime synthetic test problem1, gaussian \xi, initial 100 input data
# no consider regime switching
### calculate conjugate posterior to complete BRO, no need to use pymc3 to get posterior sampling
### EGO lambda: random sampling
import time
from switching_code.switchingBO_edit_obj_updated_newest_gaussian_2dim import * #####newest edited on Jun13
pd.options.display.float_format = '{:.2f}'.format

def gaussian_BRO_streaming(seed):

    print(method) #method='switching_joint' 
    print('number of states', n_components)

    # real world data generation (fix the regime, get random xi data)
    _, S = switching_real_world_data(switching_true_dist, n_xi + n_timestep, random_state = 40) #fixed state
    rng = np.random.default_rng(seed=seed)
    xi = np.empty(len(S))
    mean_map = {0: 2, 1: 4, 2: 10}
    for i, s in enumerate(S):
        xi[i] = stats.norm(loc=mean_map[s], scale=known_sigma).rvs(random_state=rng)

    # t = 0, given initial observations, get n_initial sets of posterior samples, n_observations = n_xi + (t)
    initial_posterior_0 = sample_posterior_mu_uniform_prior(xi[:n_xi], mu_prior_lower, mu_prior_upper, known_sigma, n_components*n_initial, seed)

    initial_lam = initial_posterior_0.reshape(-1, 1)

    # initial experiment design, get n_components*n_initial design points x
    D1 = inital_design(n_components*n_initial, seed, lower_bound, upper_bound)

    # actually joint modeling x and lambda (no reciprocal)
    D_update = np.concatenate((D1, initial_lam), axis=1) 
    X_update = np.unique(D_update[:,:dimension_x], axis = 0)#pure BRO, still GP joint modeling

    Y_list = list()

    # give an iteration to get observed values
    for sample_idx in range(D1.shape[0]):
        distribution1 = {'dist': "gaussian", "sigmas": known_sigma, "means": initial_lam[sample_idx][0]}
        xx = D1[sample_idx,:dimension_x] 
        P1 = distribution1
        Y_list.append(f.evaluate(xx, P1, n_rep, method))
    
    Y1 = np.array(Y_list).reshape(-1,1)
    Y_update = Y1

    # here, I get the initial GP model Z_n(x,lambda), the validated GP joint model
    gp = GP_model(D_update, Y_update)

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    f_hat_x_list = list()
    x_star_list = list()
    f_star_list = list()
    TIME_RE = list()

    X_ = inital_design(n_candidate, None, lower_bound, upper_bound)

    # generate MCMC sample (N_MC*(1+n_iteration))
    sample_count = n_MCMC * (1 + n_iteration)

    MCMC_posterior_0 = sample_posterior_mu_uniform_prior(xi[:n_xi], mu_prior_lower, mu_prior_upper, known_sigma, sample_count, seed)

    MCMC_posterior_lam = MCMC_posterior_0.reshape(-1, 1) 

    # take the first N_MC for MCMC mean
    mu_evaluated_list = list()
    mu_g_list = list()
    sigma_g_list = list()
    posterior_lam_list = list()

    for mcmc_idx in range(n_MCMC):
        # no reciprocal
        mu_evaluated, _ = predicted_mean_std_single_lambda_direct(X_update, MCMC_posterior_lam[mcmc_idx][0], gp)
        mu_g, sigma_g = predicted_mean_std_single_lambda_direct(X_, MCMC_posterior_lam[mcmc_idx][0], gp)

        mu_evaluated_list.append(mu_evaluated)
        mu_g_list.append(mu_g)
        sigma_g_list.append(sigma_g)
        posterior_lam_list.append(MCMC_posterior_lam[mcmc_idx][0])
    
    MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list)
    MCMC_mean_mu_g = mean_average(mu_g_list)
    MCMC_mean_sigma_g = sigma_average(sigma_g_list)

    posterior_random_lam = np.random.choice(posterior_lam_list) #random sampling mean for lambda1

    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()

        for ego_iter in range(n_iteration):
            print("iteration", ego_iter)
            print(f'posterior random lambda at {ego_iter}th EGO iteration: {posterior_random_lam}')

            # record related information to csv file
            record = {
                        'seed': seed,
                        'timestep': t,
                        'ego_iter': ego_iter,
                        'posterior_random_lam': posterior_random_lam
                    }
            
            df = pd.DataFrame([record])
            df.to_csv('log_gaussian_syn2_3regime_50data_no_RS_BRO_random.csv', mode='a', header=not os.path.exists('log_gaussian_syn2_3regime_50data_no_RS_BRO_random.csv'), index=False)

            #1. Algorithm for x 
            D_new = EGO(X_, MCMC_mean_mu_evaluated, MCMC_mean_mu_g, MCMC_mean_sigma_g)

            # 2. Algorithm for lambda (random sampling of 100 samples)
            distribution1_updated = {'dist': "gaussian", "sigmas": known_sigma, "means": posterior_random_lam} 
            
            # no reciprocal
            D_update, Y_update = update_data_switching_joint_single_gaussian(D_new, D_update, Y_update, distribution1_updated, n_rep, method, f)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0) 

            # update GP model and make prediction
            gp = GP_model(D_update, Y_update)

            X_ = inital_design(n_candidate, None, lower_bound, upper_bound)

            mu_evaluated_list = list()
            mu_g_list = list()
            sigma_g_list = list()
            posterior_lam_list = list()

            # generate N_MCMC samples to update the GP model Z_n
            for mcmc_idx in range(n_MCMC): 
                sample_idx = (ego_iter + 1) * n_MCMC + mcmc_idx
                mu_evaluated, _ = predicted_mean_std_single_lambda_direct(X_update, MCMC_posterior_lam[sample_idx][0], gp)
                mu_g, sigma_g = predicted_mean_std_single_lambda_direct(X_, MCMC_posterior_lam[sample_idx][0], gp) 

                mu_evaluated_list.append(mu_evaluated)
                mu_g_list.append(mu_g)
                sigma_g_list.append(sigma_g)
                posterior_lam_list.append(MCMC_posterior_lam[sample_idx][0])
            
            MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list)
            MCMC_mean_mu_g = mean_average(mu_g_list) 
            MCMC_mean_sigma_g = sigma_average(sigma_g_list)

            posterior_random_lam = np.random.choice(posterior_lam_list) #random sampling mean for lambda1
        
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
        MCMC_posterior_0 = sample_posterior_mu_uniform_prior(xi[:n_xi+t+1], mu_prior_lower, mu_prior_upper, known_sigma, sample_count, seed)

        MCMC_posterior_lam = MCMC_posterior_0.reshape(-1, 1) 

        # take the first N_MC for MCMC mean
        mu_evaluated_list = list()
        mu_g_list = list()
        sigma_g_list = list()
        posterior_lam_list = list()

        for mcmc_idx in range(n_MCMC):

            mu_evaluated, _ = predicted_mean_std_single_lambda_direct(X_update, MCMC_posterior_lam[mcmc_idx][0], gp)
            mu_g, sigma_g = predicted_mean_std_single_lambda_direct(X_, MCMC_posterior_lam[mcmc_idx][0], gp)

            mu_evaluated_list.append(mu_evaluated)
            mu_g_list.append(mu_g)
            sigma_g_list.append(sigma_g)

            posterior_lam_list.append(MCMC_posterior_lam[mcmc_idx][0])
        
        MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list)
        MCMC_mean_mu_g = mean_average(mu_g_list) 
        MCMC_mean_sigma_g = sigma_average(sigma_g_list)

        posterior_random_lam = np.random.choice(posterior_lam_list) #random sampling mean for lambda1

        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)
    
    output_Bayes(file_pre, minimizer_list, minimum_value_list, f_hat_x_list, TIME_RE, x_star_list, f_star_list, seed, n_initial, n_timestep, n_xi, S[n_xi:])

def Experiment_noRS_BRO_mean_calculate(seed):
    if method == 'switching_joint':
        gaussian_BRO_streaming(seed)

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
parser.add_argument('-known_sigma', '--known_sigma', type=float, help='Known standard deviation (sigma)', default=4.0)
parser.add_argument('-mu_prior_lower', '--mu_prior_lower', type=float, help='Uniform prior lower of mu', default=5.0)
parser.add_argument('-mu_prior_upper', '--mu_prior_upper', type=float, help='Uniform prior upper of mu', default=5.0)

args = parser.parse_args()
cmd = ['-t','20260604-syn2-no_RS_BRO_3regime_50data-seed-60-69', 
       '-method', 'switching_joint',
       '-problem', 'synthetic2', 
       '-smacro', '60',###60
       '-macro','70',###70 
       '-n_xi', '50',###50(note) 
       '-n_timestep', '25',###25
       '-n_initial', '20', ###20
       '-n_i', '30',###30  
       '-n_rep', '100',###100  
       '-n_candidate','100',###100 
       '-window', 'all',
       '-n_MCMC', '100', ###100
       '-n_components', '3', ###3
       '-known_sigma', '3.0', ###3.0
       '-mu_prior_lower', '0.0', ###0.0
       '-mu_prior_upper', '50.0' ###50.0
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
n_components = args.n_components
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
        Experiment_noRS_BRO_mean_calculate(seed)
        print('That took {:.2f} seconds'.format(time.time() - t0))
    
    print('That took {:.2f} seconds'.format(time.time() - starttime))

