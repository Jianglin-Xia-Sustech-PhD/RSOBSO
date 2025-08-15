import time
from switching_code.switchingBO_edit_obj_updated_newest_gaussian_2dim import * #####newest edited on Jun13
pd.options.display.float_format = '{:.2f}'.format

def no_RS_bayesian_kde(seed):

    print(method) #method: 'bayesian_hist'
    print('number of states', n_components)

    # real world data generation (fix the regime, get random xi data)
    _, S = switching_real_world_data(switching_true_dist, n_xi + n_timestep, random_state = 40) #fixed state
    rng = np.random.default_rng(seed=seed)
    xi = np.empty(len(S))
    mean_map = {0: 2, 1: 4, 2: 10}
    for i, s in enumerate(S):
        xi[i] = stats.norm(loc=mean_map[s], scale=known_sigma).rvs(random_state=rng)
    
    if window == "all":
        P = xi[:n_xi].reshape(-1) #all observed data, without moving window
    else:
        P = xi[(n_xi - int(window)):n_xi].reshape(-1)
    
    # initial experiment design
    D1 = inital_design(n_components*n_initial, seed, lower_bound, upper_bound) #n_components*n_initial

    Y_list = list()
    for i in range(D1.shape[0]):
        xx = D1[i,:dimension_x] 
        Y_list.append(f.evaluate(xx, P, n_rep, method, seed = seed)) #fixed, add seed
    
    Y1 = np.array(Y_list).reshape(-1,1)

    gp0 = GP_model(D1, Y1) 

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    f_hat_x_list = list()
    x_star_list = list()
    f_star_list = list()

    TIME_RE = list()
    D_update = D1
    X_update = np.unique(D_update,axis = 0)
    Y_update = Y1

    ## get prediction and obtain minimizer
    X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
    mu_g, sigma_g = gp0.predict(X_, return_std= True) 
    mu_evaluated, _ = gp0.predict(X_update, return_std= True) 

    ## Global Optimization algorithm
    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()
        # save the fitted kde plot in the given package for each timestep
        print(f"Saved the KDE plot for timestep {t}")
        saved_path = f"/home/admin1/HEBO/MCMC/pymc3-hmm-main/kde_plot/gaussian_syn1/seed{seed}_kde_plot_timestep{t}.pdf"
        update_kde_and_plot(P, save_path=saved_path, seed=seed)
       
        for i in range(n_iteration):
            print("iteration", i)
            
            # 1. Algorithm for x 
            D_new = EGO(X_, mu_evaluated, mu_g, sigma_g)

            # 2. Add selected samples
            D_update, Y_update = update_data(D_new, D_update, Y_update, P, n_rep, method, f, seed = seed) #add seed to get the fixed fitted kde 
            X_update = np.unique(D_update, axis = 0) 

            # 3. Update GP model and make prediction 
            gp0 = GP_model(D_update, Y_update)
            X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
            mu_g, sigma_g = gp0.predict(X_, return_std= True) 
            mu_evaluated, _ = gp0.predict(X_update, return_std= True) 
        
        # 4. Update x^*
        if window == "all":
            P = xi[:(n_xi+t+1)].reshape(-1)
        else:
            P = xi[(n_xi+t+1-int(window)):(n_xi+t+1)].reshape(-1) 
        #print(P.shape)
        S_i = S[n_xi + t]
        f_star = true_minimium[S_i]
        x_star = true_minimizer[S_i]

        hat_x = X_update[np.argmin(mu_evaluated)]
        f_hat_x = f_true[S_i].evaluate_true(hat_x)

        minimizer_list.append(hat_x)
        minimum_value_list.append(min(mu_evaluated))
        f_hat_x_list.append(f_hat_x)
        x_star_list.append(x_star)
        f_star_list.append(f_star)
        
        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)
    
    output_Bayes(file_pre, minimizer_list, minimum_value_list, f_hat_x_list, TIME_RE, x_star_list, f_star_list, seed, n_initial, n_timestep, n_xi, S[n_xi:])

def Experiment_noRS_logkde(seed):
    if method == 'bayesian_hist':
        no_RS_bayesian_kde(seed)

import argparse
parser = argparse.ArgumentParser(description='SwitchingBRO-algo')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument('-t','--time', type=str, help = 'Date of the experiments, e.g. 20241121',default = '20250320')
parser.add_argument('-method','--method', type=str, help = 'switching/hist/exp/lognorm/MAP_exp',default = "switching_joint")
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

args = parser.parse_args()
cmd = ['-t','20260605-no-RS-kde-gaus-syn2-3regime-50data-seed-60-69', 
       '-method', 'bayesian_hist',
       '-problem', 'synthetic2', 
       '-smacro', '60',###60
       '-macro','70',###70 
       '-n_xi', '50',###50 (note)
       '-n_timestep', '25',###25
       '-n_initial', '20', ###20
       '-n_i', '30',###30  
       '-n_rep', '100',###100  
       '-n_candidate','100',###100 
       '-window', 'all',
       '-n_components', '3', ###3
       '-known_sigma', '3.0' ###3.0
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
        Experiment_noRS_logkde(seed)
        print('That took {:.2f} seconds'.format(time.time() - t0))
    
    print('That took {:.2f} seconds'.format(time.time() - starttime))