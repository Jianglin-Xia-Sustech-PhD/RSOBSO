### real-world experiment
### 2-regime inventory problem, exponential demand
### the regime data are from real-world data
### initial 48 month data given
### additional 24 months (time stages)(great recession period)
# no consider regime switching
# no BRO framework, directly plug-in posterior mean (conjugate calculate)
import time
from switching_code.switchingBO import * 
pd.options.display.float_format = '{:.2f}'.format

def EGO_posterior_mean_plug_in(seed):

    print(method) #'method' to control
    print('number of states', n_components)

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
    
    if window == "all":
        P = xi[:n_xi].reshape(-1) 
    else:
        P = xi[(n_xi - int(window)):n_xi].reshape(-1)
    
    # initial experiment design
    D1 = inital_design(n_components*n_initial, seed, lower_bound, upper_bound) #n_components*n_initial
    # calculate no-rs plug-in lambda
    plug_lam_mean = compute_posterior_mean_exponential(P, alpha_prior=lam_prior_alpha, beta_prior=lam_prior_beta)

    D_update = np.concatenate((D1, np.full((D1.shape[0], 1), 1 / plug_lam_mean)), axis=1) ##joint modeling
    X_update = np.unique(D_update[:,:dimension_x], axis = 0)

    Y_list = list()
    for i in range(D1.shape[0]):
        xx = D1[i,:dimension_x] 
        P1 = {'dist':"exp", "rate": plug_lam_mean}
        Y_list.append(f.evaluate(xx, P1, n_rep, method)) 
    Y1 = np.array(Y_list).reshape(-1,1)
    Y_update = Y1
 
    gp0 = GP_model(D_update, Y_update) 

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    f_hat_x_list = list()
    x_star_list = list()
    f_star_list = list()

    TIME_RE = list()

    ## get prediction and obtain minimizer
    X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
    mu_g, sigma_g = predicted_mean_std_single_lambda(X_, plug_lam_mean, gp0)  
    mu_evaluated, _ = predicted_mean_std_single_lambda(X_update, plug_lam_mean, gp0)

    # # Global Optimization algorithm
    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()
        print(f'Plug-in posterior mean for timestep {t} is: {plug_lam_mean}')
        # record related information to csv file
        record = {
                        'seed': seed,
                        'timestep': t,
                        'plug_mean': plug_lam_mean
                }
        df = pd.DataFrame([record])
        df.to_csv('log_real_inventory_2regime_48data_joint_no_RS_plug_mean.csv', mode='a', header=not os.path.exists('log_real_inventory_2regime_48data_joint_no_RS_plug_mean.csv'), index=False)

        for i in range(n_iteration): 
            print("iteration", i)
            
            # 1. Algorithm for x 
            D_new = EGO(X_, mu_evaluated, mu_g, sigma_g)

            # 2. Add selected samples
            distribution1_updated = {'dist':"exp", "rate": plug_lam_mean}
            D_update, Y_update = update_data_switching_joint_single(D_new, D_update, Y_update, distribution1_updated, n_rep, method, f)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0) 

            # 3. Update GP model and make prediction 
            gp0 = GP_model(D_update, Y_update)
            X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
            mu_g, sigma_g = predicted_mean_std_single_lambda(X_, plug_lam_mean, gp0)  
            mu_evaluated, _ = predicted_mean_std_single_lambda(X_update, plug_lam_mean, gp0)

        # 4. Update x^*
        if window == "all":
            P = xi[:(n_xi+t+1)].reshape(-1)
        else:
            P = xi[(n_xi+t+1-int(window)):(n_xi+t+1)].reshape(-1) 

        # calculate the next plug-in lambda for the next time stage
        plug_lam_mean = compute_posterior_mean_exponential(P, alpha_prior=lam_prior_alpha, beta_prior=lam_prior_beta)
        
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

def Experiment_noRS_plug_mean(seed):
    
    if method == 'switching_joint':
        EGO_posterior_mean_plug_in(seed)

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
parser.add_argument('-lam_prior_alpha', '--lam_prior_alpha', type=float, help='Prior alpha of lam', default=1.0)
parser.add_argument('-lam_prior_beta', '--lam_prior_beta', type=float, help='Prior beta of lam', default=1.0)

args = parser.parse_args()
cmd = ['-t','20251101-no-RS-plug-mean-2regime-real-48data-joint-seed-45-49', 
       '-method', 'switching_joint', #change (joint modeling)
       '-problem', 'inventory', 
       '-smacro', '45', ##60
       '-macro','50', ###70
       '-n_xi', '48', ##48 (change)  
       '-n_timestep', '24',###24 
       '-n_initial', '20', ###20
       '-n_i', '30', ###30
       '-n_rep', '10',###10  
       '-n_candidate','100',###100 
       '-window', 'all',
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
n_components = args.n_components
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
        Experiment_noRS_plug_mean(seed)
        print('That took {:.2f} seconds'.format(time.time() - t0))
    
    print('That took {:.2f} seconds'.format(time.time() - starttime))