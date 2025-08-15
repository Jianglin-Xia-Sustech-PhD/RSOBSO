### real-world experiment
### 4-regime inventory problem, exponential demand
### the regime data are from real-world data
### initial 96 month data given
### additional 24 months (time stages)(great recession period)
# no consider regime switching
# bayesian version of histogram
# use nonparametric log kde to fit the data, get fitted pdf, and sample data
# no parameters here, so no joint modeling here
from hmmlearn import hmm
import numpy as np
import pandas as pd
from pyDOE import *
import warnings
warnings.filterwarnings("ignore")
import time
pd.options.display.float_format = '{:.2f}'.format
from switching_code.switchingBO import *

def no_RS_bayesian_kde(seed):

    print(method) #method: 'bayesian_hist'
    print('number of states', n_components)

    # read real regime data and generate corresponding random xi here
    df0 = pd.read_csv('/home/admin1/HEBO/MCMC/pymc3-hmm-main/RSDR_CVaR_dataset/weathers_state.csv')
    df0['state_name'] = df0['state'].map({1: 'HGHC', 2: 'HGLC', 3: 'LGHC', 4: 'LGLC'})
    df0['state'] = df0['state'].replace({2: 0, 4: 2})
    # given + upcoming: 200001-200712, 200801-200912 (96+24=120), use recession as out-of-sample
    df_used = df0[(df0['Date'] >= 20000101) & (df0['Date'] < 20100101)].reset_index(drop=True)
    S = df_used['state'].to_numpy() # real state data

    # random seed setting to generate xi
    rng = np.random.default_rng(seed=seed)
    xi = np.empty(len(S))
    scale_map = {0: 30, 1: 18, 2: 12, 3: 1}
    # regime 0: rate 1/30, regime 1: rate 1/18, regime 2: rate 1/12, regime 3: rate 1 (increasing rate lambda)
    for i, s in enumerate(S):
        xi[i] = stats.expon(scale=scale_map[s]).rvs(random_state=rng)

    if window == "all":
        P = xi[:n_xi].reshape(-1) #all observed data, without moving window
    else:
        P = xi[(n_xi - int(window)):n_xi].reshape(-1)
    
    # initial experiment design
    D1 = inital_design(n_components*n_initial, seed, lower_bound, upper_bound) #n_components*n_initial, only change
    Y_list = list()
    for i in range(D1.shape[0]):
        xx = D1[i,:dimension_x] 
        Y_list.append(f.evaluate(xx, P, n_rep, method, seed = seed)) ###add seed

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

    # # Global Optimization algorithm
    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()
        # save the fitted logkde plot in the given package for each timestep
        print(f"Saved the logKDE plot for timestep {t}")
        saved_path = f"/home/admin1/HEBO/MCMC/pymc3-hmm-main/kde_plot/real_inventory/4regime_96data_redesign_recession/seed{seed}_kde_plot_timestep{t}.pdf"
        update_kde_positive_and_plot(P, save_path=saved_path, seed=seed)

        for i in range(n_iteration): #no need to change here
            print("iteration", i)
            
            # 1. Algorithm for x 
            D_new = EGO(X_, mu_evaluated, mu_g, sigma_g)

            # 2. Add selected samples #add seed to get the fixed fitted logkde 
            D_update, Y_update = update_data(D_new, D_update, Y_update, P, n_rep, method, f, seed = seed) 
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

args = parser.parse_args()
cmd = ['-t','20250609-no-RS-logkde-4regime-real-96data-redesign-recession-seed-60-69', 
       '-method', 'bayesian_hist', ###change
       '-problem', 'inventory', 
       '-smacro', '60', ##60
       '-macro','70', ##70
       '-n_xi', '96', ##96(change)
       '-n_timestep', '24', ###24
       '-n_initial', '20', ###20
       '-n_i', '30', ###30 
       '-n_rep', '10', ###10 
       '-n_candidate','100', ###100 
       '-window', 'all',
       '-n_components', '4' ###4 regimes
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

true_lambdas = np.array([[1/30], [1/18], [1/12], [1]]) #increasing order, updated

switching_distributions = []
for s in range(n_components):
    switching_distributions.append({
        'dist': "exp",
        "rate": true_lambdas[s]
    })

f1 = inventory_problem(switching_distributions[0])
f2 = inventory_problem(switching_distributions[1])
f3 = inventory_problem(switching_distributions[2])
f4 = inventory_problem(switching_distributions[3])

f1.x_star = np.array([69,191]) 
f2.x_star = np.array([57,118])
f3.x_star = np.array([35,87])
f4.x_star = np.array([1,70])

f1.f_star = 222
f2.f_star = 135
f3.f_star = 97
f4.f_star = 38

f_true = [f1,f2,f3,f4]
f = f1
true_minimizer = [f1.x_star, f2.x_star, f3.x_star, f4.x_star]
true_minimium = [f1.f_star, f2.f_star, f3.f_star, f4.f_star]
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