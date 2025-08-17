import time
from switching_code.switchingBO_edit_obj_updated_newest_gaussian_portfolio import *
pd.options.display.float_format = '{:.2f}'.format

def no_RS_bayesian_kde(seed):

    print(method) #method: 'bayesian_hist'
    print('number of states', n_components)

    df0 = pd.read_csv('/root/rs_bro92/RSDR_CVaR_dataset/3_MKT2_Monthly_scaled.csv')
    # given + upcoming: 200401-200712, 200801-200912 (48+24=72)
    df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20100101)].reset_index(drop=True)
    #df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20080301)].reset_index(drop=True)
    returns = df_used[["Mkt-RF", "SMB"]].to_numpy() # 2 selected assets
    
    if window == "all":
        P = returns[:n_xi] #all observed data, without moving window
    else:
        P = returns[(n_xi - int(window)):n_xi]
    
    # initial experiment design
    D1 = initial_design_sum1(n_components*n_initial, dimension_x, seed = seed)

    Y_list = list()
    for i in range(D1.shape[0]):
        xx = D1[i,:dimension_x] 
        Y_list.append(f.evaluate_kde(xx, P, n_rep, seed = seed)) #fixed, add seed
    
    Y1 = np.array(Y_list).reshape(-1,1)

    gp0 = GP_model(D1, Y1) 

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    r_tomorrow_list = list()
    TIME_RE = list()

    D_update = D1
    X_update = np.unique(D_update, axis = 0)
    Y_update = Y1

    ## get prediction and obtain minimizer
    X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)
    mu_g, sigma_g = gp0.predict(X_, return_std= True) 
    mu_evaluated, _ = gp0.predict(X_update, return_std= True) 

    ## Global Optimization algorithm
    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()
        # save the fitted kde plot in the given package for each timestep
        print(f"Saved the KDE plot for timestep {t}")
        saved_path = f"/home/admin1/HEBO/MCMC/pymc3-hmm-main/kde_plot/portfolio_2regime/seed{seed}_kde_plot_timestep{t}.pdf"
        update_kde_and_plot_2d(P, save_path=saved_path, seed=seed)
       
        for i in range(n_iteration):
            print("iteration", i)
            
            # 1. Algorithm for x 
            D_new = EGO(X_, mu_evaluated, mu_g, sigma_g)

            # 2. Add selected samples
            D_update, Y_update = update_data_kde_2d(D_new, D_update, Y_update, P, n_rep, f, seed=seed)
            #add seed to get the fixed fitted kde 
            X_update = np.unique(D_update, axis = 0) 

            # 3. Update GP model and make prediction 
            gp0 = GP_model(D_update, Y_update)
            X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)
            mu_g, sigma_g = gp0.predict(X_, return_std= True) 
            mu_evaluated, _ = gp0.predict(X_update, return_std= True) 
        
        # 4. Update x^*
        if window == "all":
            P = returns[:(n_xi+t+1)]
        else:
            P = returns[(n_xi+t+1-int(window)):(n_xi+t+1)]
        
        hat_x = X_update[np.argmin(mu_evaluated)]
        returns_tomorrow = returns[n_xi+t]
        portfolio_return_tomorrow = np.dot(hat_x, returns_tomorrow)
        minimizer_list.append(hat_x)
        minimum_value_list.append(min(mu_evaluated))
        r_tomorrow_list.append(portfolio_return_tomorrow)
        
        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)
    
    output_portfolio(file_pre, minimizer_list, minimum_value_list, r_tomorrow_list, TIME_RE, seed, n_initial, n_timestep, n_xi)

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
cmd = ['-t','p0105-no-RS-kde-portfolio-2regime-mkt-smb-48data-seed40-44', 
       '-method', 'bayesian_hist',
       '-problem', 'portfolio', 
       '-smacro', '40',###60
       '-macro','45',###70 
       '-n_xi', '48',###48 (note)
       '-n_timestep', '24',###24
       '-n_initial', '20', ###20
       '-n_i', '30',###30  
       '-n_rep', '1000',###1000  
       '-n_candidate','100',###100 
       '-window', 'all',
       '-n_components', '2' ###2
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
        Experiment_noRS_logkde(seed)
        print('That took {:.2f} seconds'.format(time.time() - t0))
    
    print('That took {:.2f} seconds'.format(time.time() - starttime))