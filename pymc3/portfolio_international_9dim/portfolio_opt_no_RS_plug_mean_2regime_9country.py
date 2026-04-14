import time
from switching_code.switchingBO_edit_obj_updated_newest_gaussian_portfolio import *
pd.options.display.float_format = '{:.2f}'.format

def bayesian_posterior_mean_std_nd_uniform_prior(
        observed_data,
        mu_lower=0, mu_upper=50,
        sigma_lower=0.1, sigma_upper=20,
        draws=2000, seed=0):
    """
    对 N 维正态分布，使用 PyMC 在均匀先验下估计每维的 mu 和 sigma 的后验均值。

    参数:
    - observed_data: shape (n, n_assets)，多维观测数据
    - 返回: shape (2 * n_assets,) 的 numpy array: [mu1, sigma1, mu2, sigma2, ...]
    """
    n_assets = observed_data.shape[1]

    with pm.Model() as model:
        # 1. 向量化定义先验 (shape 为资产数量)
        mu = pm.Uniform('mu', lower=mu_lower, upper=mu_upper, shape=n_assets)
        sigma = pm.Uniform('sigma', lower=sigma_lower, upper=sigma_upper, shape=n_assets)

        # 2. 向量化定义似然函数
        pm.Normal('obs', mu=mu, sigma=sigma, observed=observed_data)

        # 3. 采样
        trace = pm.sample(draws=draws, tune=1000, chains=1,
                          random_seed=seed, return_inferencedata=False, progressbar=False)

    # 4. 计算每个维度的后验均值
    # trace['mu'] shape: (draws, n_assets) -> mean(axis=0) 得到 (n_assets,)
    mu_post_means = np.mean(trace['mu'], axis=0)
    sigma_post_means = np.mean(trace['sigma'], axis=0)

    # 5. 交替组合成 [mu1, sigma1, mu2, sigma2, ...]
    posterior_params = np.empty(n_assets * 2)
    posterior_params[0::2] = mu_post_means     # 偶数索引放 mu
    posterior_params[1::2] = sigma_post_means  # 奇数索引放 sigma

    return posterior_params

def EGO_posterior_mean_plug_in(seed):

    print(method) #'method': 'switching_joint'
    print('number of states', n_components)

    df0 = pd.read_csv('/home/admin1/HEBO/MCMC/pymc3-hmm-main/RSDR_CVaR_dataset/9_Country2_Monthly_scaled_all_more.csv')
    # given + upcoming: 199701-200712, 200801-200912 (187+24=211)
    df_used = df0[(df0['Date'] >= 19970101) & (df0['Date'] < 20100101)].reset_index(drop=True)
    #df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20080301)].reset_index(drop=True)
    asset_columns = [col for col in df0.columns if col != "Date"]
    returns = df_used[asset_columns].to_numpy() # all 9 country assets

    if window == "all":
        P = returns[:n_xi] #all observed data, without moving window
    else:
        P = returns[(n_xi - int(window)):n_xi]
    
    # initial experiment design
    D1 = initial_design_sum1(n_components*n_initial, dimension_x, seed = seed)

    # calculate no-rs plug-in lambda
    # 使用升级后的 nd 函数名
    plug_lam_mean = bayesian_posterior_mean_std_nd_uniform_prior(
    P, 
    mu_prior_lower, 
    mu_prior_upper,
    sigma_prior_lower, 
    sigma_prior_upper, 
    mean_sample_count, 
    seed)
    
    plug_lam_matrix = np.tile(plug_lam_mean.reshape(1, -1), (D1.shape[0], 1))
    D_update = np.concatenate((D1, plug_lam_matrix), axis=1) 
    X_update = np.unique(D_update[:,:dimension_x], axis = 0)

    Y_list = list()
    for i in range(D1.shape[0]):
        xx = D1[i,:dimension_x] 
        P1 = plug_lam_matrix[i]
        Y_list.append(f.evaluate(xx, P1, n_rep, seed)) 
    Y1 = np.array(Y_list).reshape(-1,1)
    Y_update = Y1

    gp0 = GP_model(D_update, Y_update) 

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    r_tomorrow_list = list()
    TIME_RE = list()

    ## get prediction and obtain minimizer
    X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)
    mu_g, sigma_g = predicted_mean_std_single_param_vector(X_, plug_lam_mean, gp0)
    mu_evaluated, _ = predicted_mean_std_single_param_vector(X_update, plug_lam_mean, gp0)

    # Global Optimization algorithm
    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()
        #print(f'Plug-in posterior mean for timestep {t} is: {plug_lam_mean}')
        # record related information to csv file
        record = {
                        'seed': seed,
                        'timestep': t
                }
        for i, val in enumerate(plug_lam_mean):
                record[f'posterior_plug_lam_dim{i+1}'] = val
        df = pd.DataFrame([record])
        df.to_csv('log_portfo_2regime_48data_no_RS_plug_9country.csv', mode='a', header=not os.path.exists('log_portfo_2regime_48data_no_RS_plug_9country.csv'), index=False)

        for i in range(n_iteration): 
            print("iteration", i)

            # 1. Algorithm for x 
            D_new = EGO(X_, mu_evaluated, mu_g, sigma_g)

            # 2. Add selected samples
            D_update, Y_update = update_data_single_regime_gaussian_portfolio(D_new, D_update, Y_update, plug_lam_mean, n_rep, f, seed=None)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0) 

            # 3. Update GP model and make prediction 
            gp0 = GP_model(D_update, Y_update)
            X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)

            mu_g, sigma_g = predicted_mean_std_single_param_vector(X_, plug_lam_mean, gp0)
            mu_evaluated, _ = predicted_mean_std_single_param_vector(X_update, plug_lam_mean, gp0)
      
        # 4. Update x^*
        if window == "all":
            P = returns[:(n_xi+t+1)]
        else:
            P = returns[(n_xi+t+1-int(window)):(n_xi+t+1)]
        
        # calculate the next plug-in lambda for the next time stage
        plug_lam_mean = bayesian_posterior_mean_std_nd_uniform_prior(P, mu_prior_lower, mu_prior_upper,
        sigma_prior_lower, sigma_prior_upper, mean_sample_count, seed)

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
     
def Experiment_noRS_plug_mean(seed):
    
    if method == 'switching_joint':
        EGO_posterior_mean_plug_in(seed)

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
parser.add_argument('-n_components','--n_components', type=int, help = 'Number of stages', default = 2)
parser.add_argument('-mu_prior_lower', '--mu_prior_lower', type=float, help='Uniform prior lower of mu', default=5.0)
parser.add_argument('-mu_prior_upper', '--mu_prior_upper', type=float, help='Uniform prior upper of mu', default=5.0)
parser.add_argument('-sigma_prior_lower', '--sigma_prior_lower', type=float, help='Uniform prior lower of sigma', default=5.0)
parser.add_argument('-sigma_prior_upper', '--sigma_prior_upper', type=float, help='Uniform prior upper of sigma', default=5.0)
parser.add_argument('-mean_sample_count','--mean_sample_count', type=int, help = 'number of posterior samples to calculate posterior mean', default = 1500)

args = parser.parse_args()
cmd = ['-t','30260305-no-RS-plug-portfolio-9Country-2regime-48data-seed40-44', 
       '-method', 'switching_joint', #change
       '-problem', 'portfolio', 
       '-smacro', '40',###60
       '-macro','45',###70 
       '-n_xi', '187',###187 (change)
       '-n_timestep', '24',###24
       '-n_initial', '20', ###20
       '-n_i', '30',###30  
       '-n_rep', '1000',###1000  
       '-n_candidate','100',###100 
       '-window', 'all',
       '-n_components', '2',###2
       '-mu_prior_lower', '0.0', ###0.0
       '-mu_prior_upper', '50.0', ###50.0
       '-sigma_prior_lower', '0.1', ###0.1 
       '-sigma_prior_upper', '20.0', ###20.0
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
mu_prior_lower = args.mu_prior_lower # prior distribution mean for mu
mu_prior_upper = args.mu_prior_upper # prior distribution stdev for mu
sigma_prior_lower = args.sigma_prior_lower # prior distribution mean for sigma
sigma_prior_upper = args.sigma_prior_upper # prior distribution stdev for sigma
mean_sample_count = args.mean_sample_count # number of posterior samples to calculate posterior mean

f = portfolio_problem_9stocks()
dimension_x = 9

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