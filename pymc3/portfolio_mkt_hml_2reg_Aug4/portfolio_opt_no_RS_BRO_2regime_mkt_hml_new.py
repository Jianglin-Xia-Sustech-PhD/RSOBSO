import time
import random
from switching_code.switchingBO_edit_obj_updated_newest_gaussian_portfolio import *
pd.options.display.float_format = '{:.2f}'.format

def gaussian_BRO_streaming(seed):

    print(method) #method='switching_joint' 
    print('number of states', n_components)

    df0 = pd.read_csv('/root/rs_bro92/RSDR_CVaR_dataset/3_MKT2_Monthly_scaled_mkt_hml.csv')
    # given + upcoming: 200401-200712, 200801-200912 (48+24=72)
    df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20100101)].reset_index(drop=True)
    #df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20080301)].reset_index(drop=True)
    returns = df_used[["Mkt-RF", "HML"]].to_numpy() # 2 selected assets

    # t = 0, given initial observations, get n_initial sets of posterior samples, n_observations = n_xi + (t)
    initial_posterior_0 = sample_posterior_2d_gaussian_uniform_prior(returns[:n_xi], mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, n_components*n_initial, seed)

    mu1_samples = initial_posterior_0['mu1']    # shape: (n_samples,)
    sigma1_samples = initial_posterior_0['sigma1']
    mu2_samples = initial_posterior_0['mu2']
    sigma2_samples = initial_posterior_0['sigma2']

    initial_params = np.column_stack([mu1_samples, sigma1_samples, mu2_samples, sigma2_samples])

    # initial experiment design, get n_components*n_initial design points x
    D1 = initial_design_sum1(n_components*n_initial, dimension_x, seed = seed)

    # actually joint modeling x and lambda (no reciprocal)
    D_update = np.concatenate((D1, initial_params), axis=1)
    X_update = np.unique(D_update[:,:dimension_x], axis = 0)#pure BRO, still GP joint modeling

    Y_list = list()

    # give an iteration to get observed values
    for sample_idx in range(D1.shape[0]):

        xx = D1[sample_idx, :dimension_x] 
        P1 = initial_params[sample_idx]
        Y_list.append(f.evaluate(xx, P1, n_rep, seed))
    
    Y1 = np.array(Y_list).reshape(-1,1)
    Y_update = Y1

    # here, I get the initial GP model Z_n(x,lambda), the validated GP joint model
    gp = GP_model(D_update, Y_update)

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    r_tomorrow_list = list()
    TIME_RE = list()

    X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)

    # generate MCMC sample (N_MC*(1+n_iteration))
    sample_count = n_MCMC * (1 + n_iteration)

    MCMC_posterior_0 = sample_posterior_2d_gaussian_uniform_prior(returns[:n_xi], mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, sample_count, seed)

    mu1_samples = MCMC_posterior_0['mu1']    # shape: (n_samples,)
    sigma1_samples = MCMC_posterior_0['sigma1']
    mu2_samples = MCMC_posterior_0['mu2']
    sigma2_samples = MCMC_posterior_0['sigma2']

    MCMC_posterior_params = np.column_stack([mu1_samples, sigma1_samples, mu2_samples, sigma2_samples])

    # take the first N_MC for MCMC mean
    mu_evaluated_list = list()
    mu_g_list = list()
    sigma_g_list = list()
    posterior_lam_list = list()

    for mcmc_idx in range(n_MCMC):
        # no reciprocal
        mu_evaluated, _ = predicted_mean_std_single_param_vector(X_update, MCMC_posterior_params[mcmc_idx], gp)
        mu_g, sigma_g = predicted_mean_std_single_param_vector(X_, MCMC_posterior_params[mcmc_idx], gp)
        mu_evaluated_list.append(mu_evaluated)
        mu_g_list.append(mu_g)
        sigma_g_list.append(sigma_g)
        posterior_lam_list.append(MCMC_posterior_params[mcmc_idx])
    
    MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list)
    MCMC_mean_mu_g = mean_average(mu_g_list)
    MCMC_mean_sigma_g = sigma_average(sigma_g_list)

    posterior_random_lam = random.choice(posterior_lam_list) #random sampling 

    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()

        for ego_iter in range(n_iteration):
            print("iteration", ego_iter)
            #print(f'posterior random lambda at {ego_iter}th EGO iteration: {posterior_random_lam}')

            record = {'seed': seed,
                    'timestep': t,
                    'ego_iter': ego_iter
                    }

            # 拆解 posterior_random_lam 成 4 个独立字段
            for i, val in enumerate(posterior_random_lam):
                record[f'posterior_random_lam_dim{i+1}'] = val  # dim1 到 dim4
            
            df = pd.DataFrame([record])
            df.to_csv('log_portfo_2regime_48data_no_RS_BRO_mkt_hml_new.csv', mode='a', header=not os.path.exists('log_portfo_2regime_48data_no_RS_BRO_mkt_hml_new.csv'), index=False)

            #1. Algorithm for x 
            D_new = EGO(X_, MCMC_mean_mu_evaluated, MCMC_mean_mu_g, MCMC_mean_sigma_g)

            # 2. Algorithm for lambda (random sampling of 100 samples)
            # no reciprocal
            D_update, Y_update = update_data_single_regime_gaussian_portfolio(D_new, D_update, Y_update, posterior_random_lam, n_rep, f, seed=None)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0) 

            # update GP model and make prediction
            gp = GP_model(D_update, Y_update)

            X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)

            mu_evaluated_list = list()
            mu_g_list = list()
            sigma_g_list = list()
            posterior_lam_list = list()

            # generate N_MCMC samples to update the GP model Z_n
            for mcmc_idx in range(n_MCMC): 
                sample_idx = (ego_iter + 1) * n_MCMC + mcmc_idx

                mu_evaluated, _ = predicted_mean_std_single_param_vector(X_update, MCMC_posterior_params[sample_idx], gp)
                mu_g, sigma_g = predicted_mean_std_single_param_vector(X_, MCMC_posterior_params[sample_idx], gp)

                mu_evaluated_list.append(mu_evaluated)
                mu_g_list.append(mu_g)
                sigma_g_list.append(sigma_g)
                posterior_lam_list.append(MCMC_posterior_params[sample_idx])
            
            MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list)
            MCMC_mean_mu_g = mean_average(mu_g_list) 
            MCMC_mean_sigma_g = sigma_average(sigma_g_list)

            posterior_random_lam = random.choice(posterior_lam_list)
        
        # 4. Update x^* (tomorrow best design point)
        hat_x = X_update[np.argmin(MCMC_mean_mu_evaluated)]
        returns_tomorrow = returns[n_xi+t]
        portfolio_return_tomorrow = np.dot(hat_x, returns_tomorrow)
        minimizer_list.append(hat_x)
        minimum_value_list.append(min(MCMC_mean_mu_evaluated))
        r_tomorrow_list.append(portfolio_return_tomorrow)
                
        X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)

        # input data updated, generate new MCMC sample (N_MC*(1+n_iteration))
        MCMC_posterior_0 = sample_posterior_2d_gaussian_uniform_prior(returns[:n_xi+t+1], mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, sample_count, seed)

        mu1_samples = MCMC_posterior_0['mu1']    # shape: (n_samples,)
        sigma1_samples = MCMC_posterior_0['sigma1']
        mu2_samples = MCMC_posterior_0['mu2']
        sigma2_samples = MCMC_posterior_0['sigma2']

        MCMC_posterior_params = np.column_stack([mu1_samples, sigma1_samples, mu2_samples, sigma2_samples])

        # take the first N_MC for MCMC mean
        mu_evaluated_list = list()
        mu_g_list = list()
        sigma_g_list = list()
        posterior_lam_list = list()

        for mcmc_idx in range(n_MCMC):

            mu_evaluated, _ = predicted_mean_std_single_param_vector(X_update, MCMC_posterior_params[mcmc_idx], gp)
            mu_g, sigma_g = predicted_mean_std_single_param_vector(X_, MCMC_posterior_params[mcmc_idx], gp)

            mu_evaluated_list.append(mu_evaluated)
            mu_g_list.append(mu_g)
            sigma_g_list.append(sigma_g)

            posterior_lam_list.append(MCMC_posterior_params[mcmc_idx])
        
        MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list)
        MCMC_mean_mu_g = mean_average(mu_g_list) 
        MCMC_mean_sigma_g = sigma_average(sigma_g_list)

        posterior_random_lam = random.choice(posterior_lam_list)

        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)
    
    output_portfolio(file_pre, minimizer_list, minimum_value_list, r_tomorrow_list, TIME_RE, seed, n_initial, n_timestep, n_xi)

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
parser.add_argument('-sigma_prior_lower', '--sigma_prior_lower', type=float, help='Uniform prior lower of sigma', default=5.0)
parser.add_argument('-sigma_prior_upper', '--sigma_prior_upper', type=float, help='Uniform prior upper of sigma', default=5.0)

args = parser.parse_args()
cmd = ['-t','p0104-no-RS-BRO-portfolio-2regime-mkt-hml-new-48data-seed40-44', 
       '-method', 'switching_joint',
       '-problem', 'portfolio', 
       '-smacro', '40',###60
       '-macro','45',###70 
       '-n_xi', '48',###48(note) 
       '-n_timestep', '24',###24
       '-n_initial', '20', ###20
       '-n_i', '30',###30  
       '-n_rep', '1000',###1000  
       '-n_candidate','100',###100 
       '-window', 'all',
       '-n_MCMC', '100', ###100
       '-n_components', '2', ###2
       '-mu_prior_lower', '0.0', ###0.0
       '-mu_prior_upper', '20.0', ###50.0
       '-sigma_prior_lower', '0.1', ###0.1 
       '-sigma_prior_upper', '10.0', ###20.0 
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
mu_prior_lower = args.mu_prior_lower # prior distribution mean for mu
mu_prior_upper = args.mu_prior_upper # prior distribution stdev for mu
sigma_prior_lower = args.sigma_prior_lower # prior distribution mean for sigma
sigma_prior_upper = args.sigma_prior_upper # prior distribution stdev for sigma

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
        Experiment_noRS_BRO_mean_calculate(seed)
        print('That took {:.2f} seconds'.format(time.time() - t0))
    
    print('That took {:.2f} seconds'.format(time.time() - starttime))

