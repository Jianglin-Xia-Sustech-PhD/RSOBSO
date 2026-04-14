from hmmlearn import hmm
import numpy as np
from pyDOE import *
import random
import warnings
warnings.filterwarnings("ignore")
import time
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
from switching_code.switchingBO_edit_obj_updated_newest_gaussian_portfolio import *

def SwitchingBRO_streaming(seed):

    print(method)
    print('number of states', n_components)
    startprob = np.ones(n_components) / n_components # initial probability updated
    print('start_prob', startprob)

    df0 = pd.read_csv('/home/admin1/HEBO/MCMC/pymc3-hmm-main/RSDR_CVaR_dataset/9_Country2_Monthly_scaled_all_more.csv')
    # given + upcoming: 199701-200712, 200801-200912 (187+24=211)
    df_used = df0[(df0['Date'] >= 19970101) & (df0['Date'] < 20100101)].reset_index(drop=True)
    #df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20080301)].reset_index(drop=True)
    asset_columns = [col for col in df0.columns if col != "Date"]
    returns = df_used[asset_columns].to_numpy() # all 9 country assets

    # t = 0, given initial observations, get n_initial sets of posterior samples, n_observations = n_xi + (t)
    switching_posterior = bayesian_hmm_posterior_gaussian_nd_uniform_std(mean_sample_count, seed, returns[:n_xi], n_components,mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)
    
    regime_params_all = {}

    # 假设 n_assets = 9
    n_assets = returns.shape[1] 

    for i in range(1, n_components + 1):
        # 1. 提取后验数据: 形状 (chains, draws, 9)
        # 使用 stack 或 reshape 将 chains 和 draws 合并
        mu_samples = switching_posterior.posterior[f"mu_regime_{i}"].values
        sigma_samples = switching_posterior.posterior[f"sigma_regime_{i}"].values
        
        # 合并前两个维度 -> (n_draws_total, 9)
        mu_flat = mu_samples.reshape(-1, n_assets)
        sigma_flat = sigma_samples.reshape(-1, n_assets)
        
        # 2. 构造存储结构
        # 原来是 [mu1, sigma1, mu2, sigma2]，现在我们按顺序交替放置：
        # [mu_asset1, sigma_asset1, mu_asset2, sigma_asset2, ..., mu_asset9, sigma_asset9]
        # 这样总共会有 18 列
        combined = np.empty((mu_flat.shape[0], n_assets * 2))
        combined[:, 0::2] = mu_flat     # 偶数列存 mu
        combined[:, 1::2] = sigma_flat  # 奇数列存 sigma
        
        regime_params_all[i] = combined
    
    regime_params_all_mean = {} 
    for i in range(1, n_components + 1):
        # 这里的 mean(axis=0) 会得到长度为 18 的向量：
        # [mean_mu1, mean_sigma1, ..., mean_mu9, mean_sigma9]
        regime_params_all_mean[i] = regime_params_all[i].mean(axis=0)  # shape: (18,)
    
    regime_params_all_mean_array = np.vstack([
    regime_params_all_mean[i] for i in range(1, n_components + 1)])
    
    posterior_p_all = {}
    for i in range(1, n_components + 1):
        p = switching_posterior.posterior[f"p_{i}"].values  # shape: (chain, draw, n_components)
        posterior_p_all[i] = p.reshape(-1, p.shape[-1])  # flatten chain and draw
    
    posterior_p_all_mean_normalized = {} ##########

    for i in range(1, n_components + 1):
        p_mean = posterior_p_all[i].mean(axis=0)
        p_mean_normalized = p_mean / p_mean.sum()
        posterior_p_all_mean_normalized[i] = p_mean_normalized  
    
    # weights corresponding to posterior mean
    # --- 升级后的 9 维适配版本 ---
    model = GaussianHMM(n_components=n_components, covariance_type="diag", random_state=seed)

    # 1. 初始状态概率与转移矩阵 (逻辑保持通用)
    model.startprob_ = startprob
    model.transmat_ = np.vstack([posterior_p_all_mean_normalized[i] for i in range(1, n_components + 1)])

    # 2. 构造 9 维 Means (从 18 维均值向量中提取偶数位: 0, 2, 4, ..., 16)
    means = np.vstack([
        regime_params_all_mean[i][0::2]  # 自动提取所有资产的 mu
        for i in range(1, n_components + 1)
    ])
    model.means_ = means  # shape: (n_components, 9)

    # 3. 构造 9 维 Covars (从 18 维中提取奇数位并平方: 1, 3, 5, ..., 17)
    covars = np.vstack([
        regime_params_all_mean[i][1::2]**2 # 自动提取所有资产的 sigma 并转为 variance
        for i in range(1, n_components + 1)
    ])
    model.covars_ = covars  # shape: (n_components, 9)

    # 4. 预测当前状态并推导下一阶段权重
    # 确保 returns[:n_xi] 此时是 (n_xi, 9) 的维度
    predicted_proba = model.predict_proba(returns[:n_xi])[-1:]
    weights = np.matmul(predicted_proba, model.transmat_)

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    r_tomorrow_list = list()
    TIME_RE = list()
    
    # initial experiment design, get n_initial design points x
    # 1. 初始权重生成 (n_initial, 9)
    D1 = initial_design_sum1(n_initial, dimension_x, seed=seed)

    Y_list = []
    D_update_list = []

    # 2. 使用通用循环遍历所有状态 (适配 n_components)
    for i in range(1, n_components + 1):
        # 提取当前状态的 18 维参数 [mu1, sigma1, ..., mu9, sigma9]
        P_state = regime_params_all_mean[i] # shape: (18,)
        
        # --- 计算 Y (观测值) ---
        for sample_idx in range(n_initial):
            xx = D1[sample_idx, :dimension_x]
            # 调用 9 维 evaluate 函数
            y_val = f.evaluate(xx, P_state, n_rep, seed)
            Y_list.append(y_val)
            
        # --- 构造 D (特征矩阵) ---
        # 将 18 维参数横向展开并复制 n_initial 次
        param_repeated = np.tile(P_state, (n_initial, 1)) # shape: (n_initial, 18)
        # 拼接权重 (9列) 和 参数 (18列) -> (n_initial, 27)
        D_regime = np.concatenate((D1, param_repeated), axis=1)
        D_update_list.append(D_regime)

    # 3. 最终汇总
    Y_update = np.array(Y_list).reshape(-1, 1) # (n_initial * n_components, 1)
    D_update = np.concatenate(D_update_list, axis=0) # (n_initial * n_components, 27)

    # here, I get the initial GP model Z(x,lambda), the validated GP model
    gp = GP_model(D_update, Y_update)

    # Get the unique X values (no duplicates)
    X_update = np.unique(D_update[:, :dimension_x], axis=0)
    
    X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)

    mu_evaluated, _ = predicted_mean_std_joint_model_gaussian_params(X_update, regime_params_all_mean_array, weights, gp, n_components)
    mu_g, sigma_g = predicted_mean_std_joint_model_gaussian_params(X_, regime_params_all_mean_array, weights, gp, n_components)

    for t in range(n_timestep):
        print("timestep", t)
        start = time.time()

        for ego_iter in range(n_iteration):
            print("iteration", ego_iter)
            #1. Algorithm for x 
            D_new = EGO(X_, mu_evaluated, mu_g, sigma_g)

            u = np.random.uniform(0, 1)

            # Compute cumulative probabilities for selecting each state
            cumulative_probs = np.cumsum(weights) 

            # Determine which lambda to evaluate, find first index where u ≤ cumulative_prob
            selected_state = np.searchsorted(cumulative_probs, u)

            # record related information to csv file
            record = {
                        'seed': seed,
                        'timestep': t,
                        'ego_iter': ego_iter}
            
            for idx, val in enumerate(weights.flatten()):
                record[f'weights_{idx}'] = val
            
            for state_idx, lam in regime_params_all_mean.items():
                for dim_idx, param in enumerate(lam):
                    record[f'posterior_random_lam_{state_idx}_dim{dim_idx+1}'] = param

            for dim_idx, param in enumerate(regime_params_all_mean[selected_state + 1]):
                record[f'selected_lambda_dim{dim_idx+1}'] = param

            df = pd.DataFrame([record])
            df.to_csv('log_portfo_2regime_48data_RS_plug_9country.csv', mode='a', header=not os.path.exists('log_portfo_2regime_48data_RS_plug_9country.csv'), index=False)

            D_update, Y_update = update_data_switching_joint_general_gaussian_portfolio(D_new, D_update, Y_update, regime_params_all_mean, n_rep, f, selected_state+1)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0) 

            # update GP model and make prediction
            gp = GP_model(D_update, Y_update) 

            X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)

            mu_evaluated, _ = predicted_mean_std_joint_model_gaussian_params(X_update, regime_params_all_mean_array, weights, gp, n_components)
            mu_g, sigma_g = predicted_mean_std_joint_model_gaussian_params(X_, regime_params_all_mean_array, weights, gp, n_components)

        # 4. Update x^* (tomorrow best portfolio weight)
        hat_x = X_update[np.argmin(mu_evaluated)]
        returns_tomorrow = returns[n_xi+t]
        portfolio_return_tomorrow = np.dot(hat_x, returns_tomorrow)
        minimizer_list.append(hat_x)
        minimum_value_list.append(min(mu_evaluated))
        r_tomorrow_list.append(portfolio_return_tomorrow)

        # input data updated, generate new MCMC sample (N_MC*(1+n_iteration))
        switching_posterior = bayesian_hmm_posterior_gaussian_nd_uniform_std(mean_sample_count, seed, returns[:n_xi+t+1], n_components,mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)
        
        regime_params_all = {}
        n_assets = returns.shape[1]  # 应该是 9

        for i in range(1, n_components + 1):
            # 1. 提取当前 Regime 的均值和标准差 (shape 通常是 (chains, draws, n_assets))
            mu_samples = switching_posterior.posterior[f"mu_regime_{i}"].values
            sigma_samples = switching_posterior.posterior[f"sigma_regime_{i}"].values
            
            # 2. 展平 MCMC 链和采样点 -> (total_samples, 9)
            mu_flat = mu_samples.reshape(-1, n_assets)
            sigma_flat = sigma_samples.reshape(-1, n_assets)
            
            # 3. 构造 18 列矩阵，按 [mu1, sigma1, mu2, sigma2... mu9, sigma9] 交替排列
            # 这样能完美对接你后续函数中使用的 [0::2] 和 [1::2] 切片逻辑
            combined = np.empty((mu_flat.shape[0], n_assets * 2))
            combined[:, 0::2] = mu_flat     # 填充均值
            combined[:, 1::2] = sigma_flat  # 填充标准差
            
            regime_params_all[i] = combined
        
        regime_params_all_mean = {} #########
        for i in range(1, n_components + 1):
            regime_params_all_mean[i] = regime_params_all[i].mean(axis=0)  # shape: (4,)
        
        regime_params_all_mean_array = np.vstack([regime_params_all_mean[i] for i in range(1, n_components + 1)])
        
        posterior_p_all = {}
        for i in range(1, n_components + 1):
            p = switching_posterior.posterior[f"p_{i}"].values  # shape: (chain, draw, n_components)
            posterior_p_all[i] = p.reshape(-1, p.shape[-1])  # flatten chain and draw
        
        posterior_p_all_mean_normalized = {} ##########

        for i in range(1, n_components + 1):
            p_mean = posterior_p_all[i].mean(axis=0)
            p_mean_normalized = p_mean / p_mean.sum()
            posterior_p_all_mean_normalized[i] = p_mean_normalized

        # weights corresponding to posterior mean
        model = GaussianHMM(n_components=n_components, covariance_type="diag", random_state=seed)
        model.startprob_ = startprob
        model.transmat_ = np.vstack([posterior_p_all_mean_normalized[i] for i in range(1, n_components + 1)])

        # 1. 提取所有 9 只股票的均值 (mu): 索引 0, 2, 4, ..., 16
        means = np.vstack([
            regime_params_all_mean[i][0::2] 
            for i in range(1, n_components + 1)
        ])
        model.means_ = means  # 形状应为 (2, 9)

        # 2. 提取所有 9 只股票的方差 (sigma^2): 索引 1, 3, 5, ..., 17
        covars = np.vstack([
            regime_params_all_mean[i][1::2]**2 
            for i in range(1, n_components + 1)
        ])
        model.covars_ = covars  # 形状应为 (2, 9)

        # 3. 预测状态概率
        # 确保 returns[:n_xi+t+1] 的列数也是 9
        predicted_proba = model.predict_proba(returns[:n_xi+t+1])[-1:]

        # 4. 计算下一时刻的预测权重 (Probabilistic weights for states)
        weights = np.matmul(predicted_proba, model.transmat_)

        # 7. Calculate Computing time
        Training_time = time.time() - start
        TIME_RE.append(Training_time)

    output_portfolio(file_pre, minimizer_list, minimum_value_list, r_tomorrow_list, TIME_RE, seed, n_initial, n_timestep, n_xi)

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
parser.add_argument('-mu_prior_lower', '--mu_prior_lower', type=float, help='Uniform prior lower of mu', default=5.0)
parser.add_argument('-mu_prior_upper', '--mu_prior_upper', type=float, help='Uniform prior upper of mu', default=5.0)
parser.add_argument('-sigma_prior_lower', '--sigma_prior_lower', type=float, help='Uniform prior lower of sigma', default=5.0)
parser.add_argument('-sigma_prior_upper', '--sigma_prior_upper', type=float, help='Uniform prior upper of sigma', default=5.0)
parser.add_argument('-mean_sample_count','--mean_sample_count', type=int, help = 'number of posterior samples to calculate posterior mean (follow from BRO: N_mc*ego_steps)', default = 3000)

args = parser.parse_args()
cmd = ['-t','30260305-RS-plug-portfolio-9Country-2regime-48data-seed40-41', 
       '-method', 'switching_joint',
       '-problem', 'portfolio', 
       '-smacro', '40',###40
       '-macro','42',###41 
       '-n_xi', '187',###187 (change)
       '-n_timestep', '24',###24
       '-n_initial', '20', ###20
       '-n_i', '30',###30
       '-n_rep', '1000',###1000 (more) 
       '-n_candidate','100',###100
       '-window', 'all',
       '-n_MCMC', '100',##100
       '-n_components', '2',###2 
       '-mu_prior_lower', '0.0', ###0.0 (change)
       '-mu_prior_upper', '50.0', ###50.0 (change)
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
n_MCMC = args.n_MCMC
n_components = args.n_components #number of hidden states
mu_prior_lower = args.mu_prior_lower # prior distribution mean for mu
mu_prior_upper = args.mu_prior_upper # prior distribution stdev for mu
sigma_prior_lower = args.sigma_prior_lower # prior distribution mean for sigma
sigma_prior_upper = args.sigma_prior_upper # prior distribution stdev for sigma
mean_sample_count = args.mean_sample_count 

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
        Experiment_RSBRO(seed)
        print('That took {:.2f} seconds'.format(time.time() - t0))
    
    print('That took {:.2f} seconds'.format(time.time() - starttime))

