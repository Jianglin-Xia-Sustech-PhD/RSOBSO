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
    initial_posterior_0 = bayesian_hmm_posterior_gaussian_nd_uniform_std(n_initial, seed, returns[:n_xi], n_components,mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)

    regime_params_all = {}

    # 假设 n_assets = 9
    n_assets = returns.shape[1] 

    for i in range(1, n_components + 1):
        # 1. 提取后验数据: 形状 (chains, draws, 9)
        # 使用 stack 或 reshape 将 chains 和 draws 合并
        mu_samples = initial_posterior_0.posterior[f"mu_regime_{i}"].values
        sigma_samples = initial_posterior_0.posterior[f"sigma_regime_{i}"].values
        
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
    
    # initial experiment design, get n_initial design points x
    D1 = initial_design_sum1(n_initial, dimension_x, seed = seed)

    D_update_all = {i: np.concatenate((D1, regime_params_all[i]), axis=1)
    for i in regime_params_all}

    D_update = np.concatenate(list(D_update_all.values()), axis=0)
    X_update = np.unique(D_update[:, :dimension_x], axis = 0)

    Y_all_regimes = []

    # 假设 n_components = 2, n_initial = 20
    for r_idx in range(1, n_components + 1):
        current_regime_y = []
        
        for sample_idx in range(n_initial):
            # 1. 提取第 sample_idx 个决策变量 (9维权重)
            xx = D1[sample_idx, :dimension_x] 
            
            # 2. 提取该样本对应的第 r_idx 个状态的后验参数 (18维: mu, sigma)
            # 结构: [mu1, sigma1, ..., mu9, sigma9]
            P_regime = regime_params_all[r_idx][sample_idx]
            
            # 3. 调用 9 维评估函数 (f 是你之前定义的 portfolio_problem_9stocks 实例)
            val = f.evaluate(xx, P_regime, n_rep, seed)
            current_regime_y.append(val)
            
        Y_all_regimes.append(current_regime_y)

    # 4. 拼接成最终的标签列 Y_update
    # 先水平拼接所有状态的结果，再拉成一列
    Y_update = np.concatenate(Y_all_regimes).reshape(-1, 1)

    # here, I get the initial GP model Z(x,lambda), the validated GP model
    gp = GP_model(D_update, Y_update)

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    r_tomorrow_list = list()
    TIME_RE = list()

    X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)

    # generate MCMC sample (N_MC*(1+n_iteration)), no random sampling, use posterior sample mean
    sample_count = n_MCMC * (1 + n_iteration)
    seed_mcmc = seed + 1
    MCMC_posterior_0 = bayesian_hmm_posterior_gaussian_nd_uniform_std(sample_count, seed_mcmc, returns[:n_xi], n_components,mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)
    
    regime_params_all = {}
    n_assets = returns.shape[1]  # 对应你的 9 只股票

    for i in range(1, n_components + 1):
        # 1. 获取后验数据，形状为 (chains, draws, 9)
        mu_samples = MCMC_posterior_0.posterior[f"mu_regime_{i}"].values
        sigma_samples = MCMC_posterior_0.posterior[f"sigma_regime_{i}"].values
        
        # 2. 将 chains 和 draws 合并，展平为 (total_samples, 9)
        mu_flat = mu_samples.reshape(-1, n_assets)
        sigma_flat = sigma_samples.reshape(-1, n_assets)
        
        # 3. 构造 18 列的矩阵：[mu_1, sigma_1, mu_2, sigma_2, ..., mu_9, sigma_9]
        # 我们先创建一个空矩阵，然后用步长切片填充
        total_samples = mu_flat.shape[0]
        combined = np.empty((total_samples, n_assets * 2))
        
        combined[:, 0::2] = mu_flat     # 偶数列填充所有资产的 mu
        combined[:, 1::2] = sigma_flat  # 奇数列填充所有资产的 sigma
        
        regime_params_all[i] = combined
    
    posterior_p_all = {}

    for i in range(1, n_components + 1):
        p = MCMC_posterior_0.posterior[f"p_{i}"].values  # shape: (chain, draw, n_components)
        posterior_p_all[i] = p.reshape(-1, p.shape[-1])  # flatten chain and draw

    # take the first N_MC for MCMC mean
    mu_evaluated_list = list()
    mu_g_list = list()
    sigma_g_list = list()
    time_weight_list = [[] for _ in range(n_components)]
    posterior_lam_list = [[] for _ in range(n_components)] # to save all the posterior samples in this list

    for mcmc_idx in range(n_MCMC):
        sample_idx = mcmc_idx
        
        # 1. 调用升级后的 N 维权重计算函数 (适配 9 维 obs)
        # 注意：函数名已改为 nd_gaussian
        weights = compute_hmm_weights_from_sample_idx_nd_gaussian(
            sample_idx, 
            returns[:n_xi], 
            n_components,
            posterior_p_all, 
            regime_params_all, 
            startprob
        )
        
        # 2. 构造当前样本的 9 维参数矩阵 (2, 18)
        # 每一行包含：[mu1, sigma1, ..., mu9, sigma9]
        params = np.vstack([regime_params_all[i][sample_idx] for i in range(1, n_components + 1)])
        
        # 3. 调用优化后的 GP 预测函数 (使用你重命名后的 _updated 版本)
        # 预测已观测点 X_update (目前是 40 个点)
        mu_evaluated, _ = predicted_mean_std_joint_model_gaussian_params_updated(
            X_update, params, weights, gp, n_components
        )
        
        # 4. 预测候选点 X_ (通常是较大规模的随机权重组合)
        mu_g, sigma_g = predicted_mean_std_joint_model_gaussian_params_updated(
            X_, params, weights, gp, n_components
        )

        # 5. 结果收集
        mu_evaluated_list.append(mu_evaluated)
        mu_g_list.append(mu_g)
        sigma_g_list.append(sigma_g)

        # 6. 状态权重与后验参数存储（用于后续分析）
        for state_idx in range(n_components):
            # weights[0][state_idx] 是 P(S_{t+1} = state_idx)
            time_weight_list[state_idx].append(weights[0][state_idx])
            # 存储 18 维的参数向量
            posterior_lam_list[state_idx].append(regime_params_all[state_idx+1][sample_idx])
    
    MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list) 
    MCMC_mean_mu_g = mean_average(mu_g_list) 
    MCMC_mean_sigma_g = mean_average(sigma_g_list)

    # we directly sample randomly
    posterior_random_lam_list = [random.choice(posterior_lam_list[state]) for state in range(n_components)]

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

            u = np.random.uniform(0, 1)

            # Compute cumulative probabilities for selecting each state
            cumulative_probs = np.cumsum(time_prob_weight_list) 

            # Determine which lambda to evaluate, find first index where u ≤ cumulative_prob
            selected_state = np.searchsorted(cumulative_probs, u)

            # record related information to csv file
            record = {
                        'seed': seed,
                        'timestep': t,
                        'ego_iter': ego_iter}
            
            for idx, val in enumerate(time_prob_weight_list):
                record[f'time_prob_weight_{idx}'] = val

            for state_idx, lam in enumerate(posterior_random_lam_list):
                for dim_idx, param in enumerate(lam):
                    record[f'posterior_random_lam_{state_idx}_dim{dim_idx+1}'] = param

            for dim_idx, param in enumerate(posterior_random_lam_list[selected_state]):
                record[f'selected_lambda_dim{dim_idx+1}'] = param

            df = pd.DataFrame([record])
            df.to_csv('log_portfo_2regime_48data_RS_BRO_9country.csv', mode='a', header=not os.path.exists('log_portfo_2regime_48data_RS_BRO_9country.csv'), index=False)

            D_update, Y_update = update_data_switching_joint_general_gaussian_portfolio(D_new, D_update, Y_update, posterior_random_lam_list, n_rep, f, selected_state)

            X_update = np.unique(D_update[:,:dimension_x], axis = 0) 

            # update GP model and make prediction
            gp = GP_model(D_update, Y_update) 

            X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)

            # need to calculate the weight here again 
            mu_evaluated_list = list()
            mu_g_list = list()
            sigma_g_list = list()
            time_weight_list = [[] for _ in range(n_components)]
            posterior_lam_list = [[] for _ in range(n_components)]

            # generate N_MCMC samples to update the GP model Z_n   
            for mcmc_idx in range(n_MCMC):
                # 计算当前后验样本池的全局索引
                sample_idx = (ego_iter + 1) * n_MCMC + mcmc_idx
                
                # 1. 计算状态权重 (使用适配 9 维 obs 的 nd 函数)
                # returns[:n_xi + t] 确保了随着迭代进行，信息集在更新
                weights = compute_hmm_weights_from_sample_idx_nd_gaussian(
                    sample_idx, 
                    returns[:n_xi + t], 
                    n_components,
                    posterior_p_all, 
                    regime_params_all, 
                    startprob
                )
                
                # 2. 构造 9 维参数矩阵 (2, 18)
                # 这种写法比手动写 [1], [2] 更安全，且支持 18 列参数
                params = np.vstack([regime_params_all[i][sample_idx] for i in range(1, n_components + 1)])
                
                # 3. 调用向量化加速后的 GP 预测函数
                # 预测当前已有的设计点 (X_update)
                mu_evaluated, _ = predicted_mean_std_joint_model_gaussian_params_updated(
                    X_update, params, weights, gp, n_components
                )
                
                # 预测大规模候选集 (X_)
                mu_g, sigma_g = predicted_mean_std_joint_model_gaussian_params_updated(
                    X_, params, weights, gp, n_components
                )
                
                # 结果收集
                mu_evaluated_list.append(mu_evaluated)
                mu_g_list.append(mu_g)
                sigma_g_list.append(sigma_g)

                # 4. 存储权重与后验参数
                for state_idx in range(n_components):
                    time_weight_list[state_idx].append(weights[0][state_idx])
                    posterior_lam_list[state_idx].append(regime_params_all[state_idx+1][sample_idx])

            MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list)
            MCMC_mean_mu_g = mean_average(mu_g_list)
            MCMC_mean_sigma_g = sigma_average(sigma_g_list)

            posterior_random_lam_list = [random.choice(posterior_lam_list[state]) for state in range(n_components)]

            time_avg_weight_list = [mean_average(time_weight_list[state]) for state in range(n_components)]
            time_avg_total_weight = sum(time_avg_weight_list)
            time_prob_weight_list = [time_avg_weight_list[state] / time_avg_total_weight for state in range(n_components)]

        # 4. Update x^* (tomorrow best portfolio weight)
        hat_x = X_update[np.argmin(MCMC_mean_mu_evaluated)]
        returns_tomorrow = returns[n_xi+t]
        portfolio_return_tomorrow = np.dot(hat_x, returns_tomorrow)
        minimizer_list.append(hat_x)
        minimum_value_list.append(min(MCMC_mean_mu_evaluated))
        r_tomorrow_list.append(portfolio_return_tomorrow)

        X_ = initial_design_sum1(n_candidate, dimension_x, seed = None)
        # input data updated, generate new MCMC sample (N_MC*(1+n_iteration))
        MCMC_posterior_0 = bayesian_hmm_posterior_gaussian_nd_uniform_std(sample_count, seed_mcmc+t+1, returns[:n_xi+t+1], n_components,mu_prior_lower, mu_prior_upper, sigma_prior_lower, sigma_prior_upper, startprob)

        regime_params_all = {}
        n_assets = returns.shape[1]  # 你的股票维度

        for i in range(1, n_components + 1):
            # 2. 提取后验样本：形状为 (chains, draws, 9)
            mu_samples = MCMC_posterior_0.posterior[f"mu_regime_{i}"].values
            sigma_samples = MCMC_posterior_0.posterior[f"sigma_regime_{i}"].values
            
            # 3. 展平 chains 和 draws 为 (total_samples, 9)
            mu_flat = mu_samples.reshape(-1, n_assets)
            sigma_flat = sigma_samples.reshape(-1, n_assets)
            
            # 4. 构造 18 列矩阵：[mu1, sigma1, mu2, sigma2, ..., mu9, sigma9]
            total_samples = mu_flat.shape[0]
            combined = np.empty((total_samples, n_assets * 2))
            
            combined[:, 0::2] = mu_flat     # 填充均值
            combined[:, 1::2] = sigma_flat  # 填充标准差
            
            regime_params_all[i] = combined
        
        posterior_p_all = {}

        for i in range(1, n_components + 1):
            p = MCMC_posterior_0.posterior[f"p_{i}"].values  # shape: (chain, draw, n_components)
            posterior_p_all[i] = p.reshape(-1, p.shape[-1])  # flatten chain and draw
        
        # take the first N_MC for MCMC mean
        mu_evaluated_list = list()
        mu_g_list = list()
        sigma_g_list = list()
        time_weight_list = [[] for _ in range(n_components)]
        posterior_lam_list = [[] for _ in range(n_components)]
        
        for mcmc_idx in range(n_MCMC):
            # 这里通常取前 n_MCMC 个样本来计算当前时刻的期望指标
            sample_idx = mcmc_idx
            
            # 1. 调用适配 9 维的权重计算函数
            # 注意：输入是 returns[:n_xi+t+1]，包含了最新的观测数据
            weights = compute_hmm_weights_from_sample_idx_nd_gaussian(
                sample_idx, 
                returns[:n_xi+t+1], 
                n_components, 
                posterior_p_all, 
                regime_params_all, 
                startprob
            )
            
            # 2. 构造 9 维参数矩阵 (2, 18)
            # 使用列表推导式提取各状态对应的 18 维参数 [mu1, sigma1, ..., mu9, sigma9]
            params = np.vstack([regime_params_all[i][sample_idx] for i in range(1, n_components + 1)])
            
            # 3. 调用向量化加速后的 GP 预测函数
            # 预测当前设计点 X_update (27维特征)
            mu_evaluated, _ = predicted_mean_std_joint_model_gaussian_params_updated(
                X_update, params, weights, gp, n_components
            )
            
            # 预测候选点集 X_
            mu_g, sigma_g = predicted_mean_std_joint_model_gaussian_params_updated(
                X_, params, weights, gp, n_components
            )

            # 4. 收集结果
            mu_evaluated_list.append(mu_evaluated)
            mu_g_list.append(mu_g)
            sigma_g_list.append(sigma_g)

            # 5. 存储状态权重和 18 维后验参数
            for state_idx in range(n_components):
                time_weight_list[state_idx].append(weights[0][state_idx])
                # 存入的是长度为 18 的向量
                posterior_lam_list[state_idx].append(regime_params_all[state_idx+1][sample_idx])
        
        MCMC_mean_mu_evaluated = mean_average(mu_evaluated_list) 
        MCMC_mean_mu_g = mean_average(mu_g_list) 
        MCMC_mean_sigma_g = mean_average(sigma_g_list)

        posterior_random_lam_list = [random.choice(posterior_lam_list[state]) for state in range(n_components)]

        time_avg_weight_list = [mean_average(time_weight_list[state]) for state in range(n_components)]
        time_avg_total_weight = sum(time_avg_weight_list)
        time_prob_weight_list = [time_avg_weight_list[state] / time_avg_total_weight for state in range(n_components)]

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

args = parser.parse_args()
cmd = ['-t','30260305-RS-BRO-portfolio-9Country-2regime-48data-seed40', 
       '-method', 'switching_joint',
       '-problem', 'portfolio', 
       '-smacro', '40',###40
       '-macro','41',###41 
       '-n_xi', '187',###187 (change) ######increase with dimension
       '-n_timestep', '24',###24  ####
       '-n_initial', '20', ###20
       '-n_i', '30',###30 #################
       '-n_rep', '1000',###1000 (more) ########
       '-n_candidate','100',###100  ########
       '-window', 'all',
       '-n_MCMC', '100',##100 ###############
       '-n_components', '2',###2 
       '-mu_prior_lower', '0.0', ###0.0 (change)
       '-mu_prior_upper', '50.0', ###50.0 (change)
       '-sigma_prior_lower', '0.1', ###0.1 
       '-sigma_prior_upper', '20.0', ###20.0 
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

f = portfolio_problem_9stocks() ##update high-dim
dimension_x = 9 ##update high-dim

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

