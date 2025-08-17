from __future__ import division
from builtins import range
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
import pandas as pd
import matplotlib
import os
matplotlib.rcParams['font.size'] = 8
import pyhsmm
from pyhsmm.util.text import progprint_xrange
import matplotlib.pyplot as plt
from hmmlearn import hmm
from switching_code.exponentialhmm import ExponentialHMM
import scipy.stats as stats
from scipy.stats import poisson
from hmmlearn.hmm import PoissonHMM
from exponentialhdphmm import Exponential

def switching_real_world_data(true_dist, timestep, random_state = 0):

    if true_dist.get("emission") == "exp":
        # Build an HMM instance and set parameters
        model = ExponentialHMM(n_components=true_dist.get("n_components"), random_state = random_state)

    elif true_dist.get("emission") == "poisson":
        
        model = PoissonHMM(n_components=true_dist.get("n_components"), random_state = random_state)
    
    else:
        raise NotImplementedError("Not implemented")

    # Instead of fitting it from the data, we directly set the estimated
    # parameters, the means and covariance of the components
    model.startprob_ = true_dist.get("startprob")
    model.transmat_ = true_dist.get("transmat")
    model.lambdas_ = true_dist.get("lambdas")

    # Generate samples
    X, Z = model.sample(timestep)

    return X, Z

def select_state_count(beta_posterior, num_data):
    threshold = 1 / np.sqrt(num_data) ### threshold designed
    num_states = np.sum(beta_posterior >= threshold)
    return max(num_states, 1)  # 确保至少返回1个状态

def infer_state_count(initial_data, N_max=100, burn_in=10000, total_samples=10000):
    """
    运行 HDP-HMM 进行状态数量推断
    
    参数：
    - initial_data: 初始观测数据
    - xi: 所有观测数据（用于计算 threshold）
    - N_max: 设定的最大状态数量
    - burn_in: burn-in 迭代次数
    - total_samples: 采样阶段的迭代次数

    返回：
    - state_counts: 每次采样的状态数量列表
    """
    
    state_counts = []  # 存储每次采样的状态数量
    
    # 定义观测分布
    obs_hypparams = {'alpha_0': 1.0, 'beta_0': 1.0}
    obs_distns = [Exponential(**obs_hypparams) for _ in range(N_max)]
    
    # 初始化 HDP-HMM
    model = pyhsmm.models.WeakLimitHDPHMM(
        alpha_a_0=5., alpha_b_0=1.,
        gamma_a_0=5., gamma_b_0=1.,
        init_state_concentration=5.,
        obs_distns=obs_distns
    )
    
    # 加载数据
    model.add_data(initial_data)
    
    # 1. Burn-in 阶段
    for _ in progprint_xrange(burn_in):
        model.resample_model()
    
    # 2. 正式采样阶段
    for _ in progprint_xrange(total_samples):
        model.resample_model()
        
        # 获取 β（混合权重）
        beta_values = model.trans_distn.beta
        
        # 计算状态数量
        inferred_states = select_state_count(beta_values, len(initial_data))
        
        # 记录推测的状态数量
        state_counts.append(inferred_states)
    
    return state_counts

seed = 42
true_lambdas = np.array([[1/20],[1]])

# read real regime data and generate corresponding random xi here
df0 = pd.read_csv('/root/rs_bro92/RSDR_CVaR_dataset/BB_state.csv')
df0['state_name'] = df0['state'].map({1: 'bull', 2: 'bear'})
df0['state'] = df0['state'].replace({1: 0, 2: 1})
# given + upcoming: 200401-200712, 200801-200912 (48+24=72)
df_used = df0[(df0['Date'] >= 20040101) & (df0['Date'] < 20100101)].reset_index(drop=True)
S = df_used['state'].to_numpy() # real state data

# random seed setting to generate xi
#rng = np.random.default_rng(seed=seed)
rng = np.random.RandomState(seed=seed) #old version
xi = np.empty(len(S))
# regime 0: rate 1/20, regime 1: rate 1 
for i, s in enumerate(S):
    scale = 20 if s == 0 else 1 
    xi[i] = stats.expon(scale=scale).rvs(random_state=rng)

n_xi = 48
N_max = 10 ##upper bound of number of states
num_stages = 24 # additional time stages
S_mode_list = []
S_max_list=[]
N_max_list=[]

for t in range(num_stages):

    print('timestage', t)

    N_max_list.append(N_max)

    online_data = xi[:n_xi+t]

    state_counts = infer_state_count(online_data, N_max = N_max, burn_in = 100, total_samples = 100)

    # 获取state数量的最大取值
    S_max = np.max(state_counts)

    # 计算唯一值及其对应的计数
    unique_states, counts = np.unique(state_counts, return_counts=True)

    # 计算概率
    probabilities = counts / len(state_counts)

    # 找到最大概率对应的索引
    max_index = np.argmax(probabilities)

    # 获取对应的 state 值和最大概率
    S_mode = unique_states[max_index]

    print('current N_max', N_max)
    print('current inferred number of state', S_mode)  
    print('current S_max', S_max)  

    N_max = S_max + 1 

    S_mode_list.append(S_mode)
    S_max_list.append(S_max)

# 合并三个列表
combined_data = np.column_stack((N_max_list, S_max_list, S_mode_list))

# 保存到 CSV，添加列名
np.savetxt("output_3_state_100data_new_prior_Jun14_seed42.csv", combined_data, delimiter=",", fmt="%d", header="N_max,S_max,S_mode", comments="")