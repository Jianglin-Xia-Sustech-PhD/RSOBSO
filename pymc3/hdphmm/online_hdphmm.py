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
from switching_code.switchingBO import *
import scipy.stats as stats
from hdphmm.exponentialhdphmm import Exponential

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

def infer_n_components(state_counts):
    # Compute unique values and corresponding counts
    unique_states, counts = np.unique(state_counts, return_counts=True)
    
    # Compute probability for each state
    probabilities = counts / len(state_counts)
    
    # Find the index for the maximum probability
    max_index = np.argmax(probabilities)
    
    # Get the inferred number of states (mode)
    S_mode = unique_states[max_index]
    
    return S_mode