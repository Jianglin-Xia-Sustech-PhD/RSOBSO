
# %%
import pandas as pd
import numpy as np
import time
import multiprocessing
pd.options.display.float_format = '{:.2f}'.format
from switchingBO import *

def EGO_plug_in(seed):

    print(method)

    # real world data generation
    xi, S = switching_real_world_data(switching_true_dist, n_xi + n_timestep, random_state = seed)
    if window == "all":
        P = xi[:n_xi].reshape(-1)
    else:
        P = xi[(n_xi - int(window)):n_xi].reshape(-1)
    
    # initial experiment design
    D1 = inital_design(2*n_sample, seed, lower_bound, upper_bound) ####double 2*20
    Y_list = list()
    for i in range(D1.shape[0]):
        xx = D1[i,:dimension_x] 
        Y_list.append(f.evaluate(xx, P, n_rep, method))

    Y1 = np.array(Y_list).reshape(-1,1)

    # Fit to data using Maximum Likelihood Estimation of the parameters 
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
    mu_g,sigma_g = gp0.predict(X_,return_std= True) 
    mu_evaluated,_ = gp0.predict(X_update,return_std= True) 

    # print("check", X_.shape, mu_g.shape, sigma_g.shape, mu_evaluated.shape)
    # # Global Optimization algorithm
    for t in range(n_timestep):

        start = time.time()

        for i in range(2*n_iteration):
            
            # 1. Algorithm for x 
            D_new = EGO(X_, mu_evaluated,mu_g,sigma_g)

            # 2. Add selected samples
            D_update, Y_update = update_data(D_new,D_update,Y_update,P,n_rep, method, f)
            X_update = np.unique(D_update,axis = 0)

            # 3. Update GP model and make prediction 
            gp0 = GP_model(D_update, Y_update)
            X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
            mu_g,sigma_g = gp0.predict(X_,return_std= True) 
            mu_evaluated,_ = gp0.predict(X_update,return_std= True) 

        # 4. Update x^*
        if window == "all":
            P = xi[:(n_xi+t+1)].reshape(-1)
        else:
            P = xi[(n_xi+t+1-int(window)):(n_xi+t+1)].reshape(-1) 
        print(P.shape)
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

        
    print(f_star_list)
    print(f_hat_x_list)
    output(file_pre, minimizer_list, minimum_value_list, f_hat_x_list, TIME_RE, x_star_list, f_star_list, seed, n_sample, n_timestep, n_xi, S[n_xi:])


# %%
def switchingBO(seed):
    
    print(method)
    
    # real world data generation
    xi, S = switching_real_world_data(switching_true_dist, n_xi + n_timestep, random_state = seed)
    
    switching_model = switching_model_fit(xi[:n_xi], n_components = 2, type = "exp")
    lambdas_ = switching_model.lambdas_
    distribution1 = {'dist':"exp", "rate": lambdas_[0]}
    distribution2 = {'dist':"exp", "rate": lambdas_[1]}
    swtiching_distributions_estimated = [distribution1, distribution2]

    predicted_proba = switching_model.predict_proba(xi[:n_xi])[-1,:]
    weights = np.matmul(predicted_proba, switching_model.transmat_)
               
    # initial experiment design
    D1 = inital_design(n_sample, seed, lower_bound, upper_bound)

    Y_list = list()
    for i in range(D1.shape[0]):
        xx = D1[i,:dimension_x] 
        P = swtiching_distributions_estimated[0]
        Y_list.append(f.evaluate(xx, P, n_rep, method))
    Y1 = np.array(Y_list).reshape(-1,1)

    Y_list2 = list()
    for i in range(D1.shape[0]):
        xx = D1[i,:dimension_x] 
        P = swtiching_distributions_estimated[1]
        Y_list2.append(f.evaluate(xx, P, n_rep, method))
    Y2 = np.array(Y_list2).reshape(-1,1)

    # Fit to data using Maximum Likelihood Estimation of the parameters 
    gp1 = GP_model(D1, Y1) 
    gp2 = GP_model(D1, Y2) 
    gps = [gp1, gp2]

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    f_hat_x_list = list()
    x_star_list = list()
    f_star_list = list()
    TIME_RE = list()

    D_update = D1
    X_update = np.unique(D_update,axis = 0)
    Y_update1 = Y1  
    Y_update2 = Y2

    X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
    mu_g, sigma_g = predicted_mean_std(X_, weights, gps)
    mu_evaluated, _ =  predicted_mean_std(X_update, weights, gps)

    print("check", X_.shape, mu_g.shape, sigma_g.shape, mu_evaluated.shape)

    for t in range(n_timestep):

        print("timestep", t)
        start = time.time()
        
        for i in range(n_iteration):
            print("iteration",i)

            # 1. Algorithm for x 
            D_new = EGO(X_, mu_evaluated, mu_g,sigma_g)

            # 2. Add selected samples
            D_update, Y_update1, Y_update2 = update_data_switching(D_new,D_update,Y_update1, Y_update2, swtiching_distributions_estimated ,n_rep, method, f)
            X_update = np.unique(D_update,axis = 0)

            # 3. Update GP model and make prediction 
            gp1 = GP_model(D_update, Y_update1) 
            gp2 = GP_model(D_update, Y_update2) 
            gps = [gp1, gp2]

            X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
            mu_g, sigma_g = predicted_mean_std(X_, weights, gps)
            mu_evaluated, _ =  predicted_mean_std(X_update, weights, gps)

        # 4. Update x^* 
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

        predicted_proba = switching_model.predict_proba(xi[:(n_xi+t+1)])[-1,:]
        weights = np.matmul(predicted_proba, switching_model.transmat_)
        
    print(f_star_list)
    print(f_hat_x_list)
    output(file_pre, minimizer_list, minimum_value_list, f_hat_x_list, TIME_RE, x_star_list, f_star_list, seed, n_sample, n_timestep, n_xi, S[n_xi:])


def switchingBO2(seed): 
    
    print(method)
    
    # real world data generation
    xi, S = switching_real_world_data(switching_true_dist, n_xi + n_timestep, random_state = seed)
    
    switching_model = switching_model_fit(xi[:n_xi], n_components = 2, type = "exp")
    lambdas_ = switching_model.lambdas_
    distribution1 = {'dist':"exp", "rate": lambdas_[0]}
    distribution2 = {'dist':"exp", "rate": lambdas_[1]}
    swtiching_distributions_estimated = [distribution1, distribution2]
    predicted_proba = switching_model.predict_proba(xi[:n_xi])[-1,:]
    weights = np.matmul(predicted_proba, switching_model.transmat_)
               
    # initial experiment design
    D1 = inital_design(n_sample, seed, lower_bound, upper_bound)

    Y_list = list()
    for i in range(D1.shape[0]):
        xx = D1[i,:dimension_x] 
        P = swtiching_distributions_estimated[0]
        Y_list.append(f.evaluate(xx, P, n_rep, method))
    Y1 = np.array(Y_list).reshape(-1,1)

    Y_list2 = list()
    for i in range(D1.shape[0]):
        xx = D1[i,:dimension_x] 
        P = swtiching_distributions_estimated[1]
        Y_list2.append(f.evaluate(xx, P, n_rep, method))
    Y2 = np.array(Y_list2).reshape(-1,1)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    constant_lambda1 = np.full((D1.shape[0], 1), 1.0/lambdas_[0])
    D1_lambda1 = np.concatenate((D1, constant_lambda1), axis=1)
    constant_lambda2 = np.full((D1.shape[0], 1), 1.0/lambdas_[1])
    D1_lambda2 = np.concatenate((D1, constant_lambda2), axis=1)

    D_update = np.concatenate((D1_lambda1, D1_lambda2),axis = 0)
    Y_update = np.concatenate((Y1,Y2),axis=0)
    X_update = np.unique(D_update[:,:dimension_x],axis = 0)

    gp = GP_model(D_update, Y_update) 

    # Prepare for iteration 
    minimizer_list = list()
    minimum_value_list = list()
    f_hat_x_list = list()
    x_star_list = list()
    f_star_list = list()
    TIME_RE = list()

    X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
    mu_g, sigma_g = predicted_mean_std_joint_model(X_, lambdas_,weights, gp)
    mu_evaluated, _ = predicted_mean_std_joint_model(X_update, lambdas_, weights,gp)

    print("check", X_.shape, mu_g.shape, sigma_g.shape, mu_evaluated.shape)

    for t in range(n_timestep):

        print("timestep", t)
        start = time.time()
        
        for i in range(n_iteration):
            print("iteration",i)

            # 1. Algorithm for x 
            D_new = EGO(X_, mu_evaluated, mu_g,sigma_g)

            # 2. Add selected samples
            D_update, Y_update = update_data_switching_joint(D_new,D_update,Y_update, swtiching_distributions_estimated ,n_rep, method, f)
            X_update = np.unique(D_update[:,:dimension_x],axis = 0)
            # 3. Update GP model and make prediction 
            gp = GP_model(D_update, Y_update) 
            print(D_update)
            X_ = inital_design(n_candidate, None, lower_bound, upper_bound)
            mu_g, sigma_g = predicted_mean_std_joint_model(X_, lambdas_,weights, gp)
            mu_evaluated, _ = predicted_mean_std_joint_model(X_update, lambdas_, weights,gp)


        # 4. Update x^* 
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

        switching_model = switching_model_fit(xi[:n_xi+t+1], n_components = 2, type = "exp")
        lambdas_ = switching_model.lambdas_
        distribution1 = {'dist':"exp", "rate": lambdas_[0]}
        distribution2 = {'dist':"exp", "rate": lambdas_[1]}
        swtiching_distributions_estimated = [distribution1, distribution2]
        print(swtiching_distributions_estimated)
        predicted_proba = switching_model.predict_proba(xi[:(n_xi+t+1)])[-1,:]
        weights = np.matmul(predicted_proba, switching_model.transmat_)
        
    print(f_star_list)
    print(f_hat_x_list)
    output(file_pre, minimizer_list, minimum_value_list, f_hat_x_list, TIME_RE, x_star_list, f_star_list, seed, n_sample, n_timestep, n_xi, S[n_xi:])

# %%
def Experiment(seed):
    
    if method in ["exp", 'hist']:
        EGO_plug_in(seed)

    elif method == 'switching':
        switchingBO(seed) 

    elif method == 'switching_joint':
        switchingBO2(seed) 

# %%
import argparse
parser = argparse.ArgumentParser(description='NBRO-algo')
#parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

parser.add_argument('-t','--time', type=str, help = 'Date of the experiments, e.g. 20201109',default = '20240409')
parser.add_argument('-method','--method', type=str, help = 'switching/hist/exp/lognorm',default = "switching_joint")
parser.add_argument('-problem','--problem', type=str, help = 'Function',default = 'inventory')

parser.add_argument('-smacro','--start_num',type=int, help = 'Start number of macroreplication', default = 24)
parser.add_argument('-macro','--repet_num', type=int, help = 'Number of macroreplication',default = 50)

parser.add_argument('-n_xi','--n_xi', type=int, help = 'Number of observations of the random variable',default = 100)
parser.add_argument('-n_sample','--n_sample',type=int, help = 'number of initial samples', default = 20)
parser.add_argument('-n_i','--n_iteration', type=int, help = 'Number of iteration',default = 15)
parser.add_argument('-n_timestep','--n_timestep', type=int, help = 'Number of timestep',default = 25)

parser.add_argument('-n_rep','--n_replication', type=int, help = 'Number of replications at each point',default = 2)
parser.add_argument('-n_candidate','--n_candidate', type=int, help = 'Number of candidate points for each iteration',default = 100)
parser.add_argument('-core','--n_cores', type=int, help = 'Number of Cores',default = 12)
parser.add_argument('-window','--window', type=str, help = 'Number of moving window',default = "all")

args = parser.parse_args()
cmd = ['-t','20241205_MLE_print_D_update', 
       '-method', 'switching_joint',
       '-problem', 'inventory', 
       '-smacro', '100',
       '-macro','101',
       '-n_xi', '100', #100
       '-n_timestep', '2',#5
       '-n_sample', '20', #20
       '-n_i', '5', #3
       '-n_rep', '10', #10
       '-n_candidate','100', #100
       '-core', '4',
       '-window', 'all'
       ]
args = parser.parse_args(cmd)
print(args)

time_run = args.time 
method = args.method 
problem = args.problem 

n_xi = args.n_xi
n_sample = args.n_sample # number of initial samples
n_iteration = args.n_iteration # Iteration
n_timestep = args.n_timestep
n_rep = args.n_replication  #number of replication on each point
n_candidate = args.n_candidate # Each iteration, number of the candidate points are regenerated+
window = args.window

switching_true_dist = {
    'startprob': np.array([0, 1]), 
    "transmat": np.array([[0.7, 0.3],[0.2, 0.8]]),
    "lambdas": np.array([[1],[1/20]]),
    "emission": "exp",
    "n_components": 2
    }
swtiching_distributions = [
    {'dist':switching_true_dist.get("emission"),
    "rate":switching_true_dist.get("lambdas")[0]},
    {'dist':switching_true_dist.get("emission"),
    "rate":switching_true_dist.get("lambdas")[1]}]
f1 = inventory_problem(swtiching_distributions[0])
f2 = inventory_problem(swtiching_distributions[1])

f1.x_star = np.array([1,70])
f2.x_star = np.array([63.8,127.0])
f1.f_star = 38.0
f2.f_star = 147.0

# f1.lb = [1,35]
# f1.ub = [34,100]
# f2.lb = [1,35]
# f2.ub = [34,100]
# f1.x_star = np.array([1.6,35])
# f2.x_star = np.array([29.1,73.8])
# f1.f_star = 22.3
# f2.f_star = 83.9

# f1.estimate_minimizer_minimum()
# f2.estimate_minimizer_minimum()

f_true = [f1,f2]
f = f1
true_minimizer = [f1.x_star, f2.x_star]
true_minimium = [f1.f_star, f2.f_star]
print(true_minimizer)
print(true_minimium)

dimension_x = f.dimension_x
lower_bound = f.lb 
upper_bound = f.ub
print(lower_bound, upper_bound)

if "switching" in method:
    file_pre = '_'.join(["outputs/bo",time_run, problem, method])
else:
    file_pre = '_'.join(["outputs/bo",time_run, problem, method, str(window)])

print(file_pre)

if __name__ == '__main__':

    start_num = args.start_num  
    repet_num = args.repet_num 

    import multiprocessing
    starttime = time.time()
    pool = multiprocessing.Pool(4)
    pool.map(Experiment, range(start_num,repet_num))
    pool.close()
    print('That took {} secondas'.format(time.time() - starttime))

