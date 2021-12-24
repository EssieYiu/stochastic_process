from os import name
import numpy as np
import copy
import sys
import getopt
import time
import matplotlib
from matplotlib import pyplot as plt
from numpy.lib.function_base import diff, vectorize

def AverageCostProblemValueIteration(c,K,n,p,iterations,initial_policy=[]):
    h = np.array((n+1) * [0.0])
    if initial_policy == []:
        action = np.array((n+1) * [False])
        action[n] = True
    else:
        action = np.array(initial_policy)
        action[n] = True
    index = np.array(np.arange(0,n+1))
    iter_converge = 0
    for iter in range(iterations):
        h_prev = copy.deepcopy(h)
        lambd = K + (1-p)*h[0] + p*h[1]
        #vectorization
        h_shift = np.roll(h,-1)
        manufacture_cost = K + (1-p)*h[0] + p*h[1]
        keep_cost = c*index + (1-p)*h + p*h_shift
        action = keep_cost >= manufacture_cost
        h = keep_cost * (~action) + manufacture_cost * action - lambd
        h[n] = 0.0
        action[n] = True
        diff_sum = sum(abs(h_prev - h))
        if(diff_sum < 0.00001):
            iter_converge = iter
            break
    if iter_converge == 0:
        iter_converge = iterations
    return (action,h,lambd,iter_converge)

def AverageCostProblemPolicyIteration(c,K,n,p,iterations,initial_policy=[]):
    h =np.array((n+1) * [0.0]) 
    h_shift = np.roll(h,-1)
    if initial_policy==[]:
        action = np.array((n+1) *[True])
    else:
        action = np.array(initial_policy)
    index = np.array(np.arange(0,n+1))
    iter_converge = 0
    for iter in range(iterations):
        prev_action = copy.deepcopy(action)
        for iter2 in range(iterations):
            h_prev = copy.deepcopy(h)
            lambd = K + (1-p)*h[0] + p*h[1]
            #用来计算h[i+1]
            #policy evaluation
            evaluate_keep_cost = c*index + (1-p)*h + p*h_shift
            evaluate_manufature_cost = K + p*h[1] + (1-p)*h[0]
            h = evaluate_keep_cost*(~action) + evaluate_manufature_cost*action - lambd
            h[n] = 0.0
            h_shift = np.roll(h,-1)
            if sum(abs(h_prev - h)) < 0.00001:
                break
        #policy improvement
        manufacture_cost = K + p*h[1] + (1-p)*h[0]
        keep_cost = c*index + (1-p)*h + p*h_shift
        action = keep_cost >= manufacture_cost
        action[n] = True
        if np.array_equal(action, prev_action):
            iter_converge = iter
            break
    if iter_converge == 0:
        iter_converge = iterations
    return (action,h,lambd,iter_converge)

def DiscountedProblemPolicyIteration(c,K,n,p,alpha,iterations,initial_policy=[]):
    if len(initial_policy)==0:
        action = np.array((n+1)*[True])
    else:
        action = np.array(initial_policy)
    J = np.array((n+1)*[0])
    J_shift = np.roll(J,-1)
    index = np.array(np.arange(0,n+1))
    iter_converge = 0
    for iter in range(iterations):
        prev_action = copy.deepcopy(action)
        #policy evaluation
        for iter2 in range(iterations):
            prev_J = copy.deepcopy(J)
            evaluate_manufacture_cost = K + alpha * p * J[1] + alpha * (1-p) * J[0]
            evaluate_keep_cost = c*index + alpha*p*J_shift + alpha*(1-p)*J
            J = evaluate_keep_cost*(~action) + evaluate_manufacture_cost*action
            J_shift = np.roll(J,-1)
            if sum(abs(prev_J-J)) < 0.00001:
                break
            
        #policy improvement
        keep_cost = c*index + alpha*p*J_shift + alpha*(1-p)*J
        manufacture_cost = K + alpha*p*J[1] + alpha*(1-p)*J[0]
        action = keep_cost >= manufacture_cost
        action[n] = 1
        if np.array_equal(action,prev_action):
            iter_converge = iter
            break
    if iter_converge == 0:
        iter_converge = iterations
    return(action,J,iter_converge)

def DiscountedProblemValueIteration(c,K,n,p,alpha,iterations,initial_policy=[]):
    J = np.array((n+1) * [K])
    if initial_policy == []:
        action = np.array((n+1) * [False])
        action[n] = True
    else:
        action = np.array(initial_policy)
    iter_converge = 0
    index = np.array(np.arange(0,n+1))
    for iter in range(iterations):
        J_prev = copy.deepcopy(J)
        manufature_cost = K + alpha*(1-p)*J[0] + alpha*p*J[1]
        J_shift = np.roll(J,-1)
        keep_cost = index*c + alpha*(1-p)*J + alpha*p*J_shift
        action = keep_cost >= manufature_cost
        J = keep_cost*(~action) + manufature_cost*action
        J[n] = manufature_cost
        diff_sum = sum(abs(J_prev-J))
        if(diff_sum < 0.00001):
            iter_converge = iter
            break
    if(iter_converge == 0):
        iter_converge = iterations
    return (action,J,iter_converge)


def ParameterAndPerformanceAnalysisDiscountedProblem(default_c,default_K,default_n,default_p,default_alpha,num_iter=1000):
    #param = c
    c_list = np.arange(1,50,0.5)
    convergence_list_p = []
    convergence_list_v = []
    for c in c_list:
        action_v, J_v, iter_converge_v = DiscountedProblemValueIteration(c,default_K,default_n,default_p,default_alpha,num_iter)
        action_p, J_p, iter_converge_p = DiscountedProblemPolicyIteration(c,default_K,default_n,default_p,default_alpha,num_iter)
        convergence_list_p.append(iter_converge_p)
        convergence_list_v.append(iter_converge_v)
    print(convergence_list_p)
    print(convergence_list_v)
    plt.title("Convergence Speed of Parameter 'c' in Discounted Problem")
    plt.xlabel('c')
    plt.ylabel('number of iterations')
    plt.plot(c_list,convergence_list_p,'r--',label='policy iteration')
    plt.plot(c_list,convergence_list_v,'b',label='value iteration')
    plt.legend()
    plt.show()

    #param = K
    K_list = np.arange(1,100,1)
    convergence_list_v = []
    convergence_list_p = []
    for K in K_list:
        action_v, J_v, iter_converge_v = DiscountedProblemValueIteration(default_c,K,default_n,default_p,default_alpha,num_iter)
        action_p, J_p, iter_converge_p = DiscountedProblemPolicyIteration(default_c,K,default_n,default_p,default_alpha,num_iter)
        convergence_list_v.append(iter_converge_v)
        convergence_list_p.append(iter_converge_p)

    print(convergence_list_v)
    print(convergence_list_p)
    plt.title("Convergence Speed of Parameter 'K' in Discounted Problem")
    plt.xlabel('K')
    plt.ylabel('number of iterations')
    plt.plot(K_list,convergence_list_p,'r--',label='policy iteration')
    plt.plot(K_list,convergence_list_v,'b',label='value iteration')
    plt.legend()
    plt.show()

    #param = n 
    n_list = np.arange(2,100,1)
    convergence_list_v = []
    convergence_list_p = []
    for n in n_list:
        action_v, J_v, iter_converge_v = DiscountedProblemValueIteration(default_c,default_K,n,default_p,default_alpha,num_iter)
        action_p, J_p, iter_converge_p = DiscountedProblemPolicyIteration(default_c,default_K,n,default_p,default_alpha,num_iter)
        convergence_list_v.append(iter_converge_v)
        convergence_list_p.append(iter_converge_p)
    print(convergence_list_v)
    print(convergence_list_p)
    plt.title("Convergence Speed of Parameter 'n' in Discounted Problem")
    plt.xlabel('n')
    plt.ylabel('number of iterations')
    plt.plot(n_list,convergence_list_p,'r--',label='policy iteration')
    plt.plot(n_list,convergence_list_v,'b',label='value iteration')
    plt.legend()
    plt.show()

    #param=p
    p_list = np.arange(0,1,0.01)
    convergence_list_v = []
    convergence_list_p = []
    for p in p_list:
        action_v, J_v, iter_converge_v = DiscountedProblemValueIteration(default_c,default_K,default_n,p,default_alpha,num_iter)
        action_p, J_p, iter_converge_p = DiscountedProblemPolicyIteration(default_c,default_K,default_n,p,default_alpha,num_iter)
        convergence_list_v.append(iter_converge_v)
        convergence_list_p.append(iter_converge_p)
    print(convergence_list_v)
    print(convergence_list_p)
    plt.title("Convergence Speed of Parameter 'p' in Discounted Problem")
    plt.xlabel('p')
    plt.ylabel('number of iterations')
    plt.plot(p_list,convergence_list_p,'r--',label='policy iteration')
    plt.plot(p_list,convergence_list_v,'b',label='value iteration')
    plt.legend()
    plt.show()

    #param=alpha
    alpha_list = np.arange(0,1,0.01)
    convergence_list_v = []
    convergence_list_p = []
    for alpha in alpha_list:
        action_v, J_v, iter_converge_v = DiscountedProblemValueIteration(default_c,default_K,default_n,default_p,alpha,num_iter)
        action_p, J_p, iter_converge_p = DiscountedProblemPolicyIteration(default_c,default_K,default_n,default_p,alpha,num_iter)
        convergence_list_v.append(iter_converge_v)
        convergence_list_p.append(iter_converge_p)
    print(convergence_list_v)
    print(convergence_list_p)
    plt.title("Convergence Speed of Parameter 'alpha' in Discounted Problem")
    plt.xlabel('alpha')
    plt.ylabel('number of iterations')
    plt.plot(alpha_list,convergence_list_p,'r--',label='policy iteration')
    plt.plot(alpha_list,convergence_list_v,'b',label='value iteration')
    plt.legend()
    plt.show()

def ParameterAndPerformanceAnalysisAverageCostProblem(default_c,default_K,default_n,default_p,num_iter=1000):
    #param = c
    c_list = np.arange(1,50,0.5)
    convergence_list_p = []
    convergence_list_v = []
    for c in c_list:
        action_v, h_v, lambd_v, iter_converge_v = AverageCostProblemValueIteration(c,default_K,default_n,default_p,num_iter)
        action_p, h_p, lambd_p, iter_converge_p = AverageCostProblemPolicyIteration(c,default_K,default_n,default_p,num_iter)
        convergence_list_p.append(iter_converge_p)
        convergence_list_v.append(iter_converge_v)
    print(convergence_list_p)
    print(convergence_list_v)
    plt.title("Convergence Speed of Parameter 'c' in Average Cost Problem")
    plt.xlabel('c')
    plt.ylabel('number of iterations')
    plt.plot(c_list,convergence_list_p,'r--',label='policy iteration')
    plt.plot(c_list,convergence_list_v,'b',label='value iteration')
    plt.legend()
    plt.show()

    #param = K
    K_list = np.arange(1,100,1)
    convergence_list_v = []
    convergence_list_p = []
    for K in K_list:
        action_v, h_v, lambd_v, iter_converge_v = AverageCostProblemValueIteration(default_c,K,default_n,default_p,num_iter)
        action_p, h_p, lambd_p, iter_converge_p = AverageCostProblemPolicyIteration(default_c,K,default_n,default_p,num_iter)
        convergence_list_v.append(iter_converge_v)
        convergence_list_p.append(iter_converge_p)

    print(convergence_list_v)
    print(convergence_list_p)
    plt.title("Convergence Speed of Parameter 'K' in Average Cost Problem")
    plt.xlabel('K')
    plt.ylabel('number of iterations')
    plt.plot(K_list,convergence_list_p,'r--',label='policy iteration')
    plt.plot(K_list,convergence_list_v,'b',label='value iteration')
    plt.legend()
    plt.show()

    #param = n 
    n_list = np.arange(2,100,1)
    convergence_list_v = []
    convergence_list_p = []
    for n in n_list:
        action_v, h_v, lambd_v, iter_converge_v = AverageCostProblemValueIteration(default_c,default_K,n,default_p,num_iter)
        action_p, h_p, lambd_p, iter_converge_p = AverageCostProblemPolicyIteration(default_c,default_K,n,default_p,num_iter)
        convergence_list_v.append(iter_converge_v)
        convergence_list_p.append(iter_converge_p)
    print(convergence_list_v)
    print(convergence_list_p)
    plt.title("Convergence Speed of Parameter 'n' in Average Cost Problem")
    plt.xlabel('n')
    plt.ylabel('number of iterations')
    plt.plot(n_list,convergence_list_p,'r--',label='policy iteration')
    plt.plot(n_list,convergence_list_v,'b',label='value iteration')
    plt.legend()
    plt.show()

    #param=p
    p_list = np.arange(0,1,0.01)
    convergence_list_v = []
    convergence_list_p = []
    for p in p_list:
        action_v, h_v, lambd_v, iter_converge_v = AverageCostProblemValueIteration(default_c,default_K,default_n,p,num_iter)
        action_p, h_p, lambd_p, iter_converge_p = AverageCostProblemPolicyIteration(default_c,default_K,default_n,p,num_iter)
        convergence_list_v.append(iter_converge_v)
        convergence_list_p.append(iter_converge_p)
    print(convergence_list_v)
    print(convergence_list_p)
    plt.title("Convergence Speed of Parameter 'p' in Average Cost Problem")
    plt.xlabel('p')
    plt.ylabel('number of iterations')
    plt.plot(p_list,convergence_list_p,'r--',label='policy iteration')
    plt.plot(p_list,convergence_list_v,'b',label='value iteration')
    plt.legend()
    plt.show()


def Exercise7():
    c = 1
    K = 5
    n = 10
    p = 0.5
    alpha = 0.9
    num_iter = 500
    initial_policy = (n+1)*[True]

    action, J, iter_converge = DiscountedProblemPolicyIteration(c,K,n,p,alpha,num_iter,initial_policy)
    fmt = "{:5}\t{:^5}\t{:^5}"
    print("Discounted Problem: Policy Iteration")
    print("iter_converge=",iter_converge)
    print(fmt.format("State","Action","Cost"))
    print("------------------------------------")
    for i in range(n):
        print(fmt.format(str(i),str(action[i]),str(J[i])))

    action, J, iter_converge = DiscountedProblemValueIteration(c,K,n,p,alpha,num_iter,initial_policy)
    fmt = "{:5}\t{:^5}\t{:^5}"
    print("Discounted Problem: Value Iteration")
    print("iter_converge=",iter_converge)
    print(fmt.format("State","Action","Cost"))
    print("------------------------------------")
    for i in range(n):
        print(fmt.format(str(i),str(action[i]),str(J[i])))

    
def StationaryAnalysis(c,K,n,p):
    min_f = 99999
    min_stationary_vector = []
    threshold = 0
    for m in range(1,n+1):
        #print("m=",m)
        P_matrix = np.array((m+1)*[(m+1)*[0.0]])
        for i in range(m):
            P_matrix[i][i] = 1-p
            P_matrix[i][i+1] = p
        P_matrix[m][0] = 1-p
        P_matrix[m][1] = p


        #print(P_matrix)
        P_matrix_n = copy.deepcopy(P_matrix)
        for i in range(100):
            P_matrix_n = np.dot(P_matrix_n,P_matrix)
        #print("P=",P_matrix_n)

        index = c * np.array(np.arange(m+1))
        index[m] = K
        #print("index=",index)
        stationary_vector = P_matrix_n[0]
        #print("stationary_vector",stationary_vector)
        f_value = np.dot(stationary_vector,index.T)
        #print("f_value:",f_value)
        if f_value < min_f:
            min_f = f_value
            min_stationary_vector = stationary_vector
            threshold = m
    return (threshold,min_f,min_stationary_vector)
    

def Compare(c,K,n,p):
    alpha_list = np.arange(0,1,0.01)
    num_iter = 500
    initial_policy = (n+1)*[True]

    threshold_list = []

    action, h, lambd, iter_converge = AverageCostProblemValueIteration(c,K,n,p,num_iter,initial_policy)
    fmt = "{:5}\t{:^5}\t{:^5}"
    print("Average Cost Problem: Value Iteration")
    print("lambda = ", lambd)
    avg_threshold = np.argwhere(action == True)[0]
    print("threshold",avg_threshold)
    print(fmt.format("State","Action","Cost"))
    print("------------------------------------")
    for i in range(n):
        print(fmt.format(str(i),str(action[i]),str(h[i])))
    
    action, h, lambd, iter_converge = AverageCostProblemPolicyIteration(c,K,n,p,num_iter,initial_policy)
    fmt = "{:5}\t{:^5}\t{:^5}"
    print("Average Cost Problem: Policy Iteration")
    print("lambda = ", lambd)
    print(fmt.format("State","Action","Cost"))
    print("------------------------------------")
    for i in range(n):
        print(fmt.format(str(i),str(action[i]),str(h[i])))

    for alpha in alpha_list:
        action, J, iter_converge = DiscountedProblemValueIteration(c,K,n,p,alpha,num_iter,initial_policy)
        fmt = "{:5}\t{:^5}\t{:^5}"
        print("Discounted Cost Problem: Value Iteration")
        print("lambda = ", lambd)
        threshold = np.argwhere(action == True)[0]
        threshold_list.append(threshold)
        print("threshold",threshold)
        print(fmt.format("State","Action","Cost"))
        print("------------------------------------")
        for i in range(n):
            print(fmt.format(str(i),str(action[i]),str(J[i])))

    #When >= m, process all orders
    avg_list = len(alpha_list)*[avg_threshold]
    plt.title("Compare optimal policies.")
    plt.xlabel('alpha')
    plt.ylabel('threshold m')
    plt.plot(alpha_list,avg_list,'b',label='discounted problem')
    plt.plot(alpha_list,threshold_list,'r--',label="average cost problem")
    plt.legend()
    plt.show()



if __name__ =='__main__':
    default_c = 1
    default_K = 5
    default_n = 10
    default_p = 0.5
    default_alpha = 0.9
    num_iter = 1000
    
    #---------1. Run the code with different parameter i--------------------------------------

    ParameterAndPerformanceAnalysisDiscountedProblem(default_c,default_K,default_n,default_p,default_alpha,num_iter)
    
    ParameterAndPerformanceAnalysisAverageCostProblem(default_c,default_K,default_n,default_p,num_iter)

    #---------3. For discounted problem, consider exercise 7.8---------

    Exercise7()

    #---------3. For average cost problem, write code to find the threshold by stationary analysis---------
    
    threshold, min_f, min_stationary_vector = StationaryAnalysis(default_c,default_K,default_n,default_p)
    print("Stationary Analysis.")
    print("threshold = ",threshold)
    print("min_f = ",min_f)
    print("stationary_vector=",min_stationary_vector)

    #---------4. Compare the optimal policies of the discounted problem with different values of alpha and the average cost problem---------
    Compare(1,5,10,0.5)
