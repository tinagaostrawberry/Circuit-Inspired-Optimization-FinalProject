import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp

import sys                                                                                           
sys.path.insert(0, "/home/tinagaostrawberry/Academics/Courses/Opti/finalProjectCODE/optimization_via_circuits-main/opt_problems")
from decentr_qp import *

import ciropt as co

'''
NOTE: Taken taken from original source paper github
'''

################### SET UP PARAMS FOR PROBLEM & GENERATE RANDOM MATRICES ###################
# random seed
np.random.seed(108)

# data generation
problem_spec = {}
problem_spec['n_node'] = 20 # <--- N = 20 agents!
problem_spec['vector_size'] = 50
print(problem_spec)

# Opti poblem is: min (1/N) * SUM(j=1,N){ (x^T)*Qj*x + (pj^T)*x}
#                 s.t. N = # agents
#                      (aj^T)*x <= bj
# problem_data = {Q, p, a, b}
problem_data = data_generation(problem_spec)

# Defines graph G, mixing matrix W, Vred, and Sred
# network_data = {G, W, Vred, Sred}
network_data = graph_generation(problem_spec['n_node'])

problem_data['itr_num'] = 1000

################### CALCULATE SOLUTION USING CVX ######################
f_star, x_opt_star = min_cvx_quad_constraint(problem_spec, problem_data, True)

################### DEFINE FUNCTIONS AND PROXIMALS TO IMPLEMENT METHOD ###################
Q = problem_data['Q']
p = problem_data['p']
a = problem_data['a']
b = problem_data['b']
n_node = problem_spec['n_node']
vector_size = problem_spec['vector_size']

prox_operators = []
grad_h = []
f_plus_h = []

for jj in range(n_node):
    Qj, pj, aj, bj = Q[jj], p[jj], a[jj], b[jj]
    # Get prox operator for each node
    prox_operators += [lambda z, rho, aj=aj, bj=bj : prox_fj_quad_constraint(z, rho, aj, bj) ]
    
    # Get gradient (differential part) of obj function at each node
    # NOTE: Obj function (x^T)*Qj*x + (pj^T)*x is differentiable, so "grad_hj_quad_constraint" method
    #       simply takes derivative
    grad_h += [lambda z, Qj=Qj, pj=pj : grad_hj_quad_constraint(z, Qj, pj) ]
    
    # The full gradient objective function
    f_plus_h += [lambda x_kj, Qj=Qj, pj=pj : 1/2 * x_kj @ Qj @ x_kj.T + np.dot(pj, x_kj) ]
    
    
################### FIND PARAMS WITH PG-EXTRA WITH GRID SEARCH ###################
def pg_optimization_func(param):
    problem_data['rho'] = param
    problem_data['itr_num'] = 400
    pg_f_reldiff = co.pg_extra("pg_extra", problem_spec, problem_data, network_data, x_opt_star, f_star, prox_operators=prox_operators,
                                                                    grad_h=grad_h, f_plus_h=f_plus_h,
                                                                    printing=False)[-1]
    return pg_f_reldiff

best_rho = round(co.grid_search(pg_optimization_func, np.arange(0.01, 0.1, 0.005), start=300, end=400), 2)
print("best_rho :", best_rho)
problem_data['itr_num'] = 1000

################# RUN PG-EXTRA WITH BEST WORKING PARAM ##################################
problem_data['rho'] = best_rho
pg_op_norm, pg_err_opt_star, pg_err_opt_reldiff, pg_const_vio, pg_f_reldiff = co.pg_extra("pg_extra", problem_spec, problem_data, 
                                                                        network_data, x_opt_star, f_star, prox_operators=prox_operators,
                                                                        grad_h=grad_h, f_plus_h=f_plus_h,
                                                                        printing=True, freq=200)

################# RUN PG-EXTRA // C ####################################
params={"R":0.07, "Capacitance":0.3, "h":0.8}  
pg_c_op_norm, pg_c_err_opt_star, pg_c_err_opt_reldiff, pg_c_const_vio, pg_c_f_reldiff = co.pg_extra("pg_extra_par_c", problem_spec, problem_data, 
                                                                        network_data, x_opt_star, f_star, prox_operators=prox_operators,
                                                                        grad_h=grad_h, f_plus_h=f_plus_h,
                                                                        params=params, printing=True, freq=200)

################## RUN PG-EXTRA WITH SAME STEP SIZE AS PG-EXTRA//C ###################
problem_data['rho'] = params["R"]

pg_worse_op_norm, pg_worse_err_opt_star, pg_worse_err_opt_reldiff, pg_worse_const_vio, pg_worse_f_reldiff = co.pg_extra("pg_extra", problem_spec, problem_data, 
                                                                        network_data, x_opt_star, f_star, prox_operators=prox_operators,
                                                                        grad_h=grad_h, f_plus_h=f_plus_h,
                                                                        printing=True, freq=200)

##################### SAVE IN MAT ###################################################
T = 1000

dictMat = dict()
dictMat['losses'] = [pg_f_reldiff[:T] ,pg_worse_f_reldiff[:T], pg_c_f_reldiff[:T]]
dictMat['legend1'] = 'PG-EXTRA (Best R thru grid search, '+str(best_rho)+')'
dictMat['legend2'] = 'PG-EXTRA (Same R as PG-EXTRA//C, '+str(params["R"])+')'
dictMat['legend3'] = 'PG-EXTRA//C (R=' + str(params["R"]) + ')'
from scipy.io import savemat
savemat('PGExtraC.mat',dictMat)
