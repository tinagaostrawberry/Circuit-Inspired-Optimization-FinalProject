import numpy as np
import ciropt as co
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.special import huber as sp_huber

'''
NOTE: Outline taken from "examples/hello_world.ipynb" in paper's repo
'''

np.random.seed(1)

## PART I
# Set up problem
problem = co.CircuitOpt()

# Define obj fun type
# mu = u-strongly convex (0 -> not strongly convex)
# M = M-smoothness (Inf -> Non-differentiable)
f = co.def_function(problem, mu=0, L_smooth=np.inf) # mu = 0.5

# Define optimal points
x_star, y_star, f_star = f.stationary_point(
    return_gradient_and_function_value=True
)

# Define values for RLC components
R = 1
C = 10
# Define discretization params for Rrunge-Kutta
# x(k+1/2) = x(k) + alpha*h*F(x(k))
# x(k+1)   = x(k) + beta*h*F(x(k))  + (1-beta)*h*F(x(k+1/2))
alpha = 0
beta = 1
h = problem.h
eta = problem.eta

# Define one step transition in discretized V-I relations
# 1st step
z_1 = problem.set_initial_point()
e2_1 = problem.set_initial_point()
x_1 = co.proximal_step(z_1,f,R/2)[0]
y_1 = (2/R)*(z_1-x_1)
# 2nd step
e2_2 = e2_1 - h/(2*R*C)*(R*y_1+3*e2_1)
z_2 = z_1 - h/(4*R*C)*(5*R*y_1+3*e2_1)
x_2 = co.proximal_step(z_2,f,R/2)[0]
y_2 = (2/R)*(z_2-x_2)

# Define dissipative terms:
# E2 - E1 + eta*(x(1)-x(*),y(1)-y(*))
# E1
v_C1_1 = e2_1/2 - z_1
v_C2_1 = e2_1
E_1 = (C/2)*(v_C1_1+x_star)**2 + (C/2)*(v_C2_1)**2
# E2
v_C1_2 = e2_2/2 - z_2
v_C2_2 = e2_2
E_2 = (C/2)*(v_C1_2+x_star)**2 + (C/2)*(v_C2_2)**2
# Final
E = E_2-E_1+eta*(x_1-x_star)*(y_1-y_star)

# Set up final problem
problem.set_performance_metric(E)
params = problem.solve()[:1]

# Show
for k in params[0].keys():
    print(k + ": " + str(params[0][k]))
    
    
## PART II
# Set up obj fun vals
n = 100
m = 30
A = np.random.randn(m, n)
b = np.random.randn(m)
c = np.random.randn(n) 
lambda_min = np.linalg.eigvals(A @ A.T).min()
A /= np.sqrt(lambda_min)
assert np.allclose(np.linalg.eigvals(A @ A.T).min(), 1)


def dual_obj_cvx(n, A, b, c, y, coeff=1):
    x = cp.Variable(n) 
    f = -coeff * cp.sum(cp.huber(x - c, M=1)) - x.T @ (A.T @ y)
    prob = cp.Problem(cp.Maximize(f), [])
    prob.solve() 
    return -(f.value + b @ y)
def solve_primal_cvx(n, A, b, c, coeff=1):
    x = cp.Variable(n) 
    f = coeff * cp.sum(cp.huber(x - c, M=1))
    objective = cp.Minimize(f)
    constraints = [A @ x == b]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return f.value, x.value, constraints[0].dual_value

# Calulate primal
f_star, x_star, y_star = solve_primal_cvx(n, A, b, c)
print(f"{f_star=}, {np.linalg.matrix_rank(A)=}")
assert np.allclose(dual_obj_cvx(n, A, b, c, y_star), f_star)

# Set up calculation for dual
losses = []
z_0 = np.random.randn(m)
e2_0 = np.zeros(m)
# Save vars for MATLAB
dictMat = dict()
dictMat['A'] = A
dictMat['b'] = b
dictMat['c'] = c
dictMat['z'] = z_0
dictMat['f_star'] = f_star
from scipy.io import savemat
savemat('fromPython.mat',dictMat)


