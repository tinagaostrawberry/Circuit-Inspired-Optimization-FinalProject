import ciropt as co
import cvxpy as cp
import numpy as np
import networkx as nx


R = 0.8
L_smooth = [100, 100]
mu = [0., 2]
Inductance = 2
Capacitance = 15

solver = "ipopt"
# solver = "ipopt_qcqp"
# solver = "ipopt_qcqp_matrix"

# random seed
np.random.seed(108)

G = nx.Graph()

n_nodes = 5
G.add_nodes_from([0, n_nodes-1])
G.add_edges_from([(0, 1), (0, 2), (0,3), (0,4), (1, 2), (1, 3), (1,4), (2, 3), (2, 4), (3, 4)])
S = {3, 4}
nx.draw(G, node_color='skyblue', with_labels=True, font_size=30, node_size=1500)

print(f"{mu=}, {L_smooth=}, {R=}, {Capacitance=}, {Inductance=}, {solver=}")

def dadmm_C_graph(mus, L_smooths, R, Capacitance, Inductance, G, S, params=None):
    problem = co.CircuitOpt()
    package = co.co_func
    Constraint = co.co_constr
    proximal_step = co.co_func.proximal_step
    h, eta, rho, gamma = problem.h, problem.eta, problem.rho, problem.gamma

    fs = [co.define_function(problem, mus[1 if i in S else 0], L_smooths[1 if i in S else 0], package) for i in range(G.number_of_nodes())]
    f = fs[1]
    for i in range(1, len(fs)):
        f = f + fs[i]
    x_star, y_star, f_star = f.stationary_point(return_gradient_and_function_value=True)
    y_star_f_star = [g.oracle(x_star) for g in fs]
    ys_star, fs_star = [a[0] for a in y_star_f_star], [a[1] for a in y_star_f_star]
    sumy = ys_star[0]
    for i in range(1, len(ys_star)):
        sumy = sumy + ys_star[i]
    # when f is not differentiable
    problem.add_constraint(Constraint((sumy - y_star) ** 2, "equality"))

    kcly = {}
    for e in G.edges():
        kcly[e] = problem.set_initial_point()
        kcly[(e[1], e[0])] = -kcly[e]
    for i in range(G.number_of_nodes()):
        neighs = list(G.neighbors(i))
        sumy = kcly[(i, neighs[0])]
        for n in neighs[1:]:
            sumy = sumy + kcly[(i, n)]
        sumy = sumy - ys_star[i]
        problem.add_constraint(Constraint(sumy ** 2, "equality"))

    e_1 = {}
    for e in G.edges():
        e_1[e] = problem.set_initial_point()
        e_1[(e[1], e[0])] = e_1[e]
    e_2 = {}

    i_L_1 = {}
    for e in G.edges():
        i_L_1[e] = problem.set_initial_point()
        if (e[0] in S) and (e[1] in S):
            i_L_1[(e[1], e[0])] = problem.set_initial_point()
        else:
            i_L_1[(e[1], e[0])] = -i_L_1[e]
    i_L_2 = {}

    x_2 = []
    y_2 = []
    f_2 = []
    for i in range(G.number_of_nodes()):
        neighs = list(G.neighbors(i))
        z = R * i_L_1[(i, neighs[0])] + e_1[(i, neighs[0])]
        nn = 1
        for n in neighs[1:]:
            z += R * i_L_1[(i, n)] + e_1[(i, n)]
            nn += 1
        ax_2, ay_2, af_2 = proximal_step(z / nn, fs[i], R / nn)
        x_2.append(ax_2)
        y_2.append(ay_2)
        f_2.append(af_2)

    for e in G.edges():
        if (e[0] in S) and (e[1] in S):
            e_2[e] = e_1[e] - h * (R * (i_L_1[e] + i_L_1[(e[1], e[0])]) + 2 * e_1[e] - x_2[e[0]] - x_2[e[1]]) / (Capacitance * R)
        else:
            e_2[e] = (x_2[e[0]] + x_2[e[1]]) / 2
        e_2[(e[1], e[0])] = e_2[e]

    for e in G.edges():
        i_L_2[e] = i_L_1[e] + h * (e_2[e] - x_2[e[0]]) / Inductance
        if (e[0] in S) and (e[1] in S):
            i_L_2[(e[1], e[0])] = i_L_1[(e[1], e[0])] + h * (e_2[(e[1], e[0])] - x_2[e[1]]) / Inductance
        else:
            i_L_2[(e[1], e[0])] = -i_L_2[e]

    edges = list(G.edges())
    E_1e = (e_1[edges[0]] - x_star) ** 2
    for e in edges[1:]:
        E_1e = E_1e + (e_1[e] - x_star) ** 2
    E_1e = E_1e * gamma

    il = list(i_L_1.keys())
    E_1i = (i_L_1[il[0]] - kcly[il[0]]) ** 2
    for e in il[1:]:
        E_1i = E_1i + (i_L_1[e] - kcly[e]) ** 2
    E_1i = E_1i * Inductance / 2

    Cedges = [e for e in edges if ((e[0] in S) and (e[1] in S))]
    E_1c = (e_1[Cedges[0]] - x_star) ** 2
    for e in Cedges[1:]:
        E_1c = E_1c + (e_1[e] - x_star) ** 2
    E_1c = E_1c * Capacitance / 2
    E_1 = E_1e + E_1i + E_1c

    edges = list(G.edges())
    E_2e = (e_2[edges[0]] - x_star) ** 2
    for e in edges[1:]:
        E_2e = E_2e + (e_2[e] - x_star) ** 2
    E_2e = E_2e * gamma

    il = list(i_L_2.keys())
    E_2i = (i_L_2[il[0]] - kcly[il[0]]) ** 2
    for e in il[1:]:
        E_2i = E_2i + (i_L_2[e] - kcly[e]) ** 2
    E_2i = E_2i * Inductance / 2

    E_2c = (e_2[Cedges[0]] - x_star) ** 2
    for e in Cedges[1:]:
        E_2c = E_2c + (e_1[e] - x_star) ** 2
    E_2c = E_2c * Capacitance / 2
    E_2 = E_2e + E_2i + E_2c

    edges = list(e_2.keys())
    Delta_2e = (e_2[edges[0]] - x_2[edges[0][0]]) ** 2
    for e in edges[1:]:
        Delta_2e = Delta_2e + (e_2[e] - x_2[e[0]]) ** 2
    Delta_2e = rho * Delta_2e / R
    Delta_2f = f_2[0] - fs_star[0] - ys_star[0] * (x_2[0] - x_star)
    for i in range(1, G.number_of_nodes()):
        Delta_2f = Delta_2f + f_2[i] - fs_star[i] - ys_star[i] * (x_2[i] - x_star)
    Delta_2f = Delta_2f * eta
    # currents on resistors on each net sum to 0
    Delta_2 = Delta_2e + Delta_2f
    problem.set_performance_metric(E_2 - (E_1 - Delta_2))
    return problem

#problem = co.dadmm_C_graph6( mu, L_smooth, R, Capacitance, Inductance)
problem = dadmm_C_graph(mu, L_smooth, R, Capacitance, Inductance, G, S)
problem.obj = problem.eta + problem.rho

res, sol = problem.solve(solver=solver, extra_dim=530, verbose=True)[:2]





