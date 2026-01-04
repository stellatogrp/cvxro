import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Robust Facility Location
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Consider the Facility Location Problem (FLP) that focuses on finding the best locations for facilities such as warehouses, plants, or distribution centers to minimize costs and maximize service coverage. It involves deciding where to place these facilities to effectively meet customer demands, while accounting for factors like transportation costs, facility setup costs, and capacity constraints. The goal is to determine the optimal facility locations that efficiently serve demands and maximize overall profits. The approach defined in this notebook aims to maximize profits under the condition of uncertain demand, making this a robust optimization problem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Taking the example in [1, Section 2.4], let $ T, F, N$ be the length of the horizon, the number of candidate locations to which a facility can be assigned and the number of locations that have a demand for the facility respectively.

    - $\eta \in {\bf R}$ denotes the unit price of goods
    - $c, c^{\rm stor} ,c^{\rm open} \in {\bf R}^{F}$ denotes the cost per unit of production, cost per unit capacity and the cost of opening a facility at all locations
    - ${c^{\rm ship}} \in {\bf R}^{F\times N}$ denotes the cost of shipping from one location to another
    - ${d^{\rm u}_t} \in {\bf R}^{N}$ denotes the uncertain demand for period $t$ at location
    - $X_t \in {\bf R}^{F\times N}$ denotes the proportion of the demand at a location during period $t$ that is satisfied by a facility
    - $p_t\in {\bf R}^{F}$ denotes the amount of goods that is produced at some facility at a time period $t$
    - $y$ denotes whether a facility at a location is open or closed, by taking values 1 or 0, respectively
    - ${z}$ denotes the capacity of the facility in this location in case it is open
    - $d^*$ denotes the demand for a period in the deterministic case

     Let $M > 0$ be a large constant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    \begin{array}{ll}
    \text{maximize} & \theta \\
    \text{subject to} & \sum_{\tau = 1}^T \sum_{i = 1}^F \sum_{j = 1}^N (\eta - c^{\rm ship}_{ij}) X_{ij\tau} d^{u}_{i\tau} - \sum_{\tau = 1}^T \sum_{i = 1}^F c_i p_{i\tau} - \sum_{i = 1}^F c^{\rm stor}_i z_i - \sum_{i = 1}^F c^{\rm open}_i y_i \\
                    & \sum_{i = 1}^F X_{ij\tau} \leq 1, \\
                    & \sum_{j = 1}^N X_{ij\tau} d^{\rm u}_{j\tau} \leq P_{i\tau} \quad d^{\rm u} \in \mathcal{U}\\
                    & X \ge 0 \\
                    & p_t \leq {z} \quad t =1,\dots,T \\
                    & {z} \leq My \\
                    & y \in \{0, 1\}^F
    \end{array}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We solve this problem using the ellipsoidal uncertainty set, formulated by:

    $$ \mathcal{U}_{\text{ellips}} = \{z+d^* \ | \ \| z\|_2 \le \rho\} $$
    """)
    return


@app.cell
def _():
    import numpy as np
    import cvxpy as cp
    import lropt
    import networkx as nx
    import matplotlib.pyplot as plt
    return cp, lropt, np, nx, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the following snippet, we generate data. This example has $5$ facilities and $8$ candidate locations. The length of each horizon is $10$ and the unit price of each good is $100$.
    """)
    return


@app.cell
def _(cp, lropt, np):
    np.random.seed(1)
    T = 10
    F = 5
    N = 8
    M = 1100
    ETA = 100.0
    _RHO = 0.3
    c = np.random.rand(F)
    c_stor = np.random.rand(F)
    c_open = np.random.rand(F)
    c_ship = np.random.rand(F, N)
    _d_star = np.random.rand(N * T)
    x = {}
    for _i in range(F):
        x[_i] = cp.Variable((N, T))
    d_u = lropt.UncertainParameter(N * T, uncertainty_set=lropt.Ellipsoidal(b=_d_star, rho=_RHO))
    p = cp.Variable((F, T))
    z = cp.Variable(F)
    y = cp.Variable(F, boolean=True)
    theta = cp.Variable()  #Flattened Uncertain Parameter - LROPT only supports one dimensional uncertain parameters
    return ETA, F, M, N, T, c, c_open, c_ship, c_stor, d_u, p, theta, x, y, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we define all our constraints
    """)
    return


@app.cell
def _(
    ETA,
    F,
    M,
    N,
    T,
    c,
    c_open,
    c_ship,
    c_stor,
    cp,
    d_u,
    np,
    p,
    theta,
    x,
    y,
    z,
):
    _revenue = cp.sum([((ETA - np.diag(c_ship[_i])) @ x[_i]).flatten() @ d_u for _i in range(F)])
    _cost_production = cp.sum(c @ p)
    _fixed_costs = c_stor @ z
    _penalties = c_open @ y
    constraints = [_revenue - _cost_production - _fixed_costs - _penalties >= theta, z <= M * y]
    constraints.append(cp.sum([x[_i] for _i in range(F)]) <= 1)
    for _i in range(F):
        for _t in range(T):
            constraints.append(cp.sum([x[_i][j, _t] * d_u[j * T + _t] for j in range(N)]) <= p[_i, _t])
    for _i in range(F):
        constraints.append(x[_i] >= 0)
    for _t in range(T):
        constraints.append(p.T[_t] <= z)
    constraints = constraints + [p >= 0]
    return (constraints,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we define the objective and get the optimal value for the equation.
    """)
    return


@app.cell
def _(constraints, cp, lropt, theta):
    _objective = cp.Maximize(theta)
    _prob = lropt.RobustProblem(_objective, constraints)
    _prob.solve(solver=cp.SCIP)
    return


@app.cell
def _(theta):
    print(f"The robust optimal value using  is {theta.value:.3E}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To compare the solution of the problem without an uncertainty parameter, the following code solves the problem in the deterministic case.
    """)
    return


@app.cell
def _(cp, lropt, np):
    np.random.seed(1)
    T_1 = 10
    F_1 = 5
    N_1 = 8
    M_1 = 1100
    ETA_1 = 100.0
    _RHO = 0.3
    c_1 = np.random.rand(F_1)
    c_stor_1 = np.random.rand(F_1)
    c_open_1 = np.random.rand(F_1)
    c_ship_1 = np.random.rand(F_1, N_1)
    _d_star = np.random.rand(N_1 * T_1)
    d_u_1 = lropt.UncertainParameter(N_1 * T_1, uncertainty_set=lropt.Ellipsoidal(b=_d_star, rho=_RHO))
    p_1 = cp.Variable((F_1, T_1))
    z_1 = cp.Variable(F_1)
    y_1 = cp.Variable(F_1, boolean=True)
    x_det = {}
    for _i in range(F_1):
        x_det[_i] = cp.Variable((N_1, T_1))
    theta_1 = cp.Variable()
    _revenue = cp.sum([((ETA_1 - np.diag(c_ship_1[_i])) @ x_det[_i]).flatten() @ _d_star for _i in range(F_1)])
    _cost_production = cp.sum(c_1 @ p_1)
    _fixed_costs = c_stor_1 @ z_1
    _penalties = c_open_1 @ y_1
    constraints_1 = [_revenue - _cost_production - _fixed_costs - _penalties >= theta_1, z_1 <= M_1 * y_1]
    constraints_1.append(cp.sum([x_det[_i] for _i in range(F_1)]) <= 1)
    for _i in range(F_1):
        for _t in range(T_1):
            constraints_1.append(cp.sum([x_det[_i][j, _t] * _d_star[j * T_1 + _t] for j in range(N_1)]) <= p_1[_i, _t])
    for _i in range(F_1):
        constraints_1.append(x_det[_i] >= 0)
    for _t in range(T_1):
        constraints_1.append(p_1.T[_t] <= z_1)
    constraints_1 = constraints_1 + [p_1 >= 0]
    _objective = cp.Maximize(theta_1)
    _prob = lropt.RobustProblem(_objective, constraints_1)
    _prob.solve(solver=cp.SCIP)
    return F_1, N_1, T_1, theta_1


@app.cell
def _(theta_1):
    print(f'The deterministic optimal value using  is {theta_1.value:.3E}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a facility - network graph created using the robust optimal value of $x$.
    """)
    return


@app.cell
def _(F_1, N_1, T_1, np, nx, plt, x):
    x_opt = np.hstack([v.value.flatten() for v in x.values()]).reshape((F_1 * T_1, N_1))
    G = nx.DiGraph()
    facility_nodes = range(F_1)
    location_nodes = range(F_1, F_1 + N_1)
    G.add_nodes_from(facility_nodes, bipartite=0)
    G.add_nodes_from(location_nodes, bipartite=1)
    for _i in range(F_1):
        for j in range(N_1):
            if x_opt[_i * T_1, j] > 0:
                G.add_edge(_i, F_1 + j, weight=x_opt[_i * T_1, j])
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(15, 10))
    nx.draw_networkx_nodes(G, pos, nodelist=facility_nodes, node_color='lightblue', node_size=400, edgecolors='k', node_shape='o')
    nx.draw_networkx_nodes(G, pos, nodelist=location_nodes, node_color='lightgreen', node_size=400, edgecolors='k', node_shape='o')
    edges = G.edges(data=True)
    edge_weights = [data['weight'] for u, v, data in edges]
    edge_weights = np.array(edge_weights)
    edge_colors = edge_weights
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, alpha=0.7, edge_color=edge_colors, edge_cmap=plt.cm.Blues, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels={_i: f'F{_i}' for _i in facility_nodes}, font_size=12, font_weight='bold', verticalalignment='center')
    nx.draw_networkx_labels(G, pos, labels={F_1 + j: f'L{j}' for j in range(N_1)}, font_size=12, font_weight='bold', verticalalignment='center')
    plt.rcParams.update({'font.size': 18})
    plt.title('Facility Location Network', fontsize=16)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    1. Bertsimas, Dimitris, and Dick Den Hertog. Robust and Adaptive Optimization. [Dynamic Ideas LLC], 2022.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
