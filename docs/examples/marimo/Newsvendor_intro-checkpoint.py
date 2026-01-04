import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Newsvendor Intro Example
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    import scipy as sc
    import numpy as np
    import numpy.random as npr
    import torch
    from sklearn import datasets
    import pandas as pd
    import lropt
    import sys
    sys.path.append('..')
    from utils import plot_tradeoff,plot_iters, plot_contours, plot_contours_line
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings("ignore")
    plt.rcParams.update({
        "text.usetex":True,

        "font.size":22,
        "font.family": "serif"
    })
    return (
        cp,
        lropt,
        np,
        plot_contours_line,
        plot_iters,
        plt,
        sc,
        torch,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 1: Intro example, max of affine uncertainty
    We consider a problem with max of affine uncertainty,

    $$g(u,x) = \max_{l=1,\dots,L} (P_lu + a_l)^Tx,$$
    where $P_l$ and $a_l$ are constants for all $l = 1, \dots, L$. The robust formulation is
    $$
    \begin{array}{ll}
    \text{minimize} & \tau\\
    \text{subject to}  & \max_{l=1,\dots,L} (P_lu + a_l)^Tx  \leq  \tau \quad \forall u \in \mathcal{U}(\theta)\\
    & x \geq 0,
    \end{array}
    $$


    where $\theta$ encodes the training parameters $(A,b)$.

    We formulate the Newsvendor problem with this framework, where we solve
    \begin{equation}
    	\begin{array}{ll}
    		\text{minimize} & \tau\\
    		\text{subject to} & k^Tx  + \max\{-p^Tx,- p^Tu\} \le \tau \quad \forall u \in \mathcal{U}(\theta) \\
    		& x \geq 0.
    	\end{array}
    \end{equation}
    """)
    return


@app.cell
def _(np, plt, torch):
    # Formulate constants
    n = 2
    N = 5000
    test_perc = 0.99
    # k = npr.uniform(1,4,n)
    # p = k + npr.uniform(2,5,n)
    k = np.array([4.0, 5.0])
    p = np.array([5, 6.5])
    # k_tch = torch.tensor(k, requires_grad = True)
    # p_tch = torch.tensor(p, requires_grad = True)

    def loss(t, x, k_tch, p_tch, alpha, data, mu=1, l=5, quantile=0.95, target=1.0):
        sums = torch.mean(torch.maximum(torch.maximum(k_tch @ x - data @ p_tch, k_tch @ x - x @ p_tch) - _t - alpha, torch.tensor(0.0, requires_grad=True)))
        sums = sums / (1 - quantile) + alpha
        return (_t + l * (sums - target) + mu / 2 * (sums - target) ** 2, _t, torch.mean((torch.maximum(torch.maximum(k_tch @ x - data @ p_tch, k_tch @ x - x @ p_tch) - _t, torch.tensor(0.0, requires_grad=True)) >= 0.001).float()), sums.detach().numpy())

    def gen_demand_intro(N, seed):
        np.random.seed(seed)
        sig = np.array([[0.6, -0.4], [-0.3, 0.1]])
        mu = np.array((0.9, 0.7))
        norms = np.random.multivariate_normal(mu, sig, N)
        d_train = np.exp(norms)
        return d_train

    def gen_demand_intro_2(N, seed):
        np.random.seed(seed)
        sig = np.array([[8, 3], [2, 2]])
        mu = np.array((12, 5))
        norms = np.random.multivariate_normal(mu, sig, N)
        return norms
    data = gen_demand_intro(N, seed=5)
    data2 = gen_demand_intro_2(N, seed=5)
    plt.scatter(data[:, 0], data[:, 1], color='tab:blue')
    plt.scatter(data2[:, 0], data2[:, 1], color='tab:blue')  # d_train = np.exp(norms)
    # Generate data
    data = np.vstack((data, data2))
    return data, k, loss, n, p, test_perc


@app.cell
def _(k, n, np, p):
    scenarios = {}
    num_scenarios = 3
    for _scene in range(num_scenarios):
        np.random.seed(_scene)
        scenarios[_scene] = {}
        scenarios[_scene][0] = k
        scenarios[_scene][1] = p + np.random.normal(0, 1, 2)
    scenarios = {}
    num_scenarios = 8
    for _scene in range(num_scenarios):
        np.random.seed(_scene)
        scenarios[_scene] = {}
        scenarios[_scene][0] = np.random.uniform(1, 4, n)
    # scenarios = {}
    # num_scenarios = 1
    # for scene in range(num_scenarios):
    #   np.random.seed(scene)
    #   scenarios[scene]={}
    #   scenarios[scene][0] = k
    #   scenarios[scene][1] = p
        scenarios[_scene][1] = scenarios[_scene][0] + np.random.uniform(1, 4, n)
    return num_scenarios, scenarios


@app.cell
def _(
    cp,
    data,
    loss,
    lropt,
    n,
    np,
    num_scenarios,
    sc,
    scenarios,
    test_perc,
    train_test_split,
):
    _u = lropt.UncertainParameter(n, uncertainty_set=lropt.Ellipsoidal(p=2, data=data, loss=loss))
    _x_r = cp.Variable(n)
    _t = cp.Variable()
    k_1 = cp.Parameter(2)
    p_1 = cp.Parameter(2)
    k_1.value = scenarios[0][0]
    p_1.value = scenarios[0][1]
    _objective = cp.Minimize(_t)
    _constraints = [cp.maximum(k_1 @ _x_r - p_1 @ _x_r, k_1 @ _x_r - p_1 @ _u) <= _t]
    _constraints = _constraints + [_x_r >= 0]
    _prob = lropt.RobustProblem(_objective, _constraints)
    target = -0.05
    s = 15
    train, test = train_test_split(data, test_size=int(data.shape[0] * test_perc), random_state=s)
    init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(train.T)))
    init_bval = -init @ np.mean(train, axis=0)
    result1 = _prob.train(lr=1e-05, step=800, momentum=0.8, optimizer='SGD', seed=s, init_A=init, init_b=init_bval, fixb=False, init_mu=1, init_lam=0, target_cvar=target, init_alpha=-0.01, test_percentage=test_perc, save_iters=True, scenarios=scenarios, num_scenarios=num_scenarios, max_inner_iter=15)
    df1 = result1.df
    A_fin = result1.A
    b_fin = result1.b
    A1_iters, b1_iters = result1.uncset_iters
    result2 = _prob.train(eps=True, lr=1e-05, step=500, momentum=0.8, optimizer='SGD', seed=s, init_A=A_fin, init_b=b_fin, init_mu=1, init_lam=0, target_cvar=target, init_alpha=-0.01, test_percentage=test_perc, scenarios=scenarios, num_scenarios=num_scenarios, max_inner_iter=1)
    df_r1 = result2.df
    result3 = _prob.train(eps=True, lr=1e-05, step=800, momentum=0.8, optimizer='SGD', seed=s, init_A=init, init_b=init_bval, init_mu=1, init_lam=0, target_cvar=target, init_alpha=-0.01, test_percentage=test_perc, scenarios=scenarios, num_scenarios=num_scenarios, max_inner_iter=1)
    df_r2 = result3.df
    result4 = _prob.grid(epslst=np.linspace(0.01, 2.98, 40), init_A=result3.A, init_b=result3.b, seed=s, init_alpha=-0.0, test_percentage=test_perc, scenarios=scenarios, num_scenarios=num_scenarios)
    dfgrid = result4.df
    result5 = _prob.grid(epslst=np.linspace(0.01, 2.98, 40), init_A=A_fin, init_b=b_fin, seed=s, init_alpha=-0.0, test_percentage=test_perc, scenarios=scenarios, num_scenarios=num_scenarios)
    dfgrid2 = result5.df
    return (
        A_fin,
        b_fin,
        df1,
        df_r2,
        dfgrid,
        dfgrid2,
        init,
        init_bval,
        result3,
        train,
    )


@app.cell
def _(df1, df_r2, plot_iters):
    plot_iters(df1,"news")
    plot_iters(df_r2,"news")
    return


@app.cell
def _(dfgrid, dfgrid2, np, plt):
    eps_list = np.linspace(0.01, 2.98, 40)
    inds = [13, 8, 6, 5]
    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(np.vstack(dfgrid['Violations']), axis=1)[:], np.mean(np.vstack(dfgrid['Test_val']), axis=1)[:], color='tab:blue', label='Standard set', marker='v', zorder=0)
    plt.fill(np.append(np.quantile(np.vstack(dfgrid['Violations']), 0.1, axis=1), np.quantile(np.vstack(dfgrid['Violations']), 0.9, axis=1)[::-1]), np.append(np.quantile(np.vstack(dfgrid['Test_val']), 0.1, axis=1), np.quantile(np.vstack(dfgrid['Test_val']), 0.9, axis=1)[::-1]), color='tab:blue', alpha=0.2)
    for _ind in range(4):
        plt.scatter(np.mean(np.vstack(dfgrid['Violations']), axis=1)[inds[_ind]], np.mean(np.vstack(dfgrid['Test_val']), axis=1)[inds[_ind]], color='tab:green', s=50, marker='v', zorder=10)
        plt.annotate('$\\epsilon$ = {}'.format(round(eps_list[inds[_ind]], 2)), (np.mean(np.vstack(dfgrid['Violations']), axis=1)[inds[_ind]], np.mean(np.vstack(dfgrid['Test_val']), axis=1)[inds[_ind]]), textcoords='offset points', xytext=(5, 3), ha='left', color='tab:green', fontsize=15)
    plt.plot(np.mean(np.vstack(dfgrid2['Violations']), axis=1), np.mean(np.vstack(dfgrid2['Test_val']), axis=1), color='tab:orange', label='Reshaped set', marker='^', zorder=1)  # this is the text
    plt.fill(np.append(np.quantile(np.vstack(dfgrid2['Violations']), 0.1, axis=1), np.quantile(np.vstack(dfgrid2['Violations']), 0.9, axis=1)[::-1]), np.append(np.quantile(np.vstack(dfgrid2['Test_val']), 0.1, axis=1), np.quantile(np.vstack(dfgrid2['Test_val']), 0.9, axis=1)[::-1]), color='tab:orange', alpha=0.2)  # these are the coordinates to position the label
    for _ind in [0, 1, 3]:  # how to position the text
        plt.scatter(np.mean(np.vstack(dfgrid2['Violations']), axis=1)[inds[_ind]], np.mean(np.vstack(dfgrid2['Test_val']), axis=1)[inds[_ind]], color='black', s=50, marker='^')  # distance from text to points (x,y)
        plt.annotate('$\\epsilon$ = {}'.format(round(eps_list[inds[_ind]], 2)), (np.mean(np.vstack(dfgrid2['Violations']), axis=1)[inds[_ind]], np.mean(np.vstack(dfgrid2['Test_val']), axis=1)[inds[_ind]]), textcoords='offset points', xytext=(5, 1), ha='left', color='black', fontsize=15)
    plt.ylabel('Objective valye')
    plt.xlabel('Probability of constraint violation')
    plt.ylim([-50, 70])
    plt.legend()
    # ax2.set_xlim([-1,20])
    #plt.xscale("log")
    # lgd = plt.legend(loc = "lower right", bbox_to_anchor=(1.5, 0.3))
    # plt.savefig("ex1_curves1.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig('ex1_curves_orig.pdf', bbox_inches='tight')  # this is the text  # these are the coordinates to position the label  # how to position the text  # distance from text to points (x,y)
    return eps_list, inds


@app.cell
def _(dfgrid, dfgrid2, eps_list, inds, np, plt):
    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(np.vstack(dfgrid['Violation_val']), axis=1)[:], np.mean(np.vstack(dfgrid['Test_val']), axis=1)[:], color='tab:blue', label='Standard set', marker='v', zorder=0)
    plt.fill(np.append(np.quantile(np.vstack(dfgrid['Violation_val']), 0.1, axis=1), np.quantile(np.vstack(dfgrid['Violation_val']), 0.9, axis=1)[::-1]), np.append(np.quantile(np.vstack(dfgrid['Test_val']), 0.1, axis=1), np.quantile(np.vstack(dfgrid['Test_val']), 0.9, axis=1)[::-1]), color='tab:blue', alpha=0.2)
    for _ind in range(4):
        plt.scatter(np.mean(np.vstack(dfgrid['Violation_val']), axis=1)[inds[_ind]], np.mean(np.vstack(dfgrid['Test_val']), axis=1)[inds[_ind]], color='tab:green', s=50, marker='v', zorder=10)
        plt.annotate('$\\epsilon$ = {}'.format(round(eps_list[inds[_ind]], 2)), (np.mean(np.vstack(dfgrid['Violation_val']), axis=1)[inds[_ind]], np.mean(np.vstack(dfgrid['Test_val']), axis=1)[inds[_ind]]), textcoords='offset points', xytext=(5, 3), ha='left', color='tab:green', fontsize=15)  # this is the text
    plt.plot(np.mean(np.vstack(dfgrid2['Violation_val']), axis=1), np.mean(np.vstack(dfgrid2['Test_val']), axis=1), color='tab:orange', label='Reshaped set', marker='^', zorder=1)  # these are the coordinates to position the label
    plt.fill(np.append(np.quantile(np.vstack(dfgrid2['Violation_val']), 0.1, axis=1), np.quantile(np.vstack(dfgrid2['Violation_val']), 0.9, axis=1)[::-1]), np.append(np.quantile(np.vstack(dfgrid2['Test_val']), 0.1, axis=1), np.quantile(np.vstack(dfgrid2['Test_val']), 0.9, axis=1)[::-1]), color='tab:orange', alpha=0.2)  # how to position the text
    for _ind in [1, 3]:  # distance from text to points (x,y)
        plt.scatter(np.mean(np.vstack(dfgrid2['Violation_val']), axis=1)[inds[_ind]], np.mean(np.vstack(dfgrid2['Test_val']), axis=1)[inds[_ind]], color='black', s=50, marker='^')
        plt.annotate('$\\epsilon$ = {}'.format(round(eps_list[inds[_ind]], 2)), (np.mean(np.vstack(dfgrid2['Violation_val']), axis=1)[inds[_ind]], np.mean(np.vstack(dfgrid2['Test_val']), axis=1)[inds[_ind]]), textcoords='offset points', xytext=(5, 1), ha='left', color='black', fontsize=15)
    plt.ylabel('Objective valye')
    plt.ylim([-50, 70])
    plt.xlabel('Empirical $\\mathbf{CVaR}$')
    plt.legend()
    # ax2.set_xlim([-1,20])
    # lgd = plt.legend(loc = "lower right", bbox_to_anchor=(1.5, 0.3))
    # plt.savefig("ex1_curves1.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig('ex1_cvar.pdf', bbox_inches='tight')  # this is the text  # these are the coordinates to position the label  # how to position the text  # distance from text to points (x,y)
    return


@app.cell
def _(
    A_fin,
    b_fin,
    cp,
    eps_list,
    inds,
    init,
    init_bval,
    lropt,
    np,
    num_scenarios,
    result3,
    scenarios,
):
    k_2 = np.array([4.0, 5.0])
    p_2 = np.array([5, 6.5])
    x_opt_base = {}
    x_opt_learned = {}
    t_learned = {}
    t_base = {}
    for _ind in range(4):
        x_opt_base[_ind] = {}
        x_opt_learned[_ind] = {}
        t_learned[_ind] = {}
        t_base[_ind] = {}
        for _scene in range(num_scenarios):
            n_1 = 2
            _u = lropt.UncertainParameter(n_1, uncertainty_set=lropt.Ellipsoidal(p=2, A=1 / eps_list[inds[_ind]] * result3.A, b=1 / eps_list[inds[_ind]] * result3.b))
            _x_r = cp.Variable(n_1)
            _t = cp.Variable()
            k_2 = scenarios[_scene][0]
            p_2 = scenarios[_scene][1]
            _objective = cp.Minimize(_t)
            _constraints = [cp.maximum(k_2 @ _x_r - p_2 @ _x_r, k_2 @ _x_r - p_2 @ _u) <= _t]
            _constraints = _constraints + [_x_r >= 0]
            _prob = lropt.RobustProblem(_objective, _constraints)
            _prob.solve()
            x_opt_base[_ind][_scene] = _x_r.value
            t_base[_ind][_scene] = _t.value
            n_1 = 2
            _u = lropt.UncertainParameter(n_1, uncertainty_set=lropt.Ellipsoidal(p=2, A=1 / eps_list[inds[_ind]] * A_fin, b=1 / eps_list[inds[_ind]] * b_fin))
            _x_r = cp.Variable(n_1)
            _t = cp.Variable()
            k_2 = scenarios[_scene][0]
            p_2 = scenarios[_scene][1]
            _objective = cp.Minimize(_t)
            _constraints = [cp.maximum(k_2 @ _x_r - p_2 @ _x_r, k_2 @ _x_r - p_2 @ _u) <= _t]
            _constraints = _constraints + [_x_r >= 0]
            _prob = lropt.RobustProblem(_objective, _constraints)
            _prob.solve()
            x_opt_learned[_ind][_scene] = _x_r.value
            t_learned[_ind][_scene] = _t.value
            (x_opt_learned, x_opt_base, t_learned, t_base)
    (A_fin, b_fin, init, init_bval)
    return n_1, t_base, t_learned, x_opt_base, x_opt_learned


@app.cell
def _(
    A_fin,
    b_fin,
    eps_list,
    inds,
    n_1,
    np,
    num_scenarios,
    result3,
    scenarios,
    t_base,
    t_learned,
    train,
    x_opt_base,
    x_opt_learned,
):
    K = 1
    num_p = 50
    offset = 2
    x_min, x_max = (np.min(train[:, 0]) - offset, np.max(train[:, 0]) + offset)
    y_min, y_max = (np.min(train[:, 1]) - offset, np.max(train[:, 1]) + offset)
    X = np.linspace(x_min, x_max, num_p)
    Y = np.linspace(y_min, y_max, num_p)
    x, y = np.meshgrid(X, Y)
    fin_set = {}
    init_set = {}
    for _ind in range(4):
        fin_set[_ind] = {}
        init_set[_ind] = {}
        for k_ind in range(K):
            fin_set[_ind][k_ind] = np.zeros((num_p, num_p))
            init_set[_ind][k_ind] = np.zeros((num_p, num_p))
    g_level_learned = {}
    g_level_base = {}
    for _ind in range(4):
        g_level_learned[_ind] = {}
        g_level_base[_ind] = {}
        for _scene in range(num_scenarios):
            g_level_learned[_ind][_scene] = np.zeros((num_p, num_p))
            g_level_base[_ind][_scene] = np.zeros((num_p, num_p))
        for i in range(num_p):
            for j in range(num_p):
                u_vec = [x[i, j], y[i, j]]
                for k_ind in range(K):
                    fin_set[_ind][k_ind][i, j] = np.linalg.norm(1 / eps_list[inds[_ind]] * A_fin[k_ind * n_1:(k_ind + 1) * n_1, 0:n_1] @ u_vec + 1 / eps_list[inds[_ind]] * b_fin)
                for k_ind in range(K):
                    init_set[_ind][k_ind][i, j] = np.linalg.norm(1 / eps_list[inds[_ind]] * result3.A[k_ind * n_1:(k_ind + 1) * n_1, 0:n_1] @ u_vec + 1 / eps_list[inds[_ind]] * result3.b)
                for _scene in range(num_scenarios):
                    g_level_learned[_ind][_scene][i, j] = np.maximum(scenarios[_scene][0] @ x_opt_learned[_ind][_scene] - scenarios[_scene][1] @ x_opt_learned[_ind][_scene], scenarios[_scene][0] @ x_opt_learned[_ind][_scene] - scenarios[_scene][1] @ u_vec) - t_learned[_ind][_scene]
                    g_level_base[_ind][_scene][i, j] = np.maximum(scenarios[_scene][0] @ x_opt_base[_ind][_scene] - scenarios[_scene][1] @ x_opt_base[_ind][_scene], scenarios[_scene][0] @ x_opt_base[_ind][_scene] - scenarios[_scene][1] @ u_vec) - t_base[_ind][_scene]
    return fin_set, g_level_base, g_level_learned, init_set, x, y


@app.cell
def _(
    eps_list,
    fin_set,
    g_level_base,
    g_level_learned,
    inds,
    init_set,
    num_scenarios,
    plot_contours_line,
    train,
    x,
    y,
):
    plot_contours_line(x,y,init_set, g_level_base,eps_list, inds, num_scenarios,train, "news_intro",standard = True)
    plot_contours_line(x,y,fin_set, g_level_learned,eps_list, inds, num_scenarios,train, "news_intro",standard = False)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
