import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Robust Knapsack
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Consider the robust knapsack problem introduced in [1, Section 6.1]. This problem seeks to optimize the selection of items under worst-case scenarios, ensuring that the knapsack's total value is maximized while remaining feasible despite uncertainties in values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using the same formulation as in [1], the problem can be formulated as follows:
    $$
    \begin{array}{ll}
    \text{maximize} & {c}^T {x} \\
    \text{subject to} & {w}^T {x} \leq b \quad \forall {w} \in \mathcal{U} \\
    & {x} \in \{0, 1\}^n
    \end{array}
    $$

    where there are $n$ items, ${x}$ are the binary decision variables, their values are denoted by ${c}$, and their weights ${w}$ belong to a box uncertainty set, where the expected weights are denoted by ${w_e}$, and their uncertainties are captured by $\pmb{\delta}$.
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    import numpy as np
    import cvxro

    np.random.seed(seed=1234)
    return cp, cvxro, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We define the constants as shown below:
    """)
    return


@app.cell
def _(np):
    n = 200 #Number of items
    b = 1000 #Capacity
    c = np.random.uniform(low=0., high=1., size=n) #Value of each item
    w_e = np.random.uniform(low=1., high=2, size=n) #Mean weight of each item
    delta = np.random.uniform(low=0., high=0.1, size=n) #Weights uncertainties
    return b, c, delta, n, w_e


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The uncertain parameter $\mathbf{p}$ is formulated using LROPT in the block below. We use the box uncertainty set, which is defined as follows:

    $$\mathcal{U} = \{ \mathbf{w} \mid \bar{\mathbf{w}} - \delta \leq \mathbf{w} \leq \bar{\mathbf{w}} + \delta \}$$
    """)
    return


@app.cell
def _(b, c, cp, delta, cvxro, n, np, w_e):
    uncertainty_set = cvxro.Box(rho=1, a=np.diag(delta), b=w_e)
    w = cvxro.UncertainParameter(n, uncertainty_set=uncertainty_set) #Uncertain parameter
    x = cp.Variable(n, boolean=True) #Optimization variable

    #Define and solve the problem
    objective = cp.Maximize(c@x)
    constraints = [w@x <= b]
    prob = cvxro.RobustProblem(objective=objective, constraints=constraints)
    prob.solve(solver = cp.SCIP)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    1.  Bertsimas and Sim 2004 (https://pubsonline.informs.org/doi/abs/10.1287/opre.1030.0065)
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
