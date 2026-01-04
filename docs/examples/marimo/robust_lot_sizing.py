import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Robust Lot Sizing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Consider the problem of Lot Sizing, where you need to decide how much to produce or order in each time period to meet demand and minimize costs. This involves planning for a fixed period with known demand and various costs such as setup, production, holding inventory, and handling backorders. The main goal is to create a production plan that meets demand efficiently. In its robust form, this problem also accounts for uncertainties in production yields, which can vary within a certain range.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the example taken from [1, Section 3.3] we consider a single item multi period uncapacitated lot sizing problem with backorder and production yield that determine the quantity to produce in each time period of the finite planning horizon $T$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The problem we want to solve is the uncertain linear optimization problem of the form:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    \begin{array}{ll}
    \text{minimize} & \sum_{t = 1}^T\left(s_t^Ty_t + v_t^Tx_t + {k}_t\right) \\
    \text{subject to} & k_t \geq h_t\left( p_t^Tx_t - d_t\right) \quad t =1,\dots,T \quad \forall p_t \in \mathcal{U}\\
                      & k_t \geq -b_t\left( p_t^Tx_t - d_t\right) \quad t =1,\dots,T \quad \forall p_t \in \mathcal{U}\\
                      & x_t \leq {m_t}^T{y_t} \quad t =1,\dots,T \\
                      &  x \geq \mathbf{0}\\
                      & {h'} \geq \mathbf{0}
    \end{array}
    $$

    where $s$ denotes the setup cost, $h$ denotes the inventory holding cost , $b$ denotes the backorder cost $b$, $x$ denotes the lot size to be produced, and $d$ denotes the demand for each time period in $T$. $h'$ is independently defined for each period as the highest cost between the worst inventory cost and the worst backorder cost under the uncertainty set. The model contains the decision variables $X$, the lot size to be produced, and the setup decision $Y$. The strictly positive uncertain production yield is defined as $p$. $m$ denotes the upper bound on production quantity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To solve this problem, we first import the required packages.
    """)
    return


@app.cell
def _():
    import numpy as np
    import lropt
    import cvxpy as cp
    return cp, lropt, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We start by defining the relevant constants and creating variables.
    """)
    return


@app.cell
def _(cp, np):
    np.random.seed(1)
    T = 5
    RHO = 2
    s = np.random.rand(T)
    v = np.random.rand(T)
    h = np.random.rand(T)
    b = np.random.rand(T)
    d = np.random.randint(10, 20, size=T)

    # Large positive number for setup constraint
    m = np.full(T, 1000)
    x = cp.Variable(T, nonneg=True)
    y = cp.Variable(T, boolean=True)
    k = cp.Variable(T, nonneg=True)
    return RHO, T, b, d, h, k, m, s, v, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the following we define the parameter $p$ as an uncertain parameter that belongs a Budget Uncertainty set. The budget uncertainty is defined as:
    $$
            \mathcal{U}_{\text{budg}} = \{z \ | \ \|z \|_\infty \le \rho_1,
            \|z \|_1 \leq \rho_2\}
    $$
    """)
    return


@app.cell
def _(RHO, T, lropt):
    p = lropt.UncertainParameter(T, uncertainty_set = lropt.Budget(rho1 = RHO, rho2= RHO), nonneg = True)
    return (p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We define our objective and constraints.
    """)
    return


@app.cell
def _(b, cp, d, h, k, m, p, s, v, x, y):
    objective = cp.Minimize(cp.sum(s @ y + v @ x + k))
    constraints = [
    k >= h * cp.sum(p @ x - d),
    k >= (-b) * cp.sum(p @ x - d),
    x <= m@y
    ]
    return constraints, objective


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we solve the problem and get the optimal value.
    """)
    return


@app.cell
def _(constraints, lropt, objective):
    prob = lropt.RobustProblem(objective, constraints)
    prob.solve()
    return


@app.cell
def _(k, np):
    formatted_values = [f"{value:.3E}" for value in np.array(k.value)]
    print(f"The robust optimal values for the total cost using Budget uncertainty are {', '.join(formatted_values)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    1. Metzker, P., Thevenin, S., Adulyasak, Y., & Dolgui, A. (2023). Robust optimization for lot-sizing problems under yield uncertainty. https://doi.org/10.1016/j.cor.2022.106025
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
