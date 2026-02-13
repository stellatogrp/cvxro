"""Tests for augmented Lagrangian improvement strategies.

Tests the three dual-update strategies (classic, pi, adaptive),
constraint smoothing (relu, softplus), and related settings.

All tests use minimal problem sizes (n=2, N=20) and few iterations
to keep memory low and runtime fast.
"""

import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt
import scipy as sc
import torch

from cvxro import Trainer, TrainerSettings
from cvxro.parameter import ContextParameter
from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal


def _make_portfolio_problem(n=2, N=20, seed=42):
    """Create a minimal portfolio problem with a chance constraint.

    Based on test_portfolio_intro pattern which is known to work.
    Returns (prob, trainer, data) so tests can configure settings and train.
    """
    np.random.seed(seed)
    sig = np.array([[0.5, -0.3], [-0.3, 0.4]])[:n, :n]
    mu_vec = np.random.uniform(0.2, 0.5, n)
    data = np.random.multivariate_normal(mu_vec, sig, N)

    # ContextParameter must be referenced in the problem
    dist = np.ones(n) * 3.0
    y_data = np.random.dirichlet(dist, N)
    y = ContextParameter(n, data=y_data)
    u = UncertainParameter(n, uncertainty_set=Ellipsoidal(p=2, data=data))

    x = cp.Variable(n)
    t_var = cp.Variable()
    # y is used in objective so it's discovered as a parameter
    objective = cp.Minimize(t_var + 0.2 * cp.norm(x - y, 1))
    constraints = [-x @ u <= t_var, cp.sum(x) == 1, x >= 0]
    eval_exp = -x @ u + 0.2 * cp.norm(x - y, 1)
    prob = RobustProblem(objective, constraints, eval_exp=eval_exp)

    trainer = Trainer(prob)
    return prob, trainer, data


def _base_settings(data, n=2, N=20):
    """Return TrainerSettings for a fast, minimal training run."""
    from sklearn.model_selection import train_test_split

    test_p = 0.1
    train, _ = train_test_split(
        data, test_size=max(int(N * test_p), 1), random_state=5
    )
    init_A = sc.linalg.sqrtm(np.cov(train.T) + 1e-4 * np.eye(n))
    init_b = np.mean(train, axis=0)

    settings = TrainerSettings()
    settings.lr = 0.0001
    settings.num_iter = 6
    settings.optimizer = "SGD"
    settings.momentum = 0.8
    settings.seed = 5
    settings.init_A = init_A
    settings.init_b = init_b
    settings.init_lam = 0.5
    settings.init_mu = 0.01
    settings.mu_multiplier = 1.001
    settings.init_alpha = 0.0
    settings.kappa = -0.001
    settings.test_percentage = test_p
    settings.validate_percentage = 0.01
    settings.parallel = False
    settings.random_init = False
    settings.num_random_init = 1
    settings.position = False
    settings.eta = 0.05
    settings.aug_lag_update_interval = 3  # trigger AL update twice in 6 iters
    settings.save_history = False
    settings.test_frequency = 100  # skip test eval
    settings.validate_frequency = 100  # skip validation eval
    return settings


class TestALSettings(unittest.TestCase):
    """Test that new AL settings fields exist and are configurable."""

    def test_default_values(self):
        s = TrainerSettings()
        self.assertEqual(s.dual_update_strategy, "classic")
        self.assertEqual(s.constraint_smoothing, "relu")
        self.assertAlmostEqual(s.softplus_beta, 10.0)
        self.assertAlmostEqual(s.pi_kp, 0.5)
        self.assertAlmostEqual(s.pi_ki, 0.1)
        self.assertAlmostEqual(s.pi_nu, 0.9)
        self.assertAlmostEqual(s.penalty_ema_decay, 0.99)
        self.assertAlmostEqual(s.penalty_eta_scale, 1.0)
        self.assertAlmostEqual(s.penalty_eps, 1e-8)
        self.assertTrue(s.reset_prev_cost_on_al_update)

    def test_set_strategy(self):
        s = TrainerSettings()
        for strategy in ("classic", "pi", "adaptive"):
            s.dual_update_strategy = strategy
            self.assertEqual(s.dual_update_strategy, strategy)

    def test_set_smoothing(self):
        s = TrainerSettings()
        for smoothing in ("relu", "softplus"):
            s.constraint_smoothing = smoothing
            self.assertEqual(s.constraint_smoothing, smoothing)

    def test_slots_reject_unknown(self):
        s = TrainerSettings()
        with self.assertRaises(AttributeError):
            s.nonexistent_field = 1


class TestSmoothConstraint(unittest.TestCase):
    """Test the _smooth_constraint method directly."""

    def setUp(self):
        self.n = 2
        self.N = 20
        _, self.trainer, self.data = _make_portfolio_problem(self.n, self.N)
        self.trainer.settings = _base_settings(self.data, self.n, self.N)

    def test_relu_positive(self):
        """Relu should pass through positive values."""
        self.trainer.settings.constraint_smoothing = "relu"
        h = torch.tensor([0.5, 1.0, 2.0])
        result = self.trainer._smooth_constraint(h)
        npt.assert_allclose(result.numpy(), [0.5, 1.0, 2.0])

    def test_relu_negative(self):
        """Relu should zero out negative values."""
        self.trainer.settings.constraint_smoothing = "relu"
        h = torch.tensor([-1.0, -0.5, 0.0])
        result = self.trainer._smooth_constraint(h)
        npt.assert_allclose(result.numpy(), [0.0, 0.0, 0.0])

    def test_softplus_positive(self):
        """Softplus should be close to identity for large positive values."""
        self.trainer.settings.constraint_smoothing = "softplus"
        self.trainer.settings.softplus_beta = 10.0
        h = torch.tensor([2.0, 5.0])
        result = self.trainer._smooth_constraint(h)
        npt.assert_allclose(result.numpy(), h.numpy(), atol=0.1)

    def test_softplus_negative(self):
        """Softplus should be small but positive for negative values."""
        self.trainer.settings.constraint_smoothing = "softplus"
        self.trainer.settings.softplus_beta = 10.0
        h = torch.tensor([-2.0, -5.0])
        result = self.trainer._smooth_constraint(h)
        # Should be > 0 (unlike relu which gives 0)
        self.assertTrue((result > 0).all())
        # Should be close to 0
        self.assertTrue((result < 0.1).all())

    def test_softplus_gradient_at_zero(self):
        """Softplus should have nonzero gradient at h=0 (unlike relu)."""
        self.trainer.settings.constraint_smoothing = "softplus"
        self.trainer.settings.softplus_beta = 10.0
        h = torch.tensor([0.0], requires_grad=True)
        result = self.trainer._smooth_constraint(h)
        result.backward()
        # Softplus gradient at 0 is sigmoid(0) = 0.5
        self.assertGreater(h.grad.item(), 0.0)

    def test_softplus_near_zero_nonzero(self):
        """Softplus at h=-0.01 should still give nonzero value (gradient signal)."""
        self.trainer.settings.constraint_smoothing = "softplus"
        self.trainer.settings.softplus_beta = 10.0
        h = torch.tensor([-0.01], requires_grad=True)
        result = self.trainer._smooth_constraint(h)
        result.backward()
        self.assertGreater(result.item(), 0.0)
        self.assertGreater(h.grad.item(), 0.0)


class TestClassicStrategy(unittest.TestCase):
    """Test that classic strategy still works (backward compatibility)."""

    def setUp(self):
        self.n = 2
        self.N = 20
        _, self.trainer, self.data = _make_portfolio_problem(self.n, self.N)

    def test_classic_trains(self):
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "classic"
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)
        self.assertGreater(len(result.df), 0)

    def test_classic_mu_is_scalar(self):
        """Classic strategy should keep mu as a scalar float."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "classic"
        result = self.trainer.train(settings=settings)
        mu_val = result.df["mu"].iloc[-1]
        self.assertIsInstance(mu_val, float)

    def test_classic_with_softplus(self):
        """Classic strategy with softplus smoothing should complete."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "classic"
        settings.constraint_smoothing = "softplus"
        settings.softplus_beta = 10.0
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)


class TestPIStrategy(unittest.TestCase):
    """Test PI controller dual-update strategy."""

    def setUp(self):
        self.n = 2
        self.N = 20
        _, self.trainer, self.data = _make_portfolio_problem(self.n, self.N)

    def test_pi_trains(self):
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pi"
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)
        self.assertGreater(len(result.df), 0)

    def test_pi_lambda_nonneg(self):
        """PI controller should keep lambda >= 0."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pi"
        result = self.trainer.train(settings=settings)
        for lam_arr in result.df["lam_list"]:
            self.assertTrue(np.all(lam_arr >= 0))

    def test_pi_mu_is_scalar(self):
        """PI strategy keeps mu scalar (only classic mu update as fallback)."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pi"
        result = self.trainer.train(settings=settings)
        mu_val = result.df["mu"].iloc[-1]
        self.assertIsInstance(mu_val, float)

    def test_pi_with_softplus(self):
        """PI + softplus should complete."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pi"
        settings.constraint_smoothing = "softplus"
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)

    def test_pi_custom_gains(self):
        """Training should work with custom PI gains."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pi"
        settings.pi_kp = 1.0
        settings.pi_ki = 0.5
        settings.pi_nu = 0.5
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)


class TestAdaptiveStrategy(unittest.TestCase):
    """Test adaptive penalty (PECANN-CAPU style) dual-update strategy."""

    def setUp(self):
        self.n = 2
        self.N = 20
        _, self.trainer, self.data = _make_portfolio_problem(self.n, self.N)

    def test_adaptive_trains(self):
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "adaptive"
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)
        self.assertGreater(len(result.df), 0)

    def test_adaptive_mu_is_vector(self):
        """Adaptive strategy should produce per-constraint mu (array)."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "adaptive"
        result = self.trainer.train(settings=settings)
        mu_val = result.df["mu"].iloc[-1]
        self.assertTrue(hasattr(mu_val, '__len__'),
                        f"mu should be an array, got {type(mu_val)}")

    def test_adaptive_mu_monotonic(self):
        """Adaptive mu should be monotonically non-decreasing per constraint."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "adaptive"
        result = self.trainer.train(settings=settings)
        mu_series = result.df["mu"]
        # Check that mu values are monotonically non-decreasing
        for i in range(1, len(mu_series)):
            prev = mu_series.iloc[i - 1]
            curr = mu_series.iloc[i]
            if hasattr(prev, '__len__') and hasattr(curr, '__len__'):
                self.assertTrue(np.all(curr >= prev - 1e-10),
                                f"mu decreased at step {i}: {prev} -> {curr}")

    def test_adaptive_lambda_nonneg(self):
        """Adaptive strategy should keep lambda >= 0."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "adaptive"
        result = self.trainer.train(settings=settings)
        for lam_arr in result.df["lam_list"]:
            self.assertTrue(np.all(lam_arr >= -1e-10))

    def test_adaptive_with_softplus(self):
        """Adaptive + softplus should complete."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "adaptive"
        settings.constraint_smoothing = "softplus"
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)

    def test_adaptive_custom_penalty_params(self):
        """Training should work with custom adaptive penalty params."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "adaptive"
        settings.penalty_ema_decay = 0.9
        settings.penalty_eta_scale = 2.0
        settings.penalty_eps = 1e-6
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)


class TestResetPrevCost(unittest.TestCase):
    """Test the reset_prev_cost_on_al_update setting."""

    def setUp(self):
        self.n = 2
        self.N = 20
        _, self.trainer, self.data = _make_portfolio_problem(self.n, self.N)

    def test_no_reset_trains(self):
        """Training should complete with prev_fin_cost reset disabled."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.reset_prev_cost_on_al_update = False
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)

    def test_no_reset_with_pi(self):
        """PI + no reset should complete."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pi"
        settings.reset_prev_cost_on_al_update = False
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)

    def test_no_reset_with_adaptive(self):
        """Adaptive + no reset should complete."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "adaptive"
        settings.reset_prev_cost_on_al_update = False
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)


class TestStrategyCrossSettings(unittest.TestCase):
    """Test combinations of strategies with other settings."""

    def setUp(self):
        self.n = 2
        self.N = 20
        _, self.trainer, self.data = _make_portfolio_problem(self.n, self.N)

    def test_all_strategies_same_seed_deterministic(self):
        """Each strategy should produce deterministic results with same seed."""
        for strategy in ("classic", "pi", "adaptive"):
            results = []
            for _ in range(2):
                _, trainer, data = _make_portfolio_problem(self.n, self.N)
                settings = _base_settings(data, self.n, self.N)
                settings.dual_update_strategy = strategy
                result = trainer.train(settings=settings)
                results.append(result.df["Lagrangian_val"].iloc[-1])
            npt.assert_allclose(
                results[0], results[1],
                err_msg=f"Strategy '{strategy}' not deterministic",
            )

    def test_pi_adam_optimizer(self):
        """PI strategy should work with Adam optimizer."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pi"
        settings.optimizer = "Adam"
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)

    def test_adaptive_adam_optimizer(self):
        """Adaptive strategy should work with Adam optimizer."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "adaptive"
        settings.optimizer = "Adam"
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)


if __name__ == "__main__":
    unittest.main()
