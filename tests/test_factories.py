"""
Unit tests for tests/factories.py.

Run with: python -m unittest tests.test_factories
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.factories import create_estimator
from utils.history import HistoryBuffer


def constant_objective(_x):
    return 0.0


class CreateEstimatorTests(unittest.TestCase):
    def test_finite_difference_estimators_use_gdtcalcstep_as_step(self):
        for name in ("ffd", "cfd"):
            with self.subTest(name=name):
                estimator = create_estimator(
                    name,
                    constant_objective,
                    dims=2,
                    history=HistoryBuffer(),
                    gdtcalcstep=2.5e-4,
                )

                self.assertEqual(estimator.step, 2.5e-4)

    def test_gaussian_smoothing_estimators_use_gdtcalcstep_as_radius(self):
        for name in ("gsg", "cgsg"):
            with self.subTest(name=name):
                estimator = create_estimator(
                    name,
                    constant_objective,
                    dims=2,
                    history=HistoryBuffer(),
                    gdtcalcstep=2.5e-4,
                )

                self.assertEqual(estimator.u, 2.5e-4)

    def test_nmxfd_uses_gdtcalcstep_as_sigma(self):
        estimator = create_estimator(
            "nmxfd",
            constant_objective,
            dims=2,
            history=HistoryBuffer(),
            gdtcalcstep=2.5e-4,
        )

        self.assertEqual(estimator.sigma, 2.5e-4)

    def test_sage_uses_gdtcalcstep_as_default_init_step(self):
        estimator = create_estimator(
            "sage",
            constant_objective,
            dims=2,
            history=HistoryBuffer(),
            gdtcalcstep=2.5e-4,
        )

        self.assertEqual(estimator.init_step, 2.5e-4)

    def test_explicit_sage_init_step_overrides_gdtcalcstep(self):
        estimator = create_estimator(
            "sage",
            constant_objective,
            dims=2,
            history=HistoryBuffer(),
            gdtcalcstep=2.5e-4,
            init_step=1.5e-3,
        )

        self.assertEqual(estimator.init_step, 1.5e-3)

    def test_sage_forwards_noise_bound(self):
        estimator = create_estimator(
            "sage",
            constant_objective,
            dims=2,
            history=HistoryBuffer(),
            noise_bound=0.42,
        )

        self.assertEqual(estimator.noise_bound, 0.42)
        self.assertTrue(estimator.noise_bound_is_fixed)
        self.assertEqual(estimator.ns_est, 0.42)

    def test_fixed_parameter_aliases_are_unsupported(self):
        removed_names = ("ffd1.0", "cfd1.0", "gsg1.0", "cgsg1.0", "nmxfd1.0")
        for name in removed_names:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    create_estimator(
                        name,
                        constant_objective,
                        dims=2,
                        history=HistoryBuffer(),
                    )


if __name__ == "__main__":
    unittest.main()
