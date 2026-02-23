"""
Extended Kalman Filter implementation for the simulator. Tracks the following states:

x = [x, y, theta]

We expect the following control inputs:

u = [v, w]
"""

import numpy as np
import sympy as sp
from sympy import Matrix, cos, sin
from sympy.abc import x, y, theta, v, w
from utils import wrap_angle
import random


class ExtendedKalmanFilter:
    def __init__(self, dt: float, prior: np.ndarray):
        self.DT: float = dt
        # Keep x_state as a 1D float array: shape (3,)
        self.x_state: np.ndarray = np.array(prior.flatten(), dtype=np.float64)
        self.P: np.ndarray = np.eye(3)
        self.Q: np.ndarray = self.get_Q()

        # Process model; symbolically defined
        self.f_xu = Matrix(
            [
                [x + v * cos(theta) * self.DT],
                [y + v * sin(theta) * self.DT],
                [theta + w * self.DT],
            ]
        )

        # Compute Jacobian of f(x,u) symbolically
        self.F_symbolic = self.f_xu.jacobian(Matrix([x, y, theta]))

        # Convert symbolic f(x,u) and F to numeric functions, avoids numpy arrays that have Sympy Floats in them
        self.f_xu_numeric = sp.lambdify((x, y, theta, v, w), self.f_xu, "numpy")
        self.F_numeric = sp.lambdify((x, y, theta, v, w), self.F_symbolic, "numpy")

    def predict(self, u: np.ndarray):
        """
        Predict the next state and covariance based on the current state and control input.

        Args:
            u: control input, 1D array of shape (2,) with [v, w]

        Returns:
            x_state: predicted state, 1D array of shape (3,)
            P: predicted covariance
        """
        u = u.flatten()
        v_val, w_val = float(u[0]), float(u[1])
        x_val, y_val, theta_val = self.x_state

        # Execute compiled functions
        # f_numeric returns a (3,1) array, so we flatten back to (3,)
        self.x_state = self.f_xu_numeric(
            x_val, y_val, theta_val, v_val, w_val
        ).flatten()
        self.x_state[2] = wrap_angle(
            self.x_state[2]
        )  # Ensure theta stays within [-pi, pi]

        # F_numeric returns a (3,3) Jacobian matrix
        F_eval = self.F_numeric(x_val, y_val, theta_val, v_val, w_val)

        # Predict Covariance
        self.P = F_eval @ self.P @ F_eval.T + self.Q

        return self.x_state, self.P

    def update(
        self, H: np.ndarray, R: np.ndarray, z: np.ndarray = None, y: np.ndarray = None
    ):
        """
        Update the state and covariance based on a new measurement.

        Args:
            H: measurement Jacobian matrix
            R: measurement noise covariance
            z: observation vector
            y: residual (used for LandmarkPinger where y is pre-computed)
        """
        # Ensure H and R are numpy arrays of type float64 to avoid issues with Sympy Floats
        H = np.array(H, dtype=np.float64)
        R = np.array(R, dtype=np.float64)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        if y is None:
            # For linear sensors like GPS, compute residual as y = z - Hx
            # If sensor is non-linear use pre-computed y passed in as argument
            y = z - H @ (self.x_state.reshape(-1, 1))

        self.x_state += (K @ y).flatten()
        self.x_state[2] = wrap_angle(self.x_state[2])
        self.P = (np.eye(len(self.x_state)) - K @ H) @ self.P

        return self.x_state, self.P

    def get_Q(self):
        """
        Generate white noise to apply to the process model after each prediction.
        """
        stdev = 0.001
        return np.array(
            [
                [
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                ],
                [
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                ],
                [
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                ],
            ]
        )
