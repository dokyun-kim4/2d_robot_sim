"""
Kalman Filter implementation for the simulator. Assumes linear system dynamics and Gaussian noise. For the simulator, we are tracking the following states:

x = [x, y, theta]

We expect the following control inputs:

u = [v_x, v_y, w]
"""

import numpy as np
import random

from utils import wrap_angle


class KalmanFilter:
    """
    A class that implements the basic Kalman Filter algorithm, which assumes linear system dynamics and Gaussian noise.

    Attributes:
        dt: the length of each timestep, in seconds
        x: the state vector for the system we are estimating
        P: the process model, describing the uncertainty in our estimate
        F: the state transition matrix, describing how our state naturally changes from timestep to timestep
        B: the control input model, describing how control inputs affect each state variable in the state vector
        Q: the process noise, modeling unexpected disturbance in state transitions
    """

    def __init__(self, dt: float, prior: np.ndarray):
        """
        Initialize an instance of the KalmanFilter class.

        Args:
            dt: the length of each timestep, in seconds
            prior: the initial estimates for each state variable
        """
        self.DT: float = dt
        self.x_state: np.ndarray = np.array(prior.flatten(), dtype=np.float64)
        self.P: np.ndarray = np.eye(3)
        self.F: np.ndarray = np.eye(3)
        self.B: np.ndarray = np.eye(3) * dt
        self.Q: np.ndarray = self.get_Q()

    def predict(self, u: np.ndarray):
        """
        Predicts the next state vector and its covariance matrix using the state transition matrix and an input control vector. The Kalman Filter uses the following predict equations:

        x_t+1 = F * x_t + B * u_t
        P_t+1 = F * P * F.T + Q

        Args:
            u: the input control vector
        """

        self.x_state = (self.F @ self.x_state.reshape(-1, 1) + self.B @ u).flatten()
        self.x_state[2] = wrap_angle(
            self.x_state[2]
        )  # Ensure theta stays within [-pi, pi]
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x_state, self.P

    def update(self, z, H, R):
        """
        Updates the current state prediction using observations from the environment. The Kalman Filter uses the following update equations:

        x = x + K * y
        P = P - K * H * P

        Where K and y are given by the following:
        y = z - H * x (residual: error between observation and expected observation given estimated state vector)
        K = P * H.T * inv(S) (Kalman Gain: portion of total uncertainty that is from the prediction)
        S = H * P * H.T + R (total uncertainty in the system)

        Args:
            z: the given observation, AKA a measurement taken of the environment
            H: the measurement model, which relates the state space to the measurement space
            R: the measurement noise model (covariance)
        """
        S = H @ self.P @ H.T + R

        K = self.P @ H.T @ np.linalg.inv(S)

        y = z - H @ self.x_state.reshape(-1, 1)

        self.x_state += (K @ y).flatten()

        self.P -= K @ H @ self.P

        self.x_state[2] = wrap_angle(
            self.x_state[2]
        )  # Ensure theta stays within [-pi, pi]
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
