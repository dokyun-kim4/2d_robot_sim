"""
Extended Kalman Filter implementation for the simulator. Tracks the following states:

x = [x, y, theta]

We expect the following control inputs:

u = [v, w]
"""

import numpy as np
import sympy
from sympy.abc import x, y, v, w, R, theta
from sympy import Matrix, Symbol
import random

from utils import wrap_angle


class ExtendedKalmanFilter:
    """
    This class implements the Extended Kalman Filter algorithm.
    """

    def __init__(self, dt: float, prior: np.ndarray):
        """
        Initialize an Extended Kalman Filter.

        A state vector includes the following:
            x position
            y position
            heading


        Args:
            dt: the length of each timestep, in seconds
            prior: the initial estimates for each state variable-
        """
        # TODO: set the timestep size to the given parameter
        self.DT: float = None

        # TODO: set the state vector to the given prior
        self.x: np.ndarray = None

        # TODO: set the process model to an identity matrix
        self.P: np.ndarray = None

        # TODO: define the nonlinear state transition model
        self.f_xu: Matrix = Matrix(
            [
                [None],  # calculation of x
                [None],  # calculation of y
                [None],  # calculation of theta
            ]
        )

        # TODO: define the Jacobian of the motion model symbolically
        self.F: Matrix = None

        # dictionary that maps Sympy symbols to numerical values. we will use these to substitute values into our symbolic matrices!
        self.subs: dict[Symbol, float] = {
            x: self.x_state[0],
            y: self.x_state[1],
            theta: self.x_state[2],
            v: 0,
            w: 0,
        }

    def predict(self, u: np.ndarray):
        """
        Predicts the next state vector and its covariance matrix using the state transition matrix and an input control vector. The Kalman Filter uses the following predict equations:

        x_t+1 = f(x,u)
        P_t+1 = F * P * F.T + Q

        where F is the Jacobian of f(x,u)

        Args:
            u: the input control vector
        """
        # TODO: set the value of each symbolic substitution to the actual numerical value being tracked by the EKF
        self.subs[x] = None
        self.subs[y] = None
        self.subs[theta] = None
        self.subs[v] = None
        self.subs[w] = None

        # TODO: evaluate the nonlinear motion model f(x,u) at the subsitution values
        fxu_eval = None

        # TODO: evaluate the Jacobian matrix F at the substitution values
        F_eval = None

        # TODO: calculate the next state prediction
        self.x = None

        # TODO: calculate the next covariance prediction
        self.P = None

        # return state vector and state covariance
        return self.x_state, self.P

    def update(
        self,
        H: np.ndarray,
        R: np.ndarray,
        z: np.ndarray | None,
        y: np.ndarray | None,
    ):
        """
        Updates the current state prediction using observations from the environment. The Extended Kalman Filter uses the following update equations:

        x = x + K * y
        P = P - K * H * P

        Where K and y are given by the following:
        y = z - h(x) (residual: error between observation and expected observation given estimated state vector)
        K = P * H.T * inv(S) (Kalman Gain: portion of total uncertainty that is from the prediction)
        S = H * P * H.T + R (total uncertainty in the system)

        where H is the Jacobian of h(x)

        Args:
            H: the Jacobian of the nonlinear measurement model, which relates the state space to the measurement space
            R: the measurement noise model (covariance)
            y: the residual, which is the error between the measured observation and the observation expected by the predicted state
        """
        # TODO: calculate the total uncertainty in the system
        S = None

        # TODO: calculate the Kalman Gain
        K = None

        if y is None:
            y = z - H @ self.x_state

        # TODO: update state vector
        self.x_state = None

        # TODO: update process model
        self.P = None

        # return state vector and process model
        return self.x_state, self.P

    def get_Q(self):
        """
        Generate white noise to apply to the process model after each prediction.
        """
        # TODO: explore different standard deviation values for this function!
        stdev = 0.1
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
