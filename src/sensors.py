"""
An abstract base class that all sensor classes must inherit from. This structure guarantees that all sensors have certain traits, including a name, sampling interval, and sampling function.

In addition to basic features, all sensors should have noise constants. Different sensors may use different distributions to model noise, and may take in different parameters to shape that noise. For example, one sensor might have a constant noise mean, while another might have noise that grows proportionally with distance or time.

Exteroceptive sensors measure the robot's relationship to the world. This includes GPS, cameras, LiDAR, and anything else that takes a measurement that can relate the robot's state to things beyond the robot.

Proprioceptive sensors measure the robot's relationship to its past states. This includes IMUs, wheel encoders, and anything else that measures how the robot's state is relatively changing, without relating the robot to the world.
"""

from abc import ABC, abstractmethod
from math import pi, inf
import random
# from robot import Robot
import pandas as pd
from utils import Position, Pose, Bounds, Landmark, BearingRange

import numpy as np

import sympy
from sympy.abc import x, y, k, j, theta
from sympy import symbols, Matrix, Symbol, pprint


class SensorInterface(ABC):
    """
    A basic Interface to standardize all sensors.

    Attributes:
        name: string identifier
        robot: reference robot. required for observing the environment
        interval: period between measurements
        last_meas_t: time of last sensor measurement
    """

    def __init__(self, name: str, robot, interval: float):
        """
        Initialize a sensor class instace.

        Args:
            name: reference identifier
            robot: reference robot
            interval: period between measurements
        """
        self._name = name
        self.robot = robot
        self._interval = interval
        self.last_meas_t = robot.env.time

    @property
    def name(self) -> str:
        """
        Getter for the name property.
        """
        return self._name

    @property
    def interval(self) -> float:
        """
        Getter for the interval property.
        """
        return self._interval

    @property
    def last_meas_t(self) -> float:
        """
        Getter for the time of last measurement property.
        """
        return self._last_meas_t

    @last_meas_t.setter
    def last_meas_t(self, value: float):
        """
        Setter for the time of last measurement property.
        """
        self._last_meas_t = value

    @abstractmethod
    def sample(self):
        """
        Sample the environment and return the noisy measurement(s).
        """
        pass


class WheelEncoder(SensorInterface):
    """
    This class represents a wheel encoder set that measures the robot's motor speeds.
    Reports noisy estimates of linear and angular velocities.

    Attributes:
        name: string identifier
        robot: reference robot
        interval: period between measurements
        last_meas_t: time of last measurement
        LIN_NOISE: absolute noise for linear velocity stdev
        ANG_NOISE: absolute noise for angular velocity stdev
    """

    def __init__(
        self,
        robot,
        name="wheel_encoder",
        interval=0.1,
        lin_noise=0.05,
        ang_noise=0.03,
    ):
        """
        Initialize an instance of the WheelEncoder class.

        Args:
            robot: reference robot
            name: reference identifier
            interval: period between measurements
            linear_noise_ratio: proportional noise for linear velocity
            angular_noise_ratio: proportional noise for angular
        """
        super().__init__(name, robot, interval)
        self.LIN_NOISE = lin_noise  # m/s
        self.ANG_NOISE = ang_noise  # rad/s


    def sample(self) -> pd.DataFrame:
        """
        Sample the robot's linear and angular velocity.
        """
        measurement = pd.DataFrame()
        measurement[f"{self.name}_lin_vel_actual"] = [random.gauss(self.robot.latest_lin_vel_actual, self.LIN_NOISE)]
        measurement[f"{self.name}_ang_vel_actual"] = [random.gauss(self.robot.latest_ang_vel_actual, self.ANG_NOISE)]
        return measurement


class LandmarkPinger(SensorInterface):
    """
    This class represents a sensor that measures the range and bearing between the robot and the floating-point landmarks on the map. In practice, this sensor could be a ToF sensor, a node in a network of beacons, or even a camera.

    Attributes:
        name: reference identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        MAX_RANGE (int): maximum distance from a beacon for it to be visible
        RANGE_NOISE (float): absolute noise for range stdev
        RANGE_NOISE_RATIO (float): porportional noise for range stdev
        BEARING_NOISE (float): absolute noise for bearing stdev
    """

    def __init__(
        self,
        robot,
        name="landmark_pinger",
        interval=1.0,
        range_noise=0.5,
        range_prop_noise=0.05,
        bearing_noise=pi / 6,
    ):
        """
        Initialize an instance of the LandmarkPinger class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
        """
        super().__init__(name, robot, interval)
        self.MAX_RANGE = robot.env.lm_max_range  # meters
        self.RANGE_NOISE = range_noise  # meters
        self.RANGE_PROP_NOISE = range_prop_noise
        self.BEARING_NOISE = bearing_noise  # radians

    def sample(self) -> pd.DataFrame:
        # TODO: define the nonlinear measurement model symbolically
        self.h_x: Matrix = Matrix(
            [
                [None],  # calculation of r (range)
                [None],  # calculation of phi (bearing)
            ]
        )

        # TODO: define the Jacobian of h(x) symbolically
        self.H: Matrix = None

        self.subs: dict[Symbol, float] = {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
            k: 0.0,
            j: 0.0,
        }

    def sample(self):
        """
        Reports noisy measurements of the bearing and range between the robot and all nearby landmarks.
        """
        measurements = pd.DataFrame()
        landmarks = self.robot.env.get_proximity_to_landmarks()
        for lm in landmarks.columns:
            br: BearingRange = landmarks[lm].values[0]
            # calculate combined noise of constant noise + proportional noise
            total_range_noise = self.RANGE_NOISE + self.RANGE_PROP_NOISE*br.range
            # landmarks out of range = infinite range & bearing
            if br.range > self.MAX_RANGE:
                measurements[f"{self.name}_{lm}"] = [(BearingRange(br.landmark_id, inf, inf))]
            else:
                noisy_bearing = random.gauss(br.bearing, self.BEARING_NOISE)
                noisy_range = random.gauss(br.range, total_range_noise)
                measurements[f"{self.name}_{lm}"] = [BearingRange(br.landmark_id, noisy_bearing, noisy_range)]

        return measurements
        # TODO: define the nonlinear measurement model symbolically
        self.h_x: Matrix = Matrix(
            [
                [None],  # calculation of r (range)
                [None],  # calculation of phi (bearing)
            ]
        )

        # TODO: define the Jacobian of h(x) symbolically
        self.H: Matrix = None

        self.subs: dict[Symbol, float] = {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
            k: 0.0,
            j: 0.0,
        }

    def R(self, z):
        """
        Estimate variance of a given pinger measurement.

        Args:
            z (ndarray): pinger observation [[range 0], [0 bearing]]

        Returns:
            Sensor noise model for pinger measurement
        """
        bearing_stdev = self.BEARING_NOISE
        range_stdev = self.RANGE_NOISE + z[0] * self.RANGE_PROP_NOISE
        return np.diag([range_stdev, bearing_stdev]) ** 2

    def H_eval(self, x, lm_id):
        """
        Evaluate the Jacobian of h(x) at x, which reshapes a state vector to be in the observation space. This matrix is used to turn a state prediction into an observation prediction for a specific landmark.

        Args:
            x: the current state vector, to linearize with respect to
            lm_id: the ID of the landmark that we are predicting an observation of
        """
        # TODO: find the x and y position of the given landmark
        lm_x = None
        lm_y = None

        # TODO: set the value of each symbolic substitution to the actual numerical value that was passed in
        self.subs[x] = None
        self.subs[y] = None
        self.subs[theta] = None
        self.subs[j] = None  # note: we use j for landmark x position
        self.subs[k] = None  # note: we use k for landmark y position

        # TODO: evaluate the Jacobian at the subs values and convert it to a numpy array
        H_eval = None

        # return
        return H_eval

    def y(self, z, x, lm_id):
        """
        Calculate the residual between an observation x and a predicted observation derived from a predicted state. The predicted observation is in reference to a specified landmark.
        """
        # TODO: find the x and y position of the given landmark
        lm_x = None
        lm_y = None

        # TODO: set the value of each symbolic substitution to the actual numerical value that was passed in
        self.subs[x] = None
        self.subs[y] = None
        self.subs[theta] = None
        self.subs[j] = None  # note: we use j for landmark x position
        self.subs[k] = None  # note: we use k for landmark y position

        # TODO: evaluate the measurement model at the subs values and convert it to a numpy array
        hx_eval = None

        # TODO: calculate the residual
        y = None

        # return
        return y


class GPS(SensorInterface):
    """
    This class represents a GPS sensor that measures the position of the robot in 2D space.

    Attributes:
        name (str): string identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        last_meas_t (float): time of last measurement
        X_NOISE (float): absolute noise for x stdev
        Y_NOISE (float): absolute noise for y stdev
    """

    def __init__(
        self,
        robot,
        name,
        interval,
        x_noise,
        y_noise,
    ):
        """
        Initialize an instance of the GPS class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
            x_noise (float): absolute noise for x stdev
            y_noise (float): absolute noise for y stdev
        """
        super().__init__(name, robot, interval)
        self.X_NOISE = x_noise
        self.Y_NOISE = y_noise

        # TODO: fill in the measurement model
        self.H = None

        # TODO: fill in the noise model
        self.R = None

    def sample(self):
        """
        Take a noisy GPS measurement of robot position.
        """
        measurements = pd.DataFrame()
        landmarks = self.robot.env.get_proximity_to_landmarks()
        for lm in landmarks.columns:
            br: BearingRange = landmarks[lm].values[0]
            # calculate combined noise of constant noise + proportional noise
            total_range_noise = self.RANGE_NOISE + self.RANGE_PROP_NOISE*br.range
            # landmarks out of range = infinite range & bearing
            if br.range > self.MAX_RANGE:
                measurements[f"{self.name}_{lm}"] = [(BearingRange(br.landmark_id, inf, inf))]
            else:
                noisy_bearing = random.gauss(br.bearing, self.BEARING_NOISE)
                noisy_range = random.gauss(br.range, total_range_noise)
                measurements[f"{self.name}_{lm}"] = [BearingRange(br.landmark_id, noisy_bearing, noisy_range)]

        return measurements

    def R(self, z):
        """
        Estimate variance of a given pinger measurement.

        Args:
            z (ndarray): pinger observation [[range 0], [0 bearing]]

        Returns:
            Sensor noise model for pinger measurement
        """
        bearing_stdev = self.BEARING_NOISE
        range_stdev = self.RANGE_NOISE + z[0] * self.RANGE_PROP_NOISE
        return np.diag([range_stdev, bearing_stdev]) ** 2

    def H_eval(self, x, lm_id):
        """
        Evaluate the Jacobian of h(x) at x, which reshapes a state vector to be in the observation space. This matrix is used to turn a state prediction into an observation prediction for a specific landmark.

        Args:
            x: the current state vector, to linearize with respect to
            lm_id: the ID of the landmark that we are predicting an observation of
        """
        # TODO: find the x and y position of the given landmark
        lm_x = None
        lm_y = None

        # TODO: set the value of each symbolic substitution to the actual numerical value that was passed in
        self.subs[x] = None
        self.subs[y] = None
        self.subs[theta] = None
        self.subs[j] = None  # note: we use j for landmark x position
        self.subs[k] = None  # note: we use k for landmark y position

        # TODO: evaluate the Jacobian at the subs values and convert it to a numpy array
        H_eval = None

        # return
        return H_eval

    def y(self, z, x, lm_id):
        """
        Calculate the residual between an observation x and a predicted observation derived from a predicted state. The predicted observation is in reference to a specified landmark.
        """
        # TODO: find the x and y position of the given landmark
        lm_x = None
        lm_y = None

        # TODO: set the value of each symbolic substitution to the actual numerical value that was passed in
        self.subs[x] = None
        self.subs[y] = None
        self.subs[theta] = None
        self.subs[j] = None  # note: we use j for landmark x position
        self.subs[k] = None  # note: we use k for landmark y position

        # TODO: evaluate the measurement model at the subs values and convert it to a numpy array
        hx_eval = None

        # TODO: calculate the residual
        y = None

        # return
        return y


class GPS(SensorInterface):
    """
    This class represents a GPS sensor that measures the position of the robot in 2D space.

    Attributes:
        name (str): string identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        last_meas_t (float): time of last measurement
        X_NOISE (float): absolute noise for x stdev
        Y_NOISE (float): absolute noise for y stdev
    """

    def __init__(
        self,
        robot,
        name,
        interval,
        x_noise,
        y_noise,
    ):
        """
        Initialize an instance of the GPS class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
            x_noise (float): absolute noise for x stdev
            y_noise (float): absolute noise for y stdev
        """
        super().__init__(name, robot, interval)
        self.X_NOISE = x_noise
        self.Y_NOISE = y_noise

        # TODO: fill in the measurement model
        self.H = None

        # TODO: fill in the noise model
        self.R = None

    def sample(self):
        """
        Take a noisy GPS measurement of robot position.
        """
        # TODO: fill in the function
        pass
