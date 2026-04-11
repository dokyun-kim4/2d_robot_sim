"""
A simulated robotic agent with teleoperation and sensing capabilities.

The Robot class models the robotic agent that explores the world. The robot is remote-controlled by angular and linear velocity commands read from an external file. The robot can execute motor commands to move, and can sense both externally (GPS, landmarks, obstacles) and internally (odometry, IMU).
"""

import math
import random
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from environment import Environment
from sensors import SensorInterface, WheelEncoder, GPS, LandmarkPinger, InsituInstrument
from utils import NEAR_ZERO, floating_mod_zero, DriveType
from enum import Enum
import copy


class Robot:
    """
    A class that models a simulated robotic agent.

    Attributes:
        env: the environment this robot is operating in
        sensors: list of all robot sensors
    """

    def __init__(
        self,
        env: Environment,
        sensor_info: dict,
        robot_info: dict,
        drive_type: DriveType,
    ):
        """
        Initialize an instance of the Robot class.

        Args:
            env: the environment this robot is operating in
        """
        self.drive_type = drive_type

        mtr_config = robot_info["Motor"]
        self.MTR_NOISE_LINEAR = mtr_config["linear_noise"]
        self.MTR_NOISE_ANGULAR = mtr_config["angular_noise"]

        # Save cmd vel and actual vel for spoofed sensor data
        if self.drive_type == DriveType.DIFFERENTIAL:
            self.latest_lin_vel_cmd = 0.0
            self.latest_ang_vel_cmd = 0.0
            self.latest_lin_vel_actual = 0.0
            self.latest_ang_vel_actual = 0.0
        else:
            self.latest_vx_cmd = 0.0
            self.latest_vy_cmd = 0.0
            self.latest_ang_vel_cmd = 0.0
            self.latest_vx_actual = 0.0
            self.latest_vy_actual = 0.0
            self.latest_ang_vel_actual = 0.0

        self.env = env
        gps_config = sensor_info["GPS"]
        encoder_config = sensor_info["WheelEncoder"]
        lm_pinger_config = sensor_info["LandmarkPinger"]
        inst_config = sensor_info["InsituInstrument"]
        self.sensors = [
            WheelEncoder(
                robot=self,
                interval=encoder_config["interval"],
                lin_noise=encoder_config["linear_noise"],
                ang_noise=encoder_config["angular_noise"],
                lin_noise_ratio=encoder_config["linear_noise_ratio"],
                ang_noise_ratio=encoder_config["angular_noise_ratio"],
            ),
            GPS(
                robot=self,
                interval=gps_config["interval"],
                x_noise=gps_config["x_noise"],
                y_noise=gps_config["y_noise"],
            ),
            LandmarkPinger(
                robot=self,
                max_range=self.env.lm_max_range,
                interval=lm_pinger_config["interval"],
                range_noise=lm_pinger_config["range_noise"],
                range_prop_noise=lm_pinger_config["range_prop_noise"],
                bearing_noise=lm_pinger_config["bearing_noise"],
            ),
            InsituInstrument(
                robot=self,
                interval=inst_config["interval"],
                noise=inst_config["noise"]
            )
        ]

        self.kernel = copy.deepcopy(self.env.continuous_field.kernel)  # the same as the environment kernel
        self.belief = GaussianProcessRegressor(kernel=self.kernel,
                                                n_restarts_optimizer=15,
                                                random_state=self.env.continuous_field.random_seed)
        self.pose_history = []  # store history of observation poses for belief
        self.observation_history = []  # store history of observations for belief
    

    def update_belief(self, measurement, pose):
        """
        Updates the robot's belief based on a located-observation.

        Inputs:
            measurement (float): value of the field being measured
            pose (Position): location from where the measurement was taken
        """
        self.pose_history.append((pose.pos.x, pose.pos.y))
        self.observation_history.append(measurement)
        self.belief.fit(np.asarray(self.pose_history), np.asarray(self.observation_history))

    def robot_step_differential(self, cmd: list[float]):
        """
        Differential-drive mode. Given forward linear and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            lin_vel: input linear velocity command
            ang_vel: input angular velocity command

        Returns:
            dx: change in x position
            dy: change in y position
            d-theta: change in heading
        """
        # Save commanded vel
        self.latest_lin_vel_cmd = cmd[0]
        self.latest_ang_vel_cmd = cmd[1]

        # Apply motor noise
        lin_vel = cmd[0] * (1 + random.gauss(0, self.MTR_NOISE_LINEAR))
        ang_vel = cmd[1] * (1 + random.gauss(0, self.MTR_NOISE_ANGULAR))

        self.latest_lin_vel_actual = lin_vel
        self.latest_ang_vel_actual = ang_vel

        dx = self.env.DT * lin_vel * math.cos(self.env.robot_pose.theta)
        dy = self.env.DT * lin_vel * math.sin(self.env.robot_pose.theta)
        dtheta = self.env.DT * ang_vel

        self.env.robot_step(dx, dy, dtheta)

    def robot_step_translational(self, cmd: list[float]):
        """
        Swerve-drive mode. Given x, y, and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            x_vel: input x velocity command
            y_vel: input y velocity command
            ang_vel: input angular velocity command

        Returns:
            dx: change in x position
            dy: change in y position
            d-theta: change in heading
        """
        # Save commanded vel
        self.latest_vx_cmd = cmd[0]
        self.latest_vy_cmd = cmd[1]
        self.latest_ang_vel_cmd = cmd[2]

        # Simulate noise in motors
        x_vel = cmd[0] * (1 + random.gauss(0, self.MTR_NOISE_LINEAR))
        y_vel = cmd[1] * (1 + random.gauss(0, self.MTR_NOISE_LINEAR))
        ang_vel = cmd[2] * (1 + random.gauss(0, self.MTR_NOISE_ANGULAR))

        dx = x_vel * self.env.DT
        dy = y_vel * self.env.DT
        dtheta = ang_vel * self.env.DT
        self.env.robot_step(dx, dy, dtheta)

        # save actual vel for sensors
        self.latest_vx_actual = x_vel
        self.latest_vy_actual = y_vel
        self.latest_ang_vel_actual = ang_vel

        return dx, dy, dtheta

    def take_sensor_measurements(self):
        """
        Return noisy sensor readings of the environment at this timestep, including data from all sensors, in a table format.
        """

        if self.drive_type == DriveType.DIFFERENTIAL:
            measurements = pd.DataFrame(
                {
                    "environment_time": [self.env.time],
                    # also store commanded velocities for plotting
                    "lin_vel_cmd": [self.latest_lin_vel_cmd],
                    "ang_vel_cmd": [self.latest_ang_vel_cmd],
                }
            )
        else:
            measurements = pd.DataFrame(
                {
                    "environment_time": [self.env.time],
                    # also store commanded velocities for plotting
                    "vx_cmd": [self.latest_vx_cmd],
                    "vy_cmd": [self.latest_vy_cmd],
                    "ang_vel_cmd": [self.latest_ang_vel_cmd],
                }
            )

        for sensor in self.sensors:
            if floating_mod_zero(self.env.time, sensor.interval):
                measurements = pd.merge(
                    measurements, sensor.sample(), left_index=True, right_index=True
                )

        return measurements

    def sensor_info(self):
        """
        Return a dictionary of frozen environment information.
        """
        # set up the table
        columns = ["Sensor Name", "Constant Noise", "Proportional Noise", "Model"]
        data = []

        # start with controller
        name = "MotorController"
        # linear
        lin_row = pd.DataFrame(
            0,
            index=pd.RangeIndex(1),
            columns=columns,
        )
        lin_row["Sensor Name"] = name + f"Linear"
        lin_row["Constant Noise"] = self.MTR_NOISE_LINEAR
        data.append(lin_row)
        # angular
        ang_row = pd.DataFrame(
            0,
            index=pd.RangeIndex(1),
            columns=columns,
        )
        ang_row["Sensor Name"] = name + f"Angular"
        ang_row["Constant Noise"] = self.MTR_NOISE_ANGULAR
        data.append(ang_row)

        # add the belief
        belief_row = pd.DataFrame(
            0,
            index=pd.RangeIndex(1),
            columns=columns,
        )
        belief_row["Sensor Name"] = "belief"
        belief_row["Model"] = self.belief # type: ignore
        data.append(belief_row)

        # iterate through sensors
        for sensor in self.sensors:
            name = sensor.name
            # GPS has x noise and y noise
            if isinstance(sensor, GPS):
                # x
                row = pd.DataFrame(
                    0,
                    index=pd.RangeIndex(1),
                    columns=columns,
                )
                row["Sensor Name"] = name + "X"
                row["Constant Noise"] = sensor.X_NOISE
                data.append(row)
                # y
                row = pd.DataFrame(
                    0,
                    index=pd.RangeIndex(1),
                    columns=columns,
                )
                row["Sensor Name"] = name + "Y"
                row["Constant Noise"] = sensor.Y_NOISE
                data.append(row)
            # odom has linear and angular components to consider
            elif isinstance(sensor, WheelEncoder):
                # linear
                lin_row = pd.DataFrame(
                    0,
                    index=pd.RangeIndex(1),
                    columns=columns,
                )
                lin_row["Sensor Name"] = name + f"Linear"
                lin_row["Constant Noise"] = sensor.LIN_NOISE
                lin_row["Proportional Noise"] = sensor.LIN_NOISE_RATIO
                data.append(lin_row)
                # angular
                ang_row = pd.DataFrame(
                    0,
                    index=pd.RangeIndex(1),
                    columns=columns,
                )
                ang_row["Sensor Name"] = name + f"Angular"
                ang_row["Constant Noise"] = sensor.ANG_NOISE
                ang_row["Proportional Noise"] = sensor.ANG_NOISE_RATIO
                data.append(ang_row)
            # pinger has independent range and bearing
            elif isinstance(sensor, LandmarkPinger):
                # linear
                range_row = pd.DataFrame(
                    0,
                    index=pd.RangeIndex(1),
                    columns=columns,
                )
                range_row["Sensor Name"] = name + f"Range"
                range_row["Constant Noise"] = sensor.RANGE_PROP_NOISE
                range_row["Proportional Noise"] = sensor.RANGE_PROP_NOISE
                data.append(range_row)
                # angular
                bearing_row = pd.DataFrame(
                    0,
                    index=pd.RangeIndex(1),
                    columns=columns,
                )
                bearing_row["Sensor Name"] = name + f"Angular"
                bearing_row["Constant Noise"] = sensor.BEARING_NOISE
                data.append(bearing_row)
            elif isinstance(sensor, InsituInstrument):
                instrument_row = pd.DataFrame(
                    0,
                    index=pd.RangeIndex(1),
                    columns=columns,
                )
                instrument_row["Sensor Name"] = name
                instrument_row["Constant Noise"] = sensor.noise
                data.append(instrument_row)

        # return
        return pd.concat(data)
