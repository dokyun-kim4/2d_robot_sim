"""
A simulated robotic agent with teleoperation and sensing capabilities.

The Robot class models the robotic agent that explores the world. The robot is remote-controlled by angular and linear velocity commands read from an external file. The robot can execute motor commands to move, and can sense both externally (GPS, landmarks, obstacles) and internally (odometry, IMU).
"""
import math
import random
from environment import Environment
from sensors import SensorInterface, WheelEncoder


class Robot:
    """
    A class that models a simulated robotic agent.

    Attributes:
        env: the environment this robot is operating in
        sensors: list of all robot sensors
    """

    def __init__(self, env: Environment):
        """
        Initialize an instance of the Robot class.

        Args:
            env: the environment this robot is operating in
        """
        # Save cmd vel and actual vel for spoofed sensor data
        self.latest_lin_vel_cmd = 0.0
        self.latest_ang_vel_cmd = 0.0
        self.latest_lin_vel_actual = 0.0
        self.latest_ang_vel_actual =  0.0


        self.env = env
        self.sensors = [WheelEncoder(self)]
        self.MTR_NOISE_LINEAR = 0.05
        self.MTR_NOISE_ANGULAR = 0.01

    def robot_step_differential(self, lin_vel: float, ang_vel: float):
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
        self.latest_lin_vel_cmd = lin_vel
        self.latest_ang_vel_cmd = ang_vel

        # Apply motor noise
        lin_vel = random.gauss(lin_vel, self.MTR_NOISE_LINEAR)
        ang_vel = random.gauss(ang_vel, self.MTR_NOISE_ANGULAR)

        self.latest_lin_vel_actual = lin_vel
        self.latest_ang_vel_actual = ang_vel

        # If no angular velocity
        if ang_vel == 0:
            dtheta = 0
            dx = self.env.DT * lin_vel * math.cos(self.env.robot_pose.theta)
            dy = self.env.DT * lin_vel * math.sin(self.env.robot_pose.theta)
        else:
        # Robot drives in arc, find radius via r = v/w
            r = lin_vel/ang_vel
            dtheta = ang_vel * self.env.DT
            dx = r*(math.sin(self.env.robot_pose.theta + dtheta) - math.sin(self.env.robot_pose.theta))
            dy = -r*(math.cos(self.env.robot_pose.theta + dtheta) - math.cos(self.env.robot_pose.theta))
        
        self.env.robot_step(dx, dy, dtheta)

    def robot_step_translational(self, x_vel: float, y_vel: float, ang_vel: float):
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
        # Simulate noise in motors
        x_vel *= 1 + random.gauss(0, self.MTR_NOISE_LINEAR)
        y_vel *= 1 + random.gauss(0, self.MTR_NOISE_LINEAR)
        ang_vel *= 1 + random.gauss(0, self.MTR_NOISE_ANGULAR)

        dx = x_vel * self.env.DT
        dy = y_vel * self.env.DT
        dtheta = ang_vel * self.env.DT
        self.env.robot_step(dx,dy,dtheta)

        return dx, dy, dtheta

    def take_sensor_measurements(self):
        """
        Return noisy sensor readings of the environment at this timestep, including data from all sensors, in a table format.
        """
        measurements = []
        for sensor in self.sensors:
            if (self.env.time % sensor.interval) == 0:
                sensor_data = {
                                "name": sensor.name,
                                "time": self.env.time,
                                "measurement": sensor.sample
                               }
                measurements.append(sensor_data)
        return measurements
