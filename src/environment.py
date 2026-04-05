"""
A simulation environment for a mobile robot operating in two dimensions.

The Environment class models the world that the robots navigate in. The world is continuous and two-dimensional. The world possesses an outer border, internal obstacles, and identifiable landmarks. The world also manages the passage of time and the motion of robotic agents within the world over time.

Critically, the environment tracks the robot's state. In this case, the robot's state is a vector that includes three state variables: x position, y position, and heading.
"""

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from itertools import product
import pickle
import math
from utils import Position, Pose, Bounds, Landmark, BearingRange

class Field:
    """Creates a continuous function that can be sampled.
    
    Attributes:
        DIMS (Bounds): the four corners of the environment
        variance (float): the variance of the GP kernel
        lengthscale (float): the lengthscale of the GP kernel
        random_seed (int): random seed for setting world draw
    """
    def __init__(
        self,
        dimensions: Bounds,
        variance: float = 0.1,
        lengthscale: float = 1.0,
        random_seed: int = 10,
    ):
        """
        Initialize the continuous field in an environment.

        Args:
            dimensions (Bounds): the four corners of the environment
            variance (float): the variance of the GP kernel
            lengthscale (float): the lengthscale of the GP kernel
            random_seed (int): random seed for setting consistent world draw
        """
        self.DIMS = dimensions
        self.variance = variance
        self.lengthscale = lengthscale
        self.random_seed = random_seed
        self._initialize_field()

    def _initialize_field(self):
        """
        Initializes the continuous field in an environment.

        The field is represented as a Gaussian Process with an RBF kernel. The field is bounded to a 20 x 20 grid.
        We sample one point in the field, and then fit the GP to this initial field.
        This allows us to have a continuous function that can be sampled at any point in the environment.
        """
        self.kernel = ConstantKernel(1.0, (1e-3, 1e-3)) * RBF([self.lengthscale, self.lengthscale], (self.variance, 100*self.variance))
        gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=15, random_state=self.random_seed)
        x, y = np.linspace(self.DIMS.x_min, self.DIMS.x_max, 20), np.linspace(self.DIMS.y_min, self.DIMS.y_max, 20)

        # Get all combinations of x and y coordinates
        M = np.array(list(product(x, y)))

        # Sample one point in the field and fit the GP to this initial field
        init_sample = gp.sample_y(M, 1, random_state=self.random_seed)
        gp.fit(M, init_sample)
        self.field = gp

    def info(self) -> dict:
        """
        Returns the current state of the field in addition to its parameters
        """
        return {"Variance": self.variance,
                "Lengthscale": self.lengthscale,
                "Random Seed": self.random_seed,
                "Model": self.field}

class Environment:
    """
    A class that models the world simulation environment and the robot's state.

    Attributes:
        dimensions: the horizontal and vertical size of the world
        dt: the length of each timestep, in seconds
        obstacles: a list of obstacles
        landmarks: a list of landmarks
        robot_pose: the position and heading of the robot in the world
    """

    def __init__(
        self,
        dimensions: Bounds,
        dt: float,
        obstacles: list[Bounds],
        landmarks: list[Landmark],
        field: Field,
        lm_max_range: float,
        robot_starting_pose: Pose,
    ):
        """
        Initialize an instance of the Environment class.

        Args:
            dimensions: the horizontal and vertical size of the world
            dt: the length of each timestep, in seconds
            obstacles: a list of obstacles
            landmarks: a list of landmarks
            field: a continous function representing the sampling environment
            lm_max_range: maximum range of landmark pingers
            robot_starting_pose: the initial position and heading of the robot
            
        """
        self.DIMENSIONS = dimensions
        self.DT = dt
        self.time = 0
        self.OBSTACLES = obstacles
        self.LANDMARKS = landmarks
        self.lm_max_range = lm_max_range
        self.continuous_field = field
        self.robot_pose = robot_starting_pose

    def robot_step(self, dx: float, dy: float, dtheta: float) -> None:
        """
        Update the robot's position and heading in the world. The robot should not be able to pass through obstacles or outside of the world bounds.

        Args:
            dx: change in x position
            dy: change in y position
            dtheta: change in heading

        Returns:
            Nothing, but update the robot_pose property at the end
        """
        dx, dy = self.validate_xy_motion(dx, dy)
        self.robot_pose.pos.x += dx
        self.robot_pose.pos.y += dy
        new_theta = self.robot_pose.theta + dtheta
        self.robot_pose.theta = (new_theta + math.pi) % (2 * math.pi) - math.pi
        self.time += self.DT

    def validate_xy_motion(self, dx: float, dy: float) -> tuple[float, float]:
        """
        Given attempted x and y motion by the robot, determine what motion is physically possible (i.e. doesn't go through any obstacles or barriers). Return the actual motion that will be executed.

        Args:
            dx: attempted change in x position
            dy: attempted change in y position

        Returns:
            dx: change in x position that should be executed
            dy: change in y position that should be executed
        """
        dx_valid = (
            dx
            if self.is_valid_position(
                Position((self.robot_pose.pos.x + dx), self.robot_pose.pos.y)
            )
            else 0
        )
        dy_valid = (
            dy
            if self.is_valid_position(
                Position((self.robot_pose.pos.x), self.robot_pose.pos.y + dy)
            )
            else 0
        )

        return dx_valid, dy_valid

    def is_valid_position(self, position: Position) -> bool:
        """
        Check if a given robot position is valid; i.e. not out-of-bounds or within an obstacle. Return a boolean representing whether or not this condition is true.

        Args:
            position: the robot position

        Returns:
            true if the position is valid and false otherwise
        """
        # check if within world bounds
        in_world = self.DIMENSIONS.within_bounds(position)

        # check if inside obstacles
        in_obstacle = any([obs.within_bounds(position) for obs in self.OBSTACLES])

        return in_world and not in_obstacle

    def get_robot_pose(self) -> Pose:
        """
        Return the true robot pose.
        """
        return self.robot_pose

    def get_proximity_to_landmarks(self) -> pd.DataFrame:
        """
        Return a list of the robot's true range and bearing to all landmarks.
        """
        prx_to_lms = pd.DataFrame()
        for lm in self.LANDMARKS:
            x_diff = lm.pos.x - self.robot_pose.pos.x
            y_diff = lm.pos.y - self.robot_pose.pos.y
            range = math.sqrt(x_diff**2 + y_diff**2)
            # Calculate absolute bearing (world frame) to match sensor model
            bearing = math.atan2(y_diff, x_diff) - self.robot_pose.theta
            # normalize angle to (-pi, pi]
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            prx_to_lms[f"Landmark{lm.id}"] = [BearingRange(lm.id, bearing, range)] # type: ignore
        return prx_to_lms

    def take_state_snapshot(self):
        """
        Return true state information about this timestep, including time, robot position, and the robot's bearing/range to landmarks, in a table format.
        """
        df1 = pd.DataFrame(
            {
                "Time": [self.time],
                "RobotPose": [
                    Pose(
                        Position(self.robot_pose.pos.x, self.robot_pose.pos.y),
                        self.robot_pose.theta,
                    )
                ],
            }
        )
        gt_to_lms = self.get_proximity_to_landmarks()
        df2 = pd.DataFrame()
        for lm in gt_to_lms.columns:
            df2[lm] = [gt_to_lms[lm].values[0]]

        return pd.merge(
            df1,
            df2,
            left_index=True,
            right_index=True,
        )

    def get_gt_field_value(self) -> float:
        """
        Returns the ground truth field measurement of the robot at the current ground truth pose.
        """
        return self.continuous_field.field.predict(
            np.asarray((self.robot_pose.pos.x, self.robot_pose.pos.y)).reshape(1,-1)
        ) # type: ignore

    def get_environment_info(self):
        """
        Return static information about the environment, including dimensions, timestep size, locations and dimensions of obstacles, and locations of landmarks.
        """
        info = {
            "Timestep": self.DT,
            "Obstacles": [obs.to_dict() for obs in self.OBSTACLES],
            "Landmarks": [lm.to_dict() for lm in self.LANDMARKS],
            "Dimensions": self.DIMENSIONS.to_dict(),
            "Pinger Range": self.lm_max_range,
            "Field": self.continuous_field.info()
        }

        file_path = "output/env_info.pkl"

        with open(file_path, "wb") as file:
            pickle.dump(info, file)

        print("Environment info saved to " + file_path)
        return info
