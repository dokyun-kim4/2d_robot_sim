"""
A simulation environment for a mobile robot operating in two dimensions.

The Environment class models the world that the robots navigate in. The world is continuous and two-dimensional. The world possesses an outer border, internal obstacles, and identifiable landmarks. The world also manages the passage of time and the motion of robotic agents within the world over time.

Critically, the environment tracks the robot's state. In this case, the robot's state is a vector that includes three state variables: x position, y position, and heading.
"""

import pickle
import datetime
import math
from src.utils import Position, Pose, Bounds, Landmark, BearingRange


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
        robot_starting_pose: Pose,
    ):
        """
        Initialize an instance of the Environment class.

        Args:
            dimensions: the horizontal and vertical size of the world
            dt: the length of each timestep, in seconds
            obstacles: a list of obstacles
            landmarks: a list of landmarks
            robot_starting_pose: the initial position and heading of the robot
        """
        self.DIMENSIONS = dimensions
        self.DT = dt
        self.time = 0
        self.OBSTACLES = obstacles
        self.LANDMARKS = landmarks
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
        dx,dy = self.validate_xy_motion(dx,dy)

        self.robot_pose.pos.x += dx
        self.robot_pose.pos.y += dy
        new_theta = self.robot_pose.theta = dtheta
        self.robot_pose.theta = (new_theta + math.pi) % (2*math.pi) - math.pi

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
        dx_valid = dx if self.is_valid_position(Position((self.robot_pose.pos.x + dx), self.robot_pose.pos.y)) else 0
        dy_valid = dy if self.is_valid_position(Position((self.robot_pose.pos.x), self.robot_pose.pos.y + dy)) else 0

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

    def get_proximity_to_landmarks(self) -> list[BearingRange]:
        """
        Return a list of the robot's true range and bearing to all landmarks.
        """
        prx_to_lms = []
        for lm in self.LANDMARKS:

            x_diff = lm.pos.x - self.robot_pose.pos.x 
            y_diff = lm.pos.y - self.robot_pose.pos.y
            range = math.sqrt(x_diff**2 + y_diff**2)
            bearing = math.atan2(y_diff, x_diff) - self.robot_pose.theta
            # normalize angle to (-pi, pi]
            bearing = (bearing + math.pi) % (2*math.pi) - math.pi
            prx_to_lms.append(BearingRange(lm.id, bearing, range))
        
        return prx_to_lms

    def take_state_snapshot(self):
        """
        Return true state information about this timestep, including time, robot position, and the robot's bearing/range to landmarks, in a table format.
        """
        pass

    def get_environment_info(self):
        """
        Return static information about the environment, including dimensions, timestep size, locations and dimensions of obstacles, and locations of landmarks.
        """
        info = {
                "timestep": self.DT,
                "obstacles": [obs.to_dict() for obs in self.OBSTACLES],
                "landmarks": [lm.to_dict() for lm in self.LANDMARKS],
                "world_size": self.DIMENSIONS.to_dict()
                }
        
        # Environment info is identified with current real-world time
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = "output/" + timestamp_str + ".pkl"

        with open(file_path, 'wb') as file:
            pickle.dump(info, file)
        
        print("Environment info saved to " + file_path)
        return info
