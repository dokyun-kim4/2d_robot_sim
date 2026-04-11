import numpy as np

from robot import Robot

class Actions:
    def __init__(self, action_step, num_actions, vel1_range, vel2_range):
        self.action_step = action_step  # how much time an action is composed of
        self.num_actions = num_actions  # discretization of the action space
        self.vel1_range = vel1_range  # bounds on the first velocity term
        self.vel2_range = vel2_range  # bounds on the second velocity term
        self.command_actions = {}  # list of actions as robot commands
        self._get_actions_as_commands()
    
    def _get_actions_as_commands(self):
        """Creates a list of actions in the robot command space."""
        vel1_options = np.linspace(self.vel1_range[0], self.vel1_range[1], self.num_actions)
        vel2_options = np.linspace(self.vel2_range[0], self.vel2_range[1], self.num_actions)
        Vx, Vy = np.meshgrid(vel1_options, vel2_options)
        for i, (vx, vy) in enumerate(zip(Vx.flatten(), Vy.flatten())):
            self.command_actions[i] = (vx, vy)
    
    def get_actions_as_waypoints(self, robot: Robot, drive_type="differential"):
        """Lists actions as states in the world based on robot position."""
        # get robot pose and world boundaries
        robot_pose = robot.env.robot_pose ## can change this to be estimated robot position
        bounds = robot.env.DIMENSIONS
        waypoints = {}
        # compute approximate waypoints we would want the robot to go to
        for action in self.command_actions.keys():
            vel1, vel2 = self.command_actions[action]
            if drive_type == "differential":
                if np.abs(vel2) < 0.01:
                    dx = vel1 * self.action_step * np.cos(robot_pose.theta)
                    dy = vel1 * self.action_step * np.sin(robot_pose.theta)
                    dtheta = 0.0
                else:
                    r = vel1 / vel2
                    dtheta = vel2 * self.action_step
                    dx = r * (np.sin(robot_pose.theta + dtheta) - np.sin(robot_pose.theta))
                    dy = -r * (np.cos(robot_pose.theta + dtheta) - np.cos(robot_pose.theta))
                waypoints[action] = (robot_pose.pos.x + dx, robot_pose.pos.y + dy, robot_pose.theta + dtheta)
            elif drive_type == "swerve":
                dx = vel1 * self.action_step
                dy = vel2 * self.action_step
                waypoints[action] = (robot_pose.pos.x + dx, robot_pose.pos.y + dy, 0.0)
        
        # Remove actions that lead to out-of-bounds results
        invalid_actions = []
        for action, waypoint in waypoints.items():
            if waypoint[0] > bounds.x_max or waypoint[0] < bounds.x_min or waypoint[1] > bounds.y_max or waypoint[1] < bounds.y_min:
                invalid_actions.append(action)
        if len(invalid_actions) > 0:
            for action in invalid_actions:
                del waypoints[action]
        
        return waypoints
    
    def convert_action_to_velocity(self, action):
        """Converts an action into a velocity command the robot can execute."""
        return self.command_actions[action]
    
    def info(self):
        """Saves action information."""
        return {
            "Action Step": self.action_step,
            "Num Actions": self.num_actions,
            "Vel1 Range": self.vel1_range,
            "Vel2 Range": self.vel2_range,
            "Command Actions": self.command_actions,
        }
