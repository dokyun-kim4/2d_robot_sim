"""
Main file for running the simulator.
"""

import pickle
import pandas as pd
import argparse
import yaml
from pathlib import Path
from environment import Environment, Field
from robot import Robot
from action import Actions
from reward import Reward
from utils import Position, Pose, Landmark, Bounds, DriveType
from viz import Visualizer
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_commands",
        type=Path,
        default=Path("./input/diff_vel_cmd.csv"),
        help="Path to the CSV file containing motor commands",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./input/config.yaml"),
        help="Path to the yaml configuration file",
    )

    parser.add_argument(
        "--drive_type",
        type=int,
        choices=[0, 1],
        # 0 for differential drive, 1 for translational drive
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output/"),
        help="Directory to save output files and visualizations",
    )

    args = parser.parse_args()
    if args.config.exists():
        with open(args.config, "r") as file:
            # yaml.safe_load converts the YAML structure into Python dicts/lists
            config_dict = yaml.safe_load(file)
    else:
        print(f"Error: The file {args.config} does not exist.")

    env_config = config_dict["environment"]
    robot_config = config_dict["robot"]
    sensor_config = config_dict["sensors"]
    action_config = config_dict["action"]
    reward_config = config_dict["reward"]

    # set up the environment
    dimensions = Bounds(*env_config["dimension"])
    dt = env_config["dt"]
    obstacles = [Bounds(*obs) for obs in env_config["obstacles"]]
    
    landmarks = [
        Landmark(Position(*pos), i) for i, pos in enumerate(env_config["landmarks"])
    ]
    lm_max_range = env_config["lm_max_range"]

    initial_robot_pose = Pose(
        Position(
            env_config["initial_robot_pose"][0],
            env_config["initial_robot_pose"][1]
        ),
        env_config["initial_robot_pose"][2],
    )

    field = Field(
        dimensions,
        env_config["field"]["variance"],
        env_config["field"]["lengthscale"],
        env_config["field"]["random_seed"]
    )

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        field,
        lm_max_range,
        initial_robot_pose,
    )

    # set up the robot
    robot = Robot(
        env, sensor_config, robot_config, drive_type=DriveType(args.drive_type)
    )
    
    actions = Actions(
            action_config["action_step"],
            action_config["num_actions"],
            action_config["vel1_range"],
            action_config["vel2_range"]
        )
    
    reward = Reward(reward_config)
    reward_function = reward.reward

    # set up timekeeping
    total_seconds = env_config["runtime"]
    total_timesteps = total_seconds / env.DT
    steps_since_last_action = 0
    elapsed_action_time = 0.0
    terminal = False

    # set up logging
    ground_truth_history = pd.DataFrame()
    sensor_data_history = pd.DataFrame()

    # set up input filepath and output filepaths
    input_commands_filepath = args.input_commands
    output_ground_truth_filepath = args.output_dir / "ground_truth.pkl"
    output_sensor_data_filepath = args.output_dir / "sensor_data.pkl"
    output_sensor_info_filepath = args.output_dir / "sensor_info.pkl"
    output_action_info_filepath = args.output_dir / "action_info.pkl"
    output_reward_info_filepath = args.output_dir / "reward_info.pkl"


    crnt_lin_vel, crnt_ang_vel = 0.0, 0.0

    for step in range(int(total_timesteps) + 1):
        ground_truth_history = pd.concat(
            [ground_truth_history, env.take_state_snapshot()], ignore_index=True
        )

        crnt_sensor_data = robot.take_sensor_measurements()
        sensor_data_history = pd.concat(
            [sensor_data_history, crnt_sensor_data], ignore_index=True
        )

        # Update the robot's belief
        try:
            robot.update_belief(crnt_sensor_data["InsituInstrument"].values,
                                robot.env.robot_pose)
        except:
            pass  # no measurement available to use
            
        # Select a random action
        elapsed_time = env.DT * step
        elapsed_action_time = env.DT * steps_since_last_action
        if round(elapsed_action_time) >= actions.action_step:
            # select a random valid action
            waypoint_targets = actions.get_actions_as_waypoints(robot, "differential")
            action = np.random.choice(list(waypoint_targets.keys()))

            waypoint = waypoint_targets[action]
            waypoint_mean, waypoint_cov = robot.belief.predict(np.asarray([waypoint[0], waypoint[1]]).reshape(1, -1), return_cov=True) # type: ignore
            action_reward = reward_function(waypoint_mean[0], waypoint_cov[0])
            print(f"Expected Action Reward: {action_reward}")
            
            crnt_lin_vel, crnt_ang_vel = actions.command_actions[action]
            steps_since_last_action = 0
            elapsed_action_time = 0.0


        robot.robot_step_differential([crnt_lin_vel, crnt_ang_vel])
        steps_since_last_action += 1


    pickle.dump(ground_truth_history, open(output_ground_truth_filepath, "wb"))
    pickle.dump(sensor_data_history, open(output_sensor_data_filepath, "wb"))
    pickle.dump(robot.sensor_info(), open(output_sensor_info_filepath, "wb"))
    pickle.dump(actions.info(), open(output_action_info_filepath, "wb"))
    pickle.dump(reward.info(), open(output_reward_info_filepath, "wb"))
    env.get_environment_info()

    viz = Visualizer(Path("./output/"), drive_type=robot.drive_type)
    viz.draw_all()
