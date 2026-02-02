"""
Main file for running the simulator.
"""

import pickle
import pandas as pd
import argparse
import csv
import yaml
from pathlib import Path
from environment import Environment
from robot import Robot
from utils import Position, Pose, Landmark, Bounds
from viz import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a YAML config file.")

    parser.add_argument(
        "--config", 
        type=Path, 
        required=True, 
        help="Path to the yaml configuration file"
    )
    args = parser.parse_args()
    if args.config.exists():
        with open(args.config, 'r') as file:
            # yaml.safe_load converts the YAML structure into Python dicts/lists
            config_dict = yaml.safe_load(file)
    else:
        print(f"Error: The file {args.config} does not exist.")

    env_config = config_dict['environment']
    robot_config = config_dict["robot"]
    sensor_config = config_dict["sensors"]

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
                                    env_config["initial_robot_pose"][2]
                            )

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        lm_max_range,
        initial_robot_pose,
    )

    # set up the robot
    robot = Robot(env, sensor_config)

    # set up timekeeping
    total_seconds = env_config["runtime"]
    total_timesteps = total_seconds / env.DT
    terminal = False

    # set up logging
    ground_truth_history = pd.DataFrame()
    sensor_data_history = pd.DataFrame()

    # set up input filepath and output filepaths
    input_commands_filepath = "input/vel_cmd_example.csv"
    output_ground_truth_filepath = "output/ground_truth.pkl"
    output_sensor_data_filepath = "output/sensor_data.pkl"

    # open up the instructions, pop the first
    with open(input_commands_filepath, "r") as cmd:
    

        # pop motor commands
        csv_reader = csv.reader(cmd)
        next_cmd = next(csv_reader, None)  # skip header row
        next_cmd = next(csv_reader)
        crnt_lin_vel, crnt_ang_vel = float(next_cmd[1]), float(next_cmd[2])
        
        for step in range(int(total_timesteps) + 1):

            ground_truth_history = pd.concat([ground_truth_history, env.take_state_snapshot()], ignore_index= True)
            sensor_data_history = pd.concat([sensor_data_history, robot.take_sensor_measurements()], ignore_index= True)
            
            if not terminal and float(next_cmd[0]) <= step * env.DT:
                # pocket prev vel to run until next vel flip
                crnt_lin_vel, crnt_ang_vel = float(next_cmd[1]), float(next_cmd[2])
                try:
                    # flip to next vel
                    next_cmd = next(csv_reader)
                    
                except StopIteration:
                    terminal = True
            robot.robot_step_differential(crnt_lin_vel, crnt_ang_vel)

    
    pickle.dump(ground_truth_history, open(output_ground_truth_filepath, "wb"))
    pickle.dump(sensor_data_history, open(output_sensor_data_filepath, "wb"))
    env.get_environment_info()


    viz = Visualizer(
        Path("./output/"),
    )
    viz.draw_all()
