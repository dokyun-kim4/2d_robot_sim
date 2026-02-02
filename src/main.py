"""
Main file for running the simulator.
"""

import pickle
import pandas as pd
import csv
from environment import Environment
from robot import Robot
from utils import Position, Pose, Landmark, Bounds

if __name__ == "__main__":
    # set up the environment
    dimensions = Bounds(0, 10, 0, 10)
    dt = 0.1
    obstacles = [Bounds(5, 7, 5, 7), Bounds(0, 4, 6, 8)]
    landmarks = [
                    Landmark(Position(2,2), 1), 
                    Landmark(Position(5,5), 2),
                    Landmark(Position(8,8), 3)
                ]
    lm_max_range = 3.0
    initial_robot_pose = Pose(Position(0, 0), 0.6)

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        lm_max_range,
        initial_robot_pose,
    )

    # set up the robot
    robot = Robot(env)

    # set up timekeeping
    total_seconds = 30
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
        next(csv_reader, None)  # skip header row
        next_cmd = next(csv_reader)
        start_time = float(next_cmd[0])
        next_lin_vel, next_ang_vel = float(next_cmd[1]), float(next_cmd[2])

        for step in range(int(total_timesteps) + 1):
            ground_truth_history = pd.concat([ground_truth_history, env.take_state_snapshot()])
            sensor_data_history = pd.concat([sensor_data_history, robot.take_sensor_measurements()])
            
            if not terminal and start_time >= step * env.DT:
                try:
                    next_cmd = next(csv_reader)
                    start_time = float(next_cmd[0])
                    next_lin_vel, next_ang_vel = float(next_cmd[1]), float(next_cmd[2])
                except StopIteration:
                    terminal = True
            robot.robot_step_differential(next_lin_vel, next_ang_vel)

    pickle.dump(ground_truth_history, open(output_ground_truth_filepath, "wb"))
    pickle.dump(sensor_data_history, open(output_sensor_data_filepath, "wb"))
    env.get_environment_info()
