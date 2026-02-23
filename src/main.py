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
from kalman_filter import KalmanFilter
from extended_kalman_filter import ExtendedKalmanFilter
from utils import Position, Pose, Landmark, Bounds, DriveType
from viz import Visualizer
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./input/config.yaml"),
        help="Path to the yaml configuration file",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output/"),
        help="Directory to save output files and visualizations",
    )

    parser.add_argument(
        "--input_commands",
        type=Path,
        default=Path("./input/diff_vel_cmd.csv"),
        help="Path to the CSV file containing motor commands",
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
            env_config["initial_robot_pose"][0], env_config["initial_robot_pose"][1]
        ),
        env_config["initial_robot_pose"][2],
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
    robot = Robot(env, sensor_config, robot_config)

    if robot.drive_type == DriveType.TRANSLATIONAL:
        kf = KalmanFilter(
            dt,
            np.array(
                [
                    [
                        initial_robot_pose.pos.x,
                        initial_robot_pose.pos.y,
                        initial_robot_pose.theta,
                    ]
                ]
            ).T,
        )
    else:
        # set up the Extended Kalman Filter
        kf = ExtendedKalmanFilter(
            dt,
            np.array(
                [
                    [
                        initial_robot_pose.pos.x,
                        initial_robot_pose.pos.y,
                        initial_robot_pose.theta,
                    ]
                ]
            ),
        )

    # set up timekeeping
    total_seconds = env_config["runtime"]
    total_timesteps = total_seconds / env.DT
    terminal = False

    # set up logging
    ground_truth_history = pd.DataFrame()
    sensor_data_history = pd.DataFrame()
    kalman_filter_history = []

    # set up input filepath and output filepaths
    input_commands_filepath = args.input_commands
    output_ground_truth_filepath = args.output_dir / "ground_truth.pkl"
    output_sensor_data_filepath = args.output_dir / "sensor_data.pkl"
    output_kalman_filter_filepath = args.output_dir / "kalman_filter.pkl"

    # open up the instructions, pop the first
    with open(input_commands_filepath, "r") as cmd:
        # pop motor commands
        csv_reader = csv.reader(cmd)
        next_cmd = next(csv_reader, None)  # skip header row
        next_cmd = next(csv_reader)
        crnt_lin_vel, crnt_ang_vel = float(next_cmd[1]), float(next_cmd[2])

        for step in range(int(total_timesteps) + 1):
            ground_truth_history = pd.concat(
                [ground_truth_history, env.take_state_snapshot()], ignore_index=True
            )

            crnt_sensor_measurement = robot.take_sensor_measurements()
            sensor_data_history = pd.concat(
                [sensor_data_history, crnt_sensor_measurement], ignore_index=True
            )

            if robot.drive_type == DriveType.DIFFERENTIAL:
                u = np.array(
                    [
                        crnt_sensor_measurement["wheel_encoder_lin_vel_actual"],
                        crnt_sensor_measurement["wheel_encoder_ang_vel_actual"],
                    ]
                )

            else:
                u = np.array(
                    [
                        crnt_sensor_measurement["wheel_encoder_vx_actual"],
                        crnt_sensor_measurement["wheel_encoder_vy_actual"],
                        crnt_sensor_measurement["wheel_encoder_ang_vel_actual"],
                    ]
                )
            x, P = kf.predict(u)

            if "GPS" in crnt_sensor_measurement.columns:
                gps_data = crnt_sensor_measurement["GPS"].iloc[0]
                z = np.array([[gps_data.x, gps_data.y]]).T
                gps = next((inst for inst in robot.sensors if inst.name == "GPS"), None)

                if robot.drive_type == DriveType.DIFFERENTIAL:
                    kf.update(H=gps.H, R=gps.R, z=z, y=None)
                else:
                    kf.update(H=gps.H, R=gps.R, z=z)

            # Only use landmark measurement for EKF since it is nonlinear
            if robot.drive_type == DriveType.DIFFERENTIAL:
                # First filter out all landmark pinger measurements
                landmarks = crnt_sensor_measurement.filter(like="landmark_pinger").iloc[
                    0
                ]

                # Iterate through all landmark pinger measurements and update EKF
                for lm_name, val in landmarks.items():
                    lm_id = int(lm_name.strip("_")[-1])

                    z = np.array([[val.range, val.bearing]]).T
                    print(z)
                    # First check if measurement is valid (ex: not inf when out of range)
                    if np.isinf(z).any():
                        continue

                    # Get corresponding LandmarkPinger instance to compute H and y for EKF update
                    lm_pinger = next(
                        (
                            inst
                            for inst in robot.sensors
                            if inst.name == "landmark_pinger"
                        ),
                        None,
                    )
                    x, P = kf.update(
                        H=lm_pinger.H_eval(x, lm_id),
                        R=lm_pinger.R(z),
                        z=z,
                        y=lm_pinger.y(z, x, lm_id),
                    )

            kalman_filter_history.append((kf.x_state, kf.P))

            if not terminal and float(next_cmd[0]) <= step * env.DT:
                # pocket prev vel to run until next vel flip
                crnt_cmd = [float(x) for x in next_cmd[1:]]
                try:
                    # flip to next vel
                    next_cmd = next(csv_reader)

                except StopIteration:
                    terminal = True

            if robot.drive_type == DriveType.DIFFERENTIAL:
                robot.robot_step_differential(crnt_cmd)
            else:
                robot.robot_step_translational(crnt_cmd)

    pickle.dump(ground_truth_history, open(output_ground_truth_filepath, "wb"))
    pickle.dump(sensor_data_history, open(output_sensor_data_filepath, "wb"))
    pickle.dump(kalman_filter_history, open(output_kalman_filter_filepath, "wb"))
    env.get_environment_info()

    viz = Visualizer(Path("./output/"), drive_type=robot.drive_type)
    viz.draw_all()
