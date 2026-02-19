import numpy as np

from learning.observation.maps_observation import MAPSObservationBuilder
from learning.observation.robot_state_observation import RobotStateObservationBuilder
from run_cstar_custom_map import CUSTOM_MAP_TEXT, parse_custom_map


def main():
    grid = parse_custom_map(CUSTOM_MAP_TEXT)
    robot_pos = (0, 0)
    prev_pos = (0, 0)
    explored = np.zeros_like(grid, dtype=bool)
    explored[robot_pos[0], robot_pos[1]] = True
    recent_new_coverage = [1, 1, 0, 0, 0, 0, 1, 0]

    builder = MAPSObservationBuilder()
    levels = builder.build_levels(grid, robot_pos=robot_pos, explored=explored)
    cnn_input = builder.build_cnn_input(grid, robot_pos=robot_pos, explored=explored)
    robot_state_builder = RobotStateObservationBuilder()
    robot_state = robot_state_builder.build(
        occupancy=grid,
        explored=explored,
        robot_pos=robot_pos,
        prev_pos=prev_pos,
        recent_new_coverage=recent_new_coverage,
    )

    print("MAP size:", grid.shape)
    for lv in range(6):
        x = levels[lv]
        print(
            f"level {lv} shape={x.shape} "
            f"pot[min,max]=({x[0].min():.3f},{x[0].max():.3f}) "
            f"obs_ratio[min,max]=({x[1].min():.3f},{x[1].max():.3f}) "
            f"known_ratio[min,max]=({x[2].min():.3f},{x[2].max():.3f})"
        )
    print("cnn_input shape:", cnn_input.shape)
    print("robot_state shape:", robot_state.shape)
    print("robot_state features:")
    for name, value in zip(robot_state_builder.feature_names(), robot_state):
        print(f"  {name}: {float(value):.4f}")


if __name__ == "__main__":
    main()
