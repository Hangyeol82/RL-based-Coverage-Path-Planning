"""Backward-compat import path for robot-state observation builders."""

from learning.observation.robot_state_observation import (
    RobotStateObservationBuilder,
    RobotStateObservationConfig,
)

__all__ = ["RobotStateObservationBuilder", "RobotStateObservationConfig"]
