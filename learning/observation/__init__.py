def _missing_maps_observation(*_args, **_kwargs):
    raise ModuleNotFoundError(
        "learning/observation/maps_observation.py is missing. "
        "Run: git fetch origin && git checkout origin/main -- learning/observation/maps_observation.py",
    )


try:
    from .maps_observation import MAPSObservationBuilder, MAPSObservationConfig
except ModuleNotFoundError:
    # Keep RL imports usable even if MAPS imitation module is missing on a partial checkout.
    MAPSObservationBuilder = _missing_maps_observation
    MAPSObservationConfig = _missing_maps_observation
from .robot_state_observation import (
    RobotStateObservationBuilder,
    RobotStateObservationConfig,
)
from .cpp import (
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
    compute_directional_traversability,
)

__all__ = [
    "MAPSObservationBuilder",
    "MAPSObservationConfig",
    "RobotStateObservationBuilder",
    "RobotStateObservationConfig",
    "MultiScaleCPPObservationBuilder",
    "MultiScaleCPPObservationConfig",
    "compute_directional_traversability",
]
