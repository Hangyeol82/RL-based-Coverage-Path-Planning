def build_indoor_map(*args, **kwargs):
    from .indoor import build_indoor_map as _impl

    return _impl(*args, **kwargs)


def build_indoor_curriculum_map(*args, **kwargs):
    from .indoor import build_indoor_curriculum_map as _impl

    return _impl(*args, **kwargs)


def build_random_map(*args, **kwargs):
    from .random_map import build_random_map as _impl

    return _impl(*args, **kwargs)


def build_shape_grid_map(*args, **kwargs):
    from .shape_grid import build_shape_grid_map as _impl

    return _impl(*args, **kwargs)


def build_validated_shape_grid_map(*args, **kwargs):
    from .shape_grid_presets import build_validated_shape_grid_map as _impl

    return _impl(*args, **kwargs)


def build_trail_grid_map(*args, **kwargs):
    from .trail_grid import build_trail_grid_map as _impl

    return _impl(*args, **kwargs)


def build_structured_map(*args, **kwargs):
    from .structured import build_structured_map as _impl

    return _impl(*args, **kwargs)


def build_pocket_trap_map(*args, **kwargs):
    from .structured import build_pocket_trap_map as _impl

    return _impl(*args, **kwargs)


def build_bridge_maze_map(*args, **kwargs):
    from .structured import build_bridge_maze_map as _impl

    return _impl(*args, **kwargs)


def build_room_corridor_map(*args, **kwargs):
    from .structured import build_room_corridor_map as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "build_indoor_map",
    "build_indoor_curriculum_map",
    "build_random_map",
    "build_shape_grid_map",
    "build_validated_shape_grid_map",
    "build_trail_grid_map",
    "build_structured_map",
    "build_pocket_trap_map",
    "build_bridge_maze_map",
    "build_room_corridor_map",
]
