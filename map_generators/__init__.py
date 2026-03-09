def build_indoor_map(*args, **kwargs):
    from .indoor import build_indoor_map as _impl

    return _impl(*args, **kwargs)


def build_random_map(*args, **kwargs):
    from .random_map import build_random_map as _impl

    return _impl(*args, **kwargs)


def build_shape_grid_map(*args, **kwargs):
    from .shape_grid import build_shape_grid_map as _impl

    return _impl(*args, **kwargs)


__all__ = ["build_indoor_map", "build_random_map", "build_shape_grid_map"]
