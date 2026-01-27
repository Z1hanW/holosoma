"""IsaacSim drawing adapter - moved from simulator class."""

from __future__ import annotations

import numpy as np

from holosoma.simulator.isaacsim.isaacsim import IsaacSim
from holosoma.utils.adapters.draw_utils import convert_to_list, convert_to_tuple


def clear_lines(simulator: IsaacSim) -> None:
    """Clear all debug lines from the viewer."""
    if hasattr(simulator, "draw") and simulator.draw:
        simulator.draw.clear_lines()
        simulator.draw.clear_points()


def draw_sphere(
    simulator: IsaacSim,
    pos,
    radius: float,
    color,
    env_id: int,
    pos_id: int | None = None,
) -> None:
    """Draw a sphere using points with unified type support."""
    if not hasattr(simulator, "draw") or not simulator.draw:
        return

    point_list = [convert_to_tuple(pos)]
    color_list = [convert_to_list(color) + [1.0]]
    sizes = [20]
    simulator.draw.draw_points(point_list, color_list, sizes)


def draw_line(
    simulator: IsaacSim,
    start_point,
    end_point,
    color,
    env_id: int,
) -> None:
    """Draw a line with unified type support."""
    if not hasattr(simulator, "draw") or not simulator.draw:
        return

    start_point_list = [convert_to_tuple(start_point)]
    end_point_list = [convert_to_tuple(end_point)]
    color_list = [convert_to_list(color) + [1.0]]
    sizes = [1]
    simulator.draw.draw_lines(start_point_list, end_point_list, color_list, sizes)


# Set the rest to no-op since we only need these 3
def draw_points(*args, **kwargs):
    """Draw points using IsaacSim debug draw."""
    if not args:
        return
    simulator = args[0]
    if not hasattr(simulator, "draw") or not simulator.draw:
        return

    points = args[1] if len(args) > 1 else []
    colors = args[2] if len(args) > 2 else []
    sizes = args[3] if len(args) > 3 else []

    points_arr = np.asarray(points)
    if points_arr.size == 0:
        return
    if points_arr.ndim == 1:
        points_arr = points_arr.reshape(1, -1)
    points_list = [convert_to_tuple(point) for point in points_arr]

    if not points_list:
        return

    colors_arr = np.asarray(colors)
    if colors_arr.size == 0:
        color_list = [[0.0, 1.0, 1.0, 1.0] for _ in points_list]
    else:
        if colors_arr.ndim == 1:
            base_color = colors_arr.astype(float).tolist()
            if len(base_color) == 3:
                base_color.append(1.0)
            color_list = [base_color for _ in points_list]
        else:
            color_list = []
            for color in colors_arr:
                color_vals = convert_to_list(color)
                if len(color_vals) == 3:
                    color_vals.append(1.0)
                color_list.append(color_vals)
            if len(color_list) != len(points_list):
                color_list = [color_list[0] for _ in points_list]

    sizes_arr = np.asarray(sizes)
    if sizes_arr.size == 0:
        size_list = [2.0 for _ in points_list]
    elif sizes_arr.ndim == 0:
        size_list = [float(sizes_arr.item()) for _ in points_list]
    else:
        size_list = [float(val) for val in sizes_arr.reshape(-1).tolist()]
        if len(size_list) != len(points_list):
            size_list = [float(size_list[0]) for _ in points_list]

    simulator.draw.draw_points(points_list, color_list, size_list)


def draw_height_points(*args, **kwargs):
    """No-op implementation for draw_height_points."""


def draw_foot_height_points(*args, **kwargs):
    """No-op implementation for draw_foot_height_points."""
