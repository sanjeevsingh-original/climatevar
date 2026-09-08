"""WRF nested-domain calculations."""

from __future__ import annotations


def nested_grid_spacing(parent_dx: float, grid_ratios: list[int] | tuple[int, ...]) -> list[float]:
    """Calculate horizontal grid spacing for nested WRF domains.

    Parameters
    ----------
    parent_dx:
        Grid spacing of the outermost domain in metres.
    grid_ratios:
        Nesting ratios for each child domain. For example, ``[3, 3]`` gives
        ``[parent_dx, parent_dx/3, parent_dx/9]``.
    """
    if parent_dx <= 0:
        raise ValueError("parent_dx must be positive.")
    if any(r <= 0 or int(r) != r for r in grid_ratios):
        raise ValueError("grid ratios must be positive integers.")
    spacing = [float(parent_dx)]
    for ratio in grid_ratios:
        spacing.append(spacing[-1] / int(ratio))
    return spacing
