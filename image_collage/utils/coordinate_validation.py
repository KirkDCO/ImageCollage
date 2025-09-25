"""
Coordinate system validation and standardization utilities.
Ensures consistent (width, height) interpretation throughout the codebase.
"""

import logging
from typing import Tuple, Any
import numpy as np


def validate_grid_coordinates(grid_size: Tuple[int, int], context: str = "") -> Tuple[int, int]:
    """
    Ensure consistent (width, height) interpretation and validate grid dimensions.

    Args:
        grid_size: Tuple representing (width, height)
        context: Description for error messages

    Returns:
        Tuple of (width, height) with validation

    Raises:
        ValueError: If grid dimensions are invalid
    """
    if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
        raise ValueError(f"Grid size must be (width, height) tuple in {context}")

    width, height = grid_size
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError(f"Grid dimensions must be integers in {context}")

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid grid dimensions {grid_size} in {context}")

    logging.debug(f"{context}: Using grid (width={width}, height={height})")
    return width, height


def validate_individual_shape(individual: np.ndarray, expected_grid_size: Tuple[int, int],
                             context: str = "") -> None:
    """
    Validate that individual array has correct shape for grid size.

    Args:
        individual: NumPy array representing an individual
        expected_grid_size: Expected (width, height) of grid
        context: Description for error messages

    Raises:
        ValueError: If individual shape doesn't match expected grid
    """
    width, height = validate_grid_coordinates(expected_grid_size, context)
    expected_shape = (height, width)  # Arrays use (rows, cols) = (height, width)

    if individual.shape != expected_shape:
        raise ValueError(
            f"Individual shape mismatch in {context}: "
            f"expected {expected_shape} (h={height}, w={width}) "
            f"but got {individual.shape}"
        )


def validate_array_compatibility(array1: np.ndarray, array2: np.ndarray,
                                context: str = "") -> None:
    """
    Validate that two arrays have compatible shapes for operations.

    Args:
        array1: First array
        array2: Second array
        context: Description for error messages

    Raises:
        ValueError: If arrays have incompatible shapes
    """
    if array1.shape != array2.shape:
        raise ValueError(
            f"Array shape mismatch in {context}: "
            f"array1 shape {array1.shape} vs array2 shape {array2.shape}"
        )


def safe_array_access(array: np.ndarray, i: int, j: int, context: str = "") -> Any:
    """
    Safe 2D array access with bounds checking.

    Args:
        array: 2D numpy array
        i: Row index
        j: Column index
        context: Description for error messages

    Returns:
        Array element at [i, j]

    Raises:
        IndexError: If indices are out of bounds
    """
    if len(array.shape) < 2:
        raise IndexError(f"Array must be at least 2D for safe access in {context}")

    height, width = array.shape[:2]

    if not (0 <= i < height and 0 <= j < width):
        raise IndexError(
            f"Array access out of bounds in {context}: "
            f"index ({i}, {j}) for array shape {array.shape}"
        )

    return array[i, j]


def convert_config_to_array_shape(grid_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert config grid_size (width, height) to array shape (height, width).

    Args:
        grid_size: Configuration grid size as (width, height)

    Returns:
        Array shape as (height, width)
    """
    width, height = validate_grid_coordinates(grid_size, "config_to_array_conversion")
    return height, width


def convert_array_shape_to_config(array_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert array shape (height, width) to config grid_size (width, height).

    Args:
        array_shape: Array shape as (height, width)

    Returns:
        Config grid size as (width, height)
    """
    if len(array_shape) < 2:
        raise ValueError("Array shape must have at least 2 dimensions")

    height, width = array_shape[:2]
    return width, height


def log_coordinate_interpretation(grid_size: Tuple[int, int], component: str) -> None:
    """
    Log coordinate system interpretation for debugging.

    Args:
        grid_size: Grid size tuple
        component: Component name for logging
    """
    width, height = validate_grid_coordinates(grid_size, component)
    logging.info(f"{component} coordinate system:")
    logging.info(f"  - Config grid_size: {grid_size} (width={width}, height={height})")
    logging.info(f"  - Array shape: ({height}, {width}) (rows, cols)")
    logging.info(f"  - Total tiles: {width * height}")


def ensure_coordinate_consistency(config_grid_size: Tuple[int, int],
                                 individual: np.ndarray,
                                 target_array: np.ndarray,
                                 component: str = "") -> None:
    """
    Comprehensive coordinate system consistency check.

    Args:
        config_grid_size: Grid size from configuration (width, height)
        individual: Individual array to validate
        target_array: Target array to validate against
        component: Component name for error messages

    Raises:
        ValueError: If coordinate systems are inconsistent
    """
    log_coordinate_interpretation(config_grid_size, component)

    # Validate config grid size
    width, height = validate_grid_coordinates(config_grid_size, f"{component}_config")

    # Validate individual shape matches config
    validate_individual_shape(individual, config_grid_size, f"{component}_individual")

    # Validate target array has compatible dimensions
    expected_target_shape = (height, width)
    if target_array.shape[:2] != expected_target_shape:
        raise ValueError(
            f"Target array shape mismatch in {component}: "
            f"expected {expected_target_shape} but got {target_array.shape[:2]}"
        )

    logging.info(f"{component}: Coordinate consistency validated ✓")