#!/usr/bin/env python3
"""
Test script to verify coordinate system fixes are working.
This simulates the conditions that caused the original error.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, '/opt/Projects/ImageCollage')

from image_collage.utils.coordinate_validation import (
    validate_grid_coordinates,
    validate_individual_shape,
    validate_array_compatibility,
    ensure_coordinate_consistency
)

def test_coordinate_consistency():
    """Test the coordinate system fixes."""
    print("=== Testing Coordinate System Fixes ===")

    # Test configuration from the failing run
    config_grid_size = (30, 40)  # (width, height) from config

    print(f"Config grid_size: {config_grid_size} (width={config_grid_size[0]}, height={config_grid_size[1]})")

    # Test individual creation (should be height x width)
    width, height = validate_grid_coordinates(config_grid_size, "test_individual_creation")
    individual = np.random.randint(0, 100, size=(height, width))  # (40, 30)

    print(f"Individual shape: {individual.shape}")

    # Validate individual matches config
    validate_individual_shape(individual, config_grid_size, "test_consistency")
    print("✓ Individual shape validation passed")

    # Test target tiles creation (should match individual)
    target_tiles = np.zeros((height, width, 32, 32, 3), dtype=np.uint8)  # (40, 30, 32, 32, 3)

    print(f"Target tiles shape: {target_tiles.shape}")

    # Test comprehensive consistency
    ensure_coordinate_consistency(
        config_grid_size, individual, target_tiles, "test_comprehensive"
    )
    print("✓ Comprehensive coordinate consistency validated")

    # Test two individuals for crossover compatibility
    individual2 = np.random.randint(0, 100, size=(height, width))  # Same shape
    validate_array_compatibility(individual, individual2, "test_crossover_compatibility")
    print("✓ Crossover compatibility validated")

    # Test crossover operation that was failing
    try:
        # Simulate the crossover operation that was failing
        child1 = individual.copy()
        child2 = individual2.copy()

        # Column-wise crossover (the problematic operation)
        crossover_point1 = 5
        crossover_point2 = 15

        print(f"Attempting crossover with slice [:, {crossover_point1}:{crossover_point2}]")
        print(f"child1 shape: {child1.shape}, child2 slice shape: {child2[:, crossover_point1:crossover_point2].shape}")

        child1[:, crossover_point1:crossover_point2] = child2[:, crossover_point1:crossover_point2]
        print("✓ Crossover operation successful")

    except ValueError as e:
        print(f"✗ Crossover operation failed: {e}")
        return False

    print("\n🎉 All coordinate system tests passed!")
    return True

if __name__ == "__main__":
    success = test_coordinate_consistency()
    sys.exit(0 if success else 1)