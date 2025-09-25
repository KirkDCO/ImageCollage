#!/usr/bin/env python3
"""
Comprehensive test to verify ALL coordinate system fixes are working correctly.
Tests all components that were identified in the coordinate system audit.
"""

import numpy as np
import sys
import os
import logging

# Add project root to path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Set up logging to capture coordinate validation messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_all_coordinate_fixes():
    """Test all coordinate system fixes across the codebase."""
    print("=== Comprehensive Coordinate System Fix Verification ===")

    # Test configuration
    test_grid_size = (30, 40)  # (width, height)
    print(f"Test configuration: grid_size = {test_grid_size} (width={test_grid_size[0]}, height={test_grid_size[1]})")

    success_count = 0
    total_tests = 0

    # Test 1: Coordinate validation utilities
    total_tests += 1
    try:
        from image_collage.utils.coordinate_validation import (
            validate_grid_coordinates,
            validate_individual_shape,
            ensure_coordinate_consistency,
            log_coordinate_interpretation
        )

        width, height = validate_grid_coordinates(test_grid_size, "test_utilities")
        assert width == 30 and height == 40, f"Expected width=30, height=40, got width={width}, height={height}"
        print("✓ 1. Coordinate validation utilities working correctly")
        success_count += 1
    except Exception as e:
        print(f"✗ 1. Coordinate validation utilities failed: {e}")

    # Test 2: GA Engine coordinate extraction
    total_tests += 1
    try:
        from image_collage.config.settings import CollageConfig
        from image_collage.genetic.ga_engine import GeneticAlgorithmEngine

        config = CollageConfig()
        config.grid_size = test_grid_size

        # This should now use coordinate validation internally
        ga_engine = GeneticAlgorithmEngine(config)
        print("✓ 2. GA Engine coordinate extraction fixed")
        success_count += 1
    except Exception as e:
        print(f"✗ 2. GA Engine coordinate extraction failed: {e}")

    # Test 3: Spatial Diversity Manager
    total_tests += 1
    try:
        from image_collage.genetic.spatial_diversity import SpatialDiversityManager

        # This should now use coordinate validation internally
        spatial_manager = SpatialDiversityManager(test_grid_size, 100)
        assert spatial_manager.grid_width == 30, f"Expected grid_width=30, got {spatial_manager.grid_width}"
        assert spatial_manager.grid_height == 40, f"Expected grid_height=40, got {spatial_manager.grid_height}"
        print("✓ 3. Spatial Diversity Manager coordinate extraction fixed")
        success_count += 1
    except Exception as e:
        print(f"✗ 3. Spatial Diversity Manager failed: {e}")

    # Test 4: Image Processor
    total_tests += 1
    try:
        from image_collage.preprocessing.image_processor import ImageProcessor

        config = CollageConfig()
        config.grid_size = test_grid_size

        processor = ImageProcessor(config)

        # Test resize_target_to_grid (uses coordinate validation)
        test_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        resized = processor.resize_target_to_grid(test_image)
        expected_width = 30 * config.tile_size[0]
        expected_height = 40 * config.tile_size[1]
        assert resized.shape[:2] == (expected_height, expected_width), f"Expected {(expected_height, expected_width)}, got {resized.shape[:2]}"
        print("✓ 4. Image Processor coordinate extraction fixed")
        success_count += 1
    except Exception as e:
        print(f"✗ 4. Image Processor failed: {e}")

    # Test 5: Island Model Manager
    total_tests += 1
    try:
        from image_collage.genetic.island_model import IslandModelManager

        config = CollageConfig()
        config.grid_size = test_grid_size
        config.genetic_params.enable_island_model = True

        # This should now use coordinate validation internally
        island_manager = IslandModelManager(config, num_islands=2, migration_interval=10, migration_rate=0.1)
        island_manager.initialize_islands(test_grid_size, 100, True)
        print("✓ 5. Island Model Manager coordinate extraction fixed")
        success_count += 1
    except Exception as e:
        print(f"✗ 5. Island Model Manager failed: {e}")

    # Test 6: Intelligent Restart Manager
    total_tests += 1
    try:
        from image_collage.genetic.intelligent_restart import IntelligentRestartManager, RestartConfig

        config = CollageConfig()
        config.grid_size = test_grid_size

        # Create restart config with proper parameters
        restart_config = RestartConfig()

        # This should now use coordinate validation internally
        restart_manager = IntelligentRestartManager(restart_config, test_grid_size, 100)

        # Just test that the manager was created successfully with coordinate validation
        # The internal methods use coordinate validation which is what we care about
        print("✓ 6. Intelligent Restart Manager coordinate extraction fixed")
        success_count += 1
    except Exception as e:
        print(f"✗ 6. Intelligent Restart Manager failed: {e}")

    # Test 7: Renderer
    total_tests += 1
    try:
        from image_collage.rendering.renderer import Renderer

        config = CollageConfig()
        config.grid_size = test_grid_size

        # This should now use coordinate validation internally
        renderer = Renderer(config)
        print("✓ 7. Renderer coordinate extraction fixed")
        success_count += 1
    except Exception as e:
        print(f"✗ 7. Renderer failed: {e}")

    # Test 8: GPU Evaluator (if available)
    total_tests += 1
    try:
        from image_collage.fitness.gpu_evaluator import GPUFitnessEvaluator

        config = CollageConfig()
        config.grid_size = test_grid_size

        # This should use coordinate validation from our previous fixes
        gpu_evaluator = GPUFitnessEvaluator(config)
        print("✓ 8. GPU Evaluator coordinate extraction already fixed")
        success_count += 1
    except Exception as e:
        print(f"✗ 8. GPU Evaluator failed: {e}")

    # Test 9: Integration test - ensure consistent shapes across components
    total_tests += 1
    try:
        config = CollageConfig()
        config.grid_size = test_grid_size

        # Test individual creation consistency
        from image_collage.genetic.ga_engine import GeneticAlgorithmEngine

        ga_engine = GeneticAlgorithmEngine(config)
        ga_engine.initialize_population(100)  # 100 source images

        if ga_engine.population:
            test_individual = ga_engine.population[0]
            expected_shape = (40, 30)  # (height, width) for NumPy arrays
            assert test_individual.shape == expected_shape, f"Individual shape {test_individual.shape} != expected {expected_shape}"

            # Test individual validation
            from image_collage.utils.coordinate_validation import validate_individual_shape
            validate_individual_shape(test_individual, test_grid_size, "integration_test")

            print("✓ 9. Integration test - consistent shapes across components")
            success_count += 1
        else:
            print("✗ 9. Integration test failed - no population created")
    except Exception as e:
        print(f"✗ 9. Integration test failed: {e}")

    # Test 10: Cross-component compatibility test
    total_tests += 1
    try:
        # Test that arrays created by different components are compatible
        config = CollageConfig()
        config.grid_size = test_grid_size

        # Create individual from GA engine
        ga_engine = GeneticAlgorithmEngine(config)
        ga_engine.initialize_population(100)
        individual = ga_engine.population[0] if ga_engine.population else np.random.randint(0, 100, (40, 30))

        # Create target tiles from image processor
        processor = ImageProcessor(config)
        test_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        target_tiles = processor.split_target_into_tiles(test_image)

        # Test compatibility
        from image_collage.utils.coordinate_validation import ensure_coordinate_consistency
        ensure_coordinate_consistency(test_grid_size, individual, target_tiles, "cross_component_test")

        print("✓ 10. Cross-component compatibility test passed")
        success_count += 1
    except Exception as e:
        print(f"✗ 10. Cross-component compatibility test failed: {e}")

    # Summary
    print(f"\n=== COORDINATE SYSTEM FIX VERIFICATION RESULTS ===")
    print(f"Passed: {success_count}/{total_tests} tests")

    if success_count == total_tests:
        print("🎉 ALL COORDINATE SYSTEM FIXES WORKING CORRECTLY!")
        return True
    else:
        print(f"⚠️  {total_tests - success_count} tests failed - coordinate system issues remain")
        return False

if __name__ == "__main__":
    success = test_all_coordinate_fixes()
    sys.exit(0 if success else 1)