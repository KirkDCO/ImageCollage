"""
Test suite for Image Collage Generator.

This package contains comprehensive tests for all components of the image collage
generation system, including coordinate system validation, genetic algorithm
operations, and cross-component integration tests.

Usage:
    # Run comprehensive coordinate system tests
    python3 image_collage/tests/test_comprehensive_coordinate_fixes.py

    # Run basic coordinate validation tests
    python3 image_collage/tests/test_coordinate_fix.py

    # Or using pytest (if installed)
    pytest image_collage/tests/
"""

# Test modules available in this package
__all__ = [
    'test_coordinate_fix',
    'test_comprehensive_coordinate_fixes'
]