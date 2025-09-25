"""
Utility modules for image collage generation.
"""

from .diversity_metrics import (
    calculate_hamming_distance_average,
    calculate_hamming_distance_std,
    calculate_position_wise_entropy,
    calculate_unique_individuals_ratio,
    calculate_population_entropy,
    estimate_genetic_clusters,
    normalize_diversity_score,
    calculate_population_diversity,
    calculate_cluster_diversity,
    calculate_hamming_diversity  # Legacy compatibility
)

from .coordinate_validation import (
    validate_grid_coordinates,
    validate_individual_shape,
    validate_array_compatibility,
    safe_array_access,
    convert_config_to_array_shape,
    convert_array_shape_to_config,
    log_coordinate_interpretation,
    ensure_coordinate_consistency
)

__all__ = [
    'calculate_hamming_distance_average',
    'calculate_hamming_distance_std',
    'calculate_position_wise_entropy',
    'calculate_unique_individuals_ratio',
    'calculate_population_entropy',
    'estimate_genetic_clusters',
    'normalize_diversity_score',
    'calculate_population_diversity',
    'calculate_cluster_diversity',
    'calculate_hamming_diversity',
    'validate_grid_coordinates',
    'validate_individual_shape',
    'validate_array_compatibility',
    'safe_array_access',
    'convert_config_to_array_shape',
    'convert_array_shape_to_config',
    'log_coordinate_interpretation',
    'ensure_coordinate_consistency',
]