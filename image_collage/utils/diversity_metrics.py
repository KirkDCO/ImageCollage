"""
Centralized diversity metrics calculations for genetic algorithms.

This module provides optimized, consistent implementations of diversity metrics
to eliminate code duplication across the codebase.
"""

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Any
from collections import defaultdict


def calculate_hamming_distance_average(population: List[np.ndarray]) -> float:
    """
    Calculate average pairwise Hamming distance with performance optimization.

    For large populations (>50), uses sampling to avoid O(n²) performance.
    For small populations, calculates all pairs.

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Average Hamming distance (0.0 to 1.0, normalized by array size)
    """
    if len(population) < 2:
        return 0.0

    # For large populations, use sampling to avoid O(n^2) performance
    if len(population) > 50:
        # Sample at most 1000 pairs for performance
        max_samples = min(1000, len(population) * (len(population) - 1) // 2)

        # Efficient random pair sampling without creating all pairs in memory
        sample_pairs = []
        for _ in range(max_samples):
            i = random.randint(0, len(population) - 2)
            j = random.randint(i + 1, len(population) - 1)
            sample_pairs.append((i, j))

        total_distance = 0
        for i, j in sample_pairs:
            # Use flattened arrays for faster comparison
            ind1, ind2 = population[i].flatten(), population[j].flatten()
            distance = np.sum(ind1 != ind2)
            total_distance += distance

        if sample_pairs and len(population[0].flatten()) > 0:
            # Normalize by maximum possible distance
            max_distance = len(population[0].flatten())
            avg_distance = total_distance / len(sample_pairs)
            normalized = avg_distance / max_distance
            return max(0.0, min(1.0, normalized))  # Ensure 0.0 <= result <= 1.0
        return 0.0
    else:
        # For small populations, calculate all pairs
        total_distance = 0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                ind1, ind2 = population[i].flatten(), population[j].flatten()
                distance = np.sum(ind1 != ind2)
                total_distance += distance
                comparisons += 1

        if comparisons > 0 and len(population[0].flatten()) > 0:
            max_distance = len(population[0].flatten())
            avg_distance = total_distance / comparisons
            normalized = avg_distance / max_distance
            return max(0.0, min(1.0, normalized))  # Ensure 0.0 <= result <= 1.0

        return 0.0


def calculate_hamming_distance_std(population: List[np.ndarray]) -> float:
    """
    Calculate standard deviation of pairwise Hamming distances with sampling.

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Standard deviation of Hamming distances (unnormalized)
    """
    if len(population) < 2:
        return 0.0

    # For large populations, use sampling to avoid O(n^2) performance
    if len(population) > 50:
        # Sample at most 1000 pairs for performance
        max_samples = min(1000, len(population) * (len(population) - 1) // 2)

        # Efficient random pair sampling without creating all pairs in memory
        sample_pairs = []
        for _ in range(max_samples):
            i = random.randint(0, len(population) - 2)
            j = random.randint(i + 1, len(population) - 1)
            sample_pairs.append((i, j))

        distances = []
        for i, j in sample_pairs:
            distance = np.sum(population[i].flatten() != population[j].flatten())
            distances.append(distance)

        return np.std(distances) if distances else 0.0
    else:
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.sum(population[i].flatten() != population[j].flatten())
                distances.append(distance)

        return np.std(distances) if distances else 0.0


def calculate_position_wise_entropy(population: List[np.ndarray]) -> float:
    """
    Calculate position-wise entropy across population with performance optimization.

    For large arrays (>1000 positions), uses position sampling to reduce computation.

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Average entropy per position
    """
    if not population:
        return 0.0

    # Flatten all individuals to 1D arrays
    flattened_pop = [individual.flatten() for individual in population]

    if not flattened_pop or len(flattened_pop[0]) == 0:
        return 0.0

    individual_length = len(flattened_pop[0])
    total_entropy = 0

    # For large arrays, sample positions to reduce computation
    if individual_length > 1000:
        # Sample 1000 positions maximum
        sample_positions = random.sample(range(individual_length), min(1000, individual_length))
    else:
        sample_positions = range(individual_length)

    for pos in sample_positions:
        values = [individual[pos] for individual in flattened_pop]
        value_counts = defaultdict(int)
        for value in values:
            value_counts[value] += 1

        # Calculate Shannon entropy
        entropy = 0
        total_count = len(values)
        for count in value_counts.values():
            if count > 0:
                probability = count / total_count
                entropy -= probability * math.log2(probability)

        total_entropy += entropy

    return total_entropy / len(sample_positions)


def calculate_unique_individuals_ratio(population: List[np.ndarray]) -> float:
    """
    Calculate ratio of unique individuals to total population.

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Ratio of unique individuals (0.0 to 1.0)
    """
    if not population:
        return 0.0

    unique_individuals = set()
    for individual in population:
        unique_individuals.add(tuple(individual.flatten()))

    return len(unique_individuals) / len(population)


def calculate_population_entropy(population: List[np.ndarray]) -> float:
    """
    Calculate overall population entropy based on individual uniqueness.

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Population entropy (Shannon entropy of individual distribution)
    """
    if not population:
        return 0.0

    # Treat each individual as a single "symbol"
    individual_strings = [tuple(ind.flatten()) for ind in population]
    string_counts = defaultdict(int)
    for string in individual_strings:
        string_counts[string] += 1

    # Calculate Shannon entropy
    total_count = len(individual_strings)
    entropy = 0
    for count in string_counts.values():
        if count > 0:
            probability = count / total_count
            entropy -= probability * math.log2(probability)

    return entropy


def estimate_genetic_clusters(population: List[np.ndarray], max_clusters: int = 10) -> int:
    """
    Estimate number of genetic clusters using distance-based clustering with sampling.

    Args:
        population: List of numpy arrays representing individuals
        max_clusters: Maximum number of clusters to return

    Returns:
        Estimated number of genetic clusters
    """
    if len(population) < 2:
        return 1

    # For large populations, sample distances to estimate clustering
    if len(population) > 50:
        # Sample at most 500 pairs for performance
        max_samples = min(500, len(population) * (len(population) - 1) // 2)

        # Efficient random pair sampling without creating all pairs in memory
        sample_pairs = []
        for _ in range(max_samples):
            i = random.randint(0, len(population) - 2)
            j = random.randint(i + 1, len(population) - 1)
            sample_pairs.append((i, j))

        distances = []
        for i, j in sample_pairs:
            dist = np.sum(population[i].flatten() != population[j].flatten())
            distances.append(dist)
    else:
        # Calculate all pairwise distances for small populations
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.sum(population[i].flatten() != population[j].flatten())
                distances.append(dist)

    if not distances:
        return 1

    # Use threshold-based clustering estimation
    avg_distance = np.mean(distances)
    threshold = avg_distance * 0.5

    # Estimate clusters by sampling representative individuals
    sample_size = min(20, len(population))  # Sample 20 individuals
    sample_indices = random.sample(range(len(population)), sample_size)

    clusters = []
    assigned = [False] * sample_size

    for i in range(sample_size):
        if not assigned[i]:
            cluster = [i]
            assigned[i] = True

            # Find all sampled individuals within threshold distance
            for j in range(sample_size):
                if not assigned[j]:
                    idx_i = sample_indices[i]
                    idx_j = sample_indices[j]
                    dist = np.sum(population[idx_i].flatten() != population[idx_j].flatten())
                    if dist <= threshold:
                        cluster.append(j)
                        assigned[j] = True

            clusters.append(cluster)

    # Scale cluster count estimate to full population
    cluster_ratio = len(clusters) / sample_size
    estimated_clusters = max(1, int(cluster_ratio * len(population)))
    return min(estimated_clusters, max_clusters)


def normalize_diversity_score(metrics: Dict[str, float], population_size: int,
                            grid_size: Tuple[int, int]) -> float:
    """
    Combine multiple metrics into normalized diversity score (0-1).

    Args:
        metrics: Dictionary containing diversity metrics
        population_size: Size of the population
        grid_size: Dimensions of the grid (height, width)

    Returns:
        Normalized diversity score (0.0 to 1.0)
    """
    if population_size <= 1:
        return 0.0

    # Weight different metrics based on importance
    weights = {
        'hamming_distance_avg': 0.25,
        'position_wise_entropy': 0.20,
        'unique_individuals_ratio': 0.20,
        'fitness_coefficient_variation': 0.15,
        'cluster_count': 0.10,
        'population_entropy': 0.10
    }

    # Normalize each metric to 0-1 range
    normalized = {}

    # Hamming distance (normalize by max possible distance)
    max_hamming = grid_size[0] * grid_size[1]
    normalized['hamming_distance_avg'] = min(1.0,
        metrics.get('hamming_distance_avg', 0.0) / max_hamming) if max_hamming > 0 else 0.0

    # Position entropy (normalize by max possible entropy)
    max_position_entropy = math.log2(population_size) if population_size > 1 else 1
    normalized['position_wise_entropy'] = (metrics.get('position_wise_entropy', 0.0) /
                                         max_position_entropy)

    # Unique individuals ratio (already 0-1)
    normalized['unique_individuals_ratio'] = metrics.get('unique_individuals_ratio', 0.0)

    # Fitness coefficient of variation (cap at 1.0)
    normalized['fitness_coefficient_variation'] = min(1.0,
        metrics.get('fitness_coefficient_variation', 0.0))

    # Cluster count (normalize by reasonable maximum - use sqrt of population for better scaling)
    max_clusters = min(population_size, max(10, int(math.sqrt(population_size))))
    normalized['cluster_count'] = min(1.0,
        metrics.get('cluster_count', 1.0) / max_clusters)

    # Population entropy (normalize by max possible)
    max_pop_entropy = math.log2(population_size) if population_size > 1 else 1
    normalized['population_entropy'] = (metrics.get('population_entropy', 0.0) /
                                      max_pop_entropy)

    # Calculate weighted average
    total_score = sum(weights[key] * normalized[key] for key in weights.keys())
    return total_score


# Convenience function for backward compatibility
def calculate_population_diversity(population: List[np.ndarray]) -> float:
    """
    Calculate average pairwise diversity in population (same as normalized Hamming distance).

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Average pairwise diversity (0.0 to 1.0)
    """
    return calculate_hamming_distance_average(population)


def calculate_cluster_diversity(population: List[np.ndarray]) -> float:
    """
    Estimate cluster-based diversity using genetic clustering.

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Cluster diversity score (normalized by population size)
    """
    if len(population) <= 1:
        return 0.0

    cluster_count = estimate_genetic_clusters(population)
    max_possible_clusters = min(len(population), 10)
    return cluster_count / max_possible_clusters


def calculate_hamming_diversity(population: List[np.ndarray]) -> float:
    """
    Legacy function name for backward compatibility.

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Average Hamming distance (0.0 to 1.0, normalized)
    """
    return calculate_hamming_distance_average(population)