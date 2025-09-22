"""
Fitness sharing implementation for maintaining population diversity.

Implements fitness sharing as described in DIVERSITY.md to reduce fitness
based on local population density, encouraging diversity preservation.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class FitnessSharingConfig:
    """Configuration for fitness sharing parameters."""
    sharing_radius: float = 5.0
    alpha: float = 1.0
    enable_dynamic_radius: bool = True
    min_radius: float = 2.0
    max_radius: float = 10.0


class FitnessSharing:
    """
    Fitness sharing implementation to maintain population diversity.

    Adjusts individual fitness based on local population density to prevent
    overcrowding in high-fitness regions of the solution space.
    """

    def __init__(self, config: FitnessSharingConfig):
        self.config = config
        self.sharing_history = []

    def calculate_shared_fitness(self, population: List[np.ndarray],
                               fitness_scores: List[float]) -> List[float]:
        """
        Adjust fitness based on local population density.

        Args:
            population: List of individuals (genomes)
            fitness_scores: Original fitness scores

        Returns:
            List of shared (adjusted) fitness scores
        """
        if len(population) < 2:
            return fitness_scores.copy()

        shared_fitness = []
        current_radius = self._calculate_dynamic_radius(population) if self.config.enable_dynamic_radius else self.config.sharing_radius

        for i, individual in enumerate(population):
            # Calculate sharing function values with all other individuals
            sharing_sum = 0.0

            for j, other in enumerate(population):
                distance = self._calculate_genetic_distance(individual, other)

                if distance < current_radius:
                    sharing_value = self._sharing_function(distance, current_radius)
                    sharing_sum += sharing_value

            # Adjust fitness by sharing sum (higher sharing_sum = lower effective fitness)
            if sharing_sum > 0:
                shared = fitness_scores[i] / sharing_sum
            else:
                shared = fitness_scores[i]

            shared_fitness.append(shared)

        # Record sharing statistics
        self._record_sharing_stats(fitness_scores, shared_fitness, current_radius)

        return shared_fitness

    def _calculate_genetic_distance(self, individual1: np.ndarray, individual2: np.ndarray) -> float:
        """Calculate genetic distance between two individuals."""
        # Use Hamming distance for discrete genetic representations
        hamming_distance = np.sum(individual1 != individual2)
        return float(hamming_distance)

    def _sharing_function(self, distance: float, radius: float) -> float:
        """
        Calculate sharing function value based on distance.

        Args:
            distance: Genetic distance between individuals
            radius: Sharing radius

        Returns:
            Sharing value (0 to 1)
        """
        if distance >= radius:
            return 0.0
        else:
            return 1.0 - (distance / radius) ** self.config.alpha

    def _calculate_dynamic_radius(self, population: List[np.ndarray]) -> float:
        """
        Dynamically adjust sharing radius based on population diversity.

        Returns:
            Adjusted sharing radius
        """
        if len(population) < 2:
            return self.config.sharing_radius

        # Calculate average genetic distance in population
        total_distance = 0.0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._calculate_genetic_distance(population[i], population[j])
                total_distance += distance
                comparisons += 1

        if comparisons == 0:
            return self.config.sharing_radius

        avg_distance = total_distance / comparisons

        # Adjust radius based on population diversity
        # High diversity -> larger radius (less sharing)
        # Low diversity -> smaller radius (more sharing)
        max_possible_distance = len(population[0])  # Maximum Hamming distance
        diversity_ratio = avg_distance / max_possible_distance

        # Scale radius between min and max based on diversity
        radius_range = self.config.max_radius - self.config.min_radius
        dynamic_radius = self.config.min_radius + (diversity_ratio * radius_range)

        return max(self.config.min_radius, min(self.config.max_radius, dynamic_radius))

    def _record_sharing_stats(self, original_fitness: List[float],
                            shared_fitness: List[float], radius: float) -> None:
        """Record statistics about fitness sharing effects."""
        stats = {
            'radius': radius,
            'original_mean': np.mean(original_fitness),
            'shared_mean': np.mean(shared_fitness),
            'original_std': np.std(original_fitness),
            'shared_std': np.std(shared_fitness),
            'sharing_impact': np.mean(np.array(shared_fitness) / np.array(original_fitness))
        }

        self.sharing_history.append(stats)

    def get_sharing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about fitness sharing effects."""
        if not self.sharing_history:
            return {}

        recent_stats = self.sharing_history[-10:]  # Last 10 generations

        return {
            'average_radius': np.mean([s['radius'] for s in recent_stats]),
            'average_impact': np.mean([s['sharing_impact'] for s in recent_stats]),
            'diversity_improvement': np.mean([s['shared_std'] / s['original_std']
                                            for s in recent_stats if s['original_std'] > 0]),
            'total_generations_tracked': len(self.sharing_history)
        }


class CrowdingReplacement:
    """
    Crowding replacement strategy for diversity maintenance.

    Replaces the most similar individual rather than the worst,
    helping to maintain population diversity.
    """

    def __init__(self, crowding_factor: float = 2.0):
        self.crowding_factor = crowding_factor

    def replace_individual(self, population: List[np.ndarray],
                          fitness_scores: List[float],
                          new_individual: np.ndarray,
                          new_fitness: float) -> int:
        """
        Find the best individual to replace with crowding replacement.

        Args:
            population: Current population
            fitness_scores: Current fitness scores
            new_individual: New individual to add
            new_fitness: Fitness of new individual

        Returns:
            Index of individual to replace
        """
        if len(population) == 0:
            return 0

        # Find most similar individual to new_individual
        min_distance = float('inf')
        most_similar_idx = 0

        for i, individual in enumerate(population):
            distance = np.sum(individual != new_individual)
            if distance < min_distance:
                min_distance = distance
                most_similar_idx = i

        # Replace if new individual is better than most similar
        if new_fitness < fitness_scores[most_similar_idx]:
            return most_similar_idx
        else:
            # Fall back to worst replacement if not better than similar
            return int(np.argmax(fitness_scores))

    def replacement_with_tournament(self, population: List[np.ndarray],
                                  fitness_scores: List[float],
                                  new_individual: np.ndarray,
                                  new_fitness: float,
                                  tournament_size: int = 3) -> int:
        """
        Crowding replacement with tournament selection among similar individuals.

        Args:
            population: Current population
            fitness_scores: Current fitness scores
            new_individual: New individual to add
            new_fitness: Fitness of new individual
            tournament_size: Size of similarity tournament

        Returns:
            Index of individual to replace
        """
        if len(population) < tournament_size:
            return self.replace_individual(population, fitness_scores, new_individual, new_fitness)

        # Calculate distances to all individuals
        distances = []
        for i, individual in enumerate(population):
            distance = np.sum(individual != new_individual)
            distances.append((distance, i))

        # Sort by distance (most similar first)
        distances.sort(key=lambda x: x[0])

        # Select tournament among most similar individuals
        tournament_candidates = distances[:tournament_size]

        # Choose worst fitness among similar individuals
        worst_fitness = -float('inf')
        worst_idx = tournament_candidates[0][1]

        for _, idx in tournament_candidates:
            if fitness_scores[idx] > worst_fitness:
                worst_fitness = fitness_scores[idx]
                worst_idx = idx

        return worst_idx