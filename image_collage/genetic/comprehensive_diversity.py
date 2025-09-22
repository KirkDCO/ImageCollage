"""
Comprehensive diversity metrics and management for genetic algorithms.

Implements the full range of diversity preservation techniques outlined in DIVERSITY.md
including advanced selection strategies, fitness scaling, and diversity-based replacement.
"""

import numpy as np
import random
import math
import statistics
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class DiversityMetrics:
    """Comprehensive diversity metrics for population assessment."""
    # Genetic diversity measures
    hamming_distance_avg: float
    hamming_distance_std: float
    position_wise_entropy: float
    unique_individuals_ratio: float

    # Fitness diversity measures
    fitness_variance: float
    fitness_range: float
    fitness_coefficient_variation: float

    # Structural diversity measures
    cluster_count: int
    population_entropy: float

    # Normalized combined score
    normalized_diversity: float


class ComprehensiveDiversityManager:
    """Advanced diversity management with all techniques from DIVERSITY.md."""

    def __init__(self, config, grid_size: Tuple[int, int], num_source_images: int):
        self.config = config
        self.grid_size = grid_size
        self.num_source_images = num_source_images

        # Diversity tracking
        self.metrics_history = []
        self.adaptation_history = []

        # Adaptive parameters
        self.base_mutation_rate = config.genetic_params.mutation_rate
        self.base_crossover_rate = config.genetic_params.crossover_rate
        self.base_tournament_size = config.genetic_params.tournament_size

        # Selection strategies
        self.selection_methods = ["tournament", "roulette", "rank", "sigma_scaled"]
        self.method_weights = [1.0] * len(self.selection_methods)

        # Diversity thresholds
        self.critical_diversity_threshold = 0.1
        self.low_diversity_threshold = 0.3
        self.high_diversity_threshold = 0.7

    def calculate_comprehensive_diversity(self, population: List[np.ndarray],
                                        fitness_scores: List[float]) -> DiversityMetrics:
        """Calculate comprehensive diversity metrics as per DIVERSITY.md."""

        # Genetic diversity measures
        hamming_avg = self._average_hamming_distance(population)
        hamming_std = self._hamming_distance_std(population)
        position_entropy = self._position_wise_entropy(population)
        unique_ratio = self._unique_individuals_ratio(population)

        # Fitness diversity measures
        fitness_var = np.var(fitness_scores)
        fitness_range = max(fitness_scores) - min(fitness_scores) if fitness_scores else 0
        fitness_cv = np.std(fitness_scores) / np.mean(fitness_scores) if np.mean(fitness_scores) > 0 else 0

        # Structural diversity measures
        cluster_count = self._estimate_genetic_clusters(population)
        pop_entropy = self._population_entropy(population)

        # Create metrics object
        metrics = DiversityMetrics(
            hamming_distance_avg=hamming_avg,
            hamming_distance_std=hamming_std,
            position_wise_entropy=position_entropy,
            unique_individuals_ratio=unique_ratio,
            fitness_variance=fitness_var,
            fitness_range=fitness_range,
            fitness_coefficient_variation=fitness_cv,
            cluster_count=cluster_count,
            population_entropy=pop_entropy,
            normalized_diversity=0.0  # Will be calculated below
        )

        # Calculate normalized diversity score
        metrics.normalized_diversity = self._normalize_diversity_score(metrics, len(population))

        self.metrics_history.append(metrics)
        return metrics

    def _average_hamming_distance(self, population: List[np.ndarray]) -> float:
        """Calculate average pairwise Hamming distance."""
        from ..utils.diversity_metrics import calculate_hamming_distance_average
        # Return unnormalized distance for compatibility with existing code
        normalized_distance = calculate_hamming_distance_average(population)
        if population and len(population) > 0:
            max_distance = population[0].size
            return normalized_distance * max_distance
        return 0.0

    def _hamming_distance_std(self, population: List[np.ndarray]) -> float:
        """Calculate standard deviation of pairwise Hamming distances."""
        from ..utils.diversity_metrics import calculate_hamming_distance_std
        return calculate_hamming_distance_std(population)

    def _position_wise_entropy(self, population: List[np.ndarray]) -> float:
        """Calculate entropy for each position and return average."""
        from ..utils.diversity_metrics import calculate_position_wise_entropy
        return calculate_position_wise_entropy(population)

    def _unique_individuals_ratio(self, population: List[np.ndarray]) -> float:
        """Calculate ratio of unique individuals to total population."""
        from ..utils.diversity_metrics import calculate_unique_individuals_ratio
        return calculate_unique_individuals_ratio(population)

    def _estimate_genetic_clusters(self, population: List[np.ndarray], max_clusters: int = 10) -> int:
        """Estimate number of genetic clusters."""
        from ..utils.diversity_metrics import estimate_genetic_clusters
        return estimate_genetic_clusters(population, max_clusters)

    def _population_entropy(self, population: List[np.ndarray]) -> float:
        """Calculate overall population entropy."""
        from ..utils.diversity_metrics import calculate_population_entropy
        return calculate_population_entropy(population)

    def _normalize_diversity_score(self, metrics: DiversityMetrics, population_size: int) -> float:
        """Combine multiple metrics into normalized diversity score (0-1)."""
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
        # Use grid size to calculate max possible distance
        max_hamming = self.grid_size[0] * self.grid_size[1]
        normalized['hamming_distance_avg'] = min(1.0, metrics.hamming_distance_avg / max_hamming) if max_hamming > 0 else 0.0

        # Position entropy (normalize by max possible entropy)
        max_position_entropy = math.log2(population_size) if population_size > 1 else 1
        normalized['position_wise_entropy'] = metrics.position_wise_entropy / max_position_entropy

        # Unique individuals ratio (already 0-1)
        normalized['unique_individuals_ratio'] = metrics.unique_individuals_ratio

        # Fitness coefficient of variation (cap at 1.0)
        normalized['fitness_coefficient_variation'] = min(1.0, metrics.fitness_coefficient_variation)

        # Cluster count (normalize by reasonable maximum - use sqrt of population for better scaling)
        max_clusters = min(population_size, max(10, int(math.sqrt(population_size))))
        normalized['cluster_count'] = min(1.0, metrics.cluster_count / max_clusters)

        # Population entropy (normalize by max possible)
        max_pop_entropy = math.log2(population_size) if population_size > 1 else 1
        normalized['population_entropy'] = metrics.population_entropy / max_pop_entropy

        # Calculate weighted average
        total_score = sum(weights[key] * normalized[key] for key in weights.keys())
        return total_score

    # Advanced Selection Strategies

    def linear_scaling(self, fitness_scores: List[float], scaling_factor: float = 2.0) -> List[float]:
        """Linear fitness scaling to control selection pressure."""
        if not fitness_scores:
            return []

        min_fitness = min(fitness_scores)
        max_fitness = max(fitness_scores)

        if max_fitness == min_fitness:
            return [1.0] * len(fitness_scores)

        avg_fitness = sum(fitness_scores) / len(fitness_scores)

        scaled_scores = []
        for fitness in fitness_scores:
            if max_fitness != avg_fitness:
                scaled = avg_fitness + (fitness - avg_fitness) * scaling_factor / (max_fitness - avg_fitness)
                scaled_scores.append(max(0.1, scaled))
            else:
                scaled_scores.append(1.0)

        return scaled_scores

    def sigma_scaling(self, fitness_scores: List[float], c: float = 2.0) -> List[float]:
        """Sigma scaling to normalize fitness based on population standard deviation."""
        if len(fitness_scores) <= 1:
            return [1.0] * len(fitness_scores)

        mean_fitness = statistics.mean(fitness_scores)
        std_fitness = statistics.stdev(fitness_scores)

        if std_fitness == 0:
            return [1.0] * len(fitness_scores)

        scaled_scores = []
        for fitness in fitness_scores:
            scaled = 1.0 + (fitness - mean_fitness) / (c * std_fitness)
            scaled_scores.append(max(0.1, scaled))

        return scaled_scores

    def rank_based_selection_weights(self, population_size: int, selection_pressure: float = 1.5) -> List[float]:
        """Generate selection weights based on fitness rank."""
        weights = []
        for rank in range(population_size):
            weight = 2 - selection_pressure + 2 * (selection_pressure - 1) * rank / (population_size - 1)
            weights.append(weight)
        return weights

    def multi_modal_selection(self, population: List[np.ndarray], fitness_scores: List[float],
                            num_parents: int, diversity_score: float) -> List[np.ndarray]:
        """Use multiple selection methods based on diversity state."""
        parents = []

        # Adjust method weights based on diversity
        if diversity_score < self.low_diversity_threshold:
            # Low diversity - favor more diverse selection methods
            method_weights = [0.2, 0.3, 0.3, 0.2]  # Less tournament, more roulette/rank
        elif diversity_score > self.high_diversity_threshold:
            # High diversity - can use more selective methods
            method_weights = [0.5, 0.2, 0.2, 0.1]  # More tournament
        else:
            # Balanced diversity
            method_weights = [0.3, 0.25, 0.25, 0.2]

        for i in range(num_parents):
            method = random.choices(self.selection_methods, weights=method_weights)[0]

            if method == "tournament":
                parent = self._tournament_selection(population, fitness_scores)
            elif method == "roulette":
                parent = self._roulette_selection(population, fitness_scores)
            elif method == "rank":
                parent = self._rank_selection(population, fitness_scores)
            elif method == "sigma_scaled":
                parent = self._sigma_scaled_selection(population, fitness_scores)

            parents.append(parent)

        return parents

    def _tournament_selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        """Tournament selection with adaptive tournament size."""
        tournament_size = self.config.genetic_params.tournament_size
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()

    def _roulette_selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        """Roulette wheel selection (fitness proportionate)."""
        # Convert to maximization problem (invert fitness for minimization)
        max_fitness = max(fitness_scores) if fitness_scores else 1
        inverted_fitness = [max_fitness - f + 1 for f in fitness_scores]

        total_fitness = sum(inverted_fitness)
        if total_fitness == 0:
            return random.choice(population).copy()

        selection_probs = [f / total_fitness for f in inverted_fitness]
        selected_idx = np.random.choice(len(population), p=selection_probs)
        return population[selected_idx].copy()

    def _rank_selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        """Rank-based selection."""
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        rank_weights = self.rank_based_selection_weights(len(population))

        # Normalize weights
        total_weight = sum(rank_weights)
        selection_probs = [w / total_weight for w in rank_weights]

        selected_rank = np.random.choice(len(population), p=selection_probs)
        selected_idx = sorted_indices[selected_rank]
        return population[selected_idx].copy()

    def _sigma_scaled_selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        """Selection using sigma-scaled fitness."""
        scaled_fitness = self.sigma_scaling(fitness_scores)
        total_fitness = sum(scaled_fitness)

        if total_fitness == 0:
            return random.choice(population).copy()

        selection_probs = [f / total_fitness for f in scaled_fitness]
        selected_idx = np.random.choice(len(population), p=selection_probs)
        return population[selected_idx].copy()

    # Advanced Replacement Strategies

    def crowding_replacement(self, population: List[np.ndarray], fitness_scores: List[float],
                           new_individual: np.ndarray, new_fitness: float) -> int:
        """Replace most similar individual rather than worst."""
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
            # Fall back to worst replacement
            return fitness_scores.index(max(fitness_scores))

    def fitness_sharing(self, population: List[np.ndarray], fitness_scores: List[float],
                       sharing_radius: float = 5.0, alpha: float = 1.0) -> List[float]:
        """Adjust fitness based on local population density."""
        shared_fitness = []

        for i, individual in enumerate(population):
            # Calculate sharing function values
            sharing_sum = 0
            for j, other in enumerate(population):
                distance = np.sum(individual != other)
                if distance < sharing_radius:
                    sharing_value = 1 - (distance / sharing_radius) ** alpha
                    sharing_sum += sharing_value

            # Adjust fitness by sharing sum
            if sharing_sum > 0:
                shared = fitness_scores[i] / sharing_sum
            else:
                shared = fitness_scores[i]

            shared_fitness.append(shared)

        return shared_fitness

    def dissimilar_parent_selection(self, population: List[np.ndarray], fitness_scores: List[float],
                                  diversity_weight: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Select parents that are both fit and genetically diverse."""

        def combined_score(idx1: int, idx2: int) -> float:
            # Fitness component (lower is better)
            fitness_component = (fitness_scores[idx1] + fitness_scores[idx2]) / 2

            # Diversity component (higher Hamming distance is better)
            hamming_dist = np.sum(population[idx1] != population[idx2])
            max_possible_dist = len(population[idx1])
            diversity_component = hamming_dist / max_possible_dist

            # Combined score (lower is better, so subtract diversity)
            return fitness_component - diversity_weight * diversity_component

        best_score = float('inf')
        best_parents = None

        # Sample parent pairs and find best combination
        num_samples = min(100, len(population) * (len(population) - 1) // 2)
        for _ in range(num_samples):
            idx1, idx2 = random.sample(range(len(population)), 2)
            score = combined_score(idx1, idx2)

            if score < best_score:
                best_score = score
                best_parents = (population[idx1].copy(), population[idx2].copy())

        return best_parents if best_parents else (population[0].copy(), population[1].copy())

    def adaptive_parameter_control(self, diversity_metrics: DiversityMetrics, generation: int) -> Dict[str, float]:
        """Dynamically adjust GA parameters based on diversity state."""
        diversity_score = diversity_metrics.normalized_diversity

        adaptations = {}

        # Mutation rate adaptation
        if diversity_score < self.critical_diversity_threshold:
            # Critical - significant increase
            new_rate = min(0.4, self.base_mutation_rate * 3.0)
            adaptations['mutation_rate'] = new_rate
        elif diversity_score < self.low_diversity_threshold:
            # Low diversity - moderate increase
            new_rate = min(0.25, self.base_mutation_rate * 2.0)
            adaptations['mutation_rate'] = new_rate
        elif diversity_score > self.high_diversity_threshold:
            # High diversity - can reduce
            new_rate = max(0.01, self.base_mutation_rate * 0.7)
            adaptations['mutation_rate'] = new_rate

        # Tournament size adaptation (selection pressure)
        if diversity_score < self.low_diversity_threshold:
            # Reduce selection pressure
            new_size = max(2, self.base_tournament_size - 2)
            adaptations['tournament_size'] = new_size
        elif diversity_score > self.high_diversity_threshold:
            # Can increase selection pressure
            new_size = min(10, self.base_tournament_size + 1)
            adaptations['tournament_size'] = new_size

        # Record adaptation
        self.adaptation_history.append({
            'generation': generation,
            'diversity_score': diversity_score,
            'adaptations': adaptations.copy()
        })

        return adaptations

    def should_perform_intervention(self, diversity_metrics: DiversityMetrics, generation: int) -> List[str]:
        """Determine what interventions are needed based on diversity state."""
        interventions = []
        diversity_score = diversity_metrics.normalized_diversity

        if diversity_score < self.critical_diversity_threshold:
            interventions.extend(['parameter_adaptation', 'population_restart', 'immigration'])
        elif diversity_score < self.low_diversity_threshold:
            interventions.extend(['parameter_adaptation', 'immigration'])

        # Check for stagnation
        if len(self.metrics_history) >= 20:
            recent_diversity = [m.normalized_diversity for m in self.metrics_history[-20:]]
            diversity_change = max(recent_diversity) - min(recent_diversity)
            if diversity_change < 0.05:
                interventions.append('stagnation_restart')

        return interventions