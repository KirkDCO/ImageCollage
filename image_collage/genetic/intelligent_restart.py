"""
Intelligent restart mechanisms for genetic algorithm evolution.

Implements advanced restart strategies as described in DIVERSITY.md to prevent
premature convergence and maintain evolutionary progress.
"""

import numpy as np
import random
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import deque

# Import coordinate validation utilities
from ..utils.coordinate_validation import (
    validate_grid_coordinates,
    log_coordinate_interpretation
)


@dataclass
class RestartConfig:
    """Configuration for intelligent restart system."""
    diversity_threshold: float = 0.1
    stagnation_threshold: int = 50
    elite_preservation_ratio: float = 0.1
    fitness_improvement_threshold: float = 0.001
    enable_adaptive_thresholds: bool = True
    restart_interval_min: int = 25
    restart_interval_max: int = 100


class IntelligentRestartManager:
    """
    Advanced restart manager with multiple trigger conditions.

    Monitors population diversity and fitness stagnation to determine
    optimal restart timing while preserving valuable genetic material.
    """

    def __init__(self, config: RestartConfig, population_size: int, genome_length: int, grid_size: tuple = None):
        self.config = config
        self.population_size = population_size
        self.genome_length = genome_length
        self.grid_size = grid_size  # Store grid_size for proper 2D individual creation

        # State tracking
        self.stagnation_counter = 0
        self.restart_history = []
        self.best_fitness_history = deque(maxlen=config.stagnation_threshold + 10)
        self.diversity_history = deque(maxlen=20)

        # Adaptive thresholds
        self.current_diversity_threshold = config.diversity_threshold
        self.current_stagnation_threshold = config.stagnation_threshold

        logging.info(f"Intelligent restart manager initialized with diversity threshold {config.diversity_threshold}")

    def should_restart(self, diversity_score: float, current_best_fitness: float,
                      generation: int) -> Dict[str, Any]:
        """
        Determine if population should be restarted based on multiple criteria.

        Args:
            diversity_score: Current population diversity (0-1)
            current_best_fitness: Best fitness in current generation
            generation: Current generation number

        Returns:
            Dictionary with restart decision and reasons
        """
        self.best_fitness_history.append(current_best_fitness)
        self.diversity_history.append(diversity_score)

        restart_reasons = []
        restart_urgency = 0  # 0=no restart, 1=suggested, 2=recommended, 3=critical

        # Check diversity-based restart
        if diversity_score < self.current_diversity_threshold:
            restart_reasons.append(f"diversity_critical_{diversity_score:.3f}")
            restart_urgency = max(restart_urgency, 3)
        elif diversity_score < self.current_diversity_threshold * 2:
            restart_reasons.append(f"diversity_low_{diversity_score:.3f}")
            restart_urgency = max(restart_urgency, 1)

        # Check stagnation-based restart
        stagnation_check = self._check_fitness_stagnation()
        if stagnation_check['is_stagnant']:
            restart_reasons.append(f"fitness_stagnation_{stagnation_check['generations']}")
            restart_urgency = max(restart_urgency, 2)

        # Check convergence rate
        convergence_check = self._check_convergence_rate()
        if convergence_check['is_problematic']:
            restart_reasons.append(f"convergence_rate_{convergence_check['rate']:.4f}")
            restart_urgency = max(restart_urgency, 1)

        # Check restart interval
        if self._should_restart_by_interval(generation):
            restart_reasons.append(f"interval_restart_{generation}")
            restart_urgency = max(restart_urgency, 1)

        # Adaptive threshold adjustment
        if self.config.enable_adaptive_thresholds:
            self._update_adaptive_thresholds(diversity_score, restart_urgency > 0)

        should_restart = restart_urgency >= 2  # Restart on recommended or critical

        restart_decision = {
            'should_restart': should_restart,
            'urgency': restart_urgency,
            'reasons': restart_reasons,
            'diversity_score': diversity_score,
            'stagnation_generations': len(self.best_fitness_history) if stagnation_check['is_stagnant'] else 0,
            'generation': generation
        }

        if should_restart:
            logging.info(f"Restart triggered at generation {generation}: {', '.join(restart_reasons)}")

        return restart_decision

    def perform_restart(self, population: List[np.ndarray], fitness_scores: List[float],
                       generation: int, **kwargs) -> Tuple[List[np.ndarray], List[float]]:
        """
        Restart population while preserving elite individuals.

        Args:
            population: Current population
            fitness_scores: Current fitness scores
            generation: Current generation
            **kwargs: Additional context (source_images count, etc.)

        Returns:
            Tuple of (new_population, new_fitness_scores)
        """
        # Sort by fitness and preserve elite
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        elite_count = max(1, int(self.population_size * self.config.elite_preservation_ratio))

        new_population = []
        new_fitness_scores = []

        # Preserve elite individuals
        elite_individuals = []
        for i in range(elite_count):
            idx = sorted_indices[i]
            elite_individuals.append(population[idx].copy())
            new_population.append(population[idx].copy())
            new_fitness_scores.append(fitness_scores[idx])

        # Generate new individuals with diversity strategies
        remaining_slots = self.population_size - elite_count
        new_individuals = self._generate_diverse_restart_population(
            elite_individuals, remaining_slots, **kwargs
        )

        new_population.extend(new_individuals)
        new_fitness_scores.extend([float('inf')] * len(new_individuals))  # Will be evaluated

        # Record restart event
        restart_event = {
            'generation': generation,
            'elite_preserved': elite_count,
            'new_individuals': len(new_individuals),
            'best_fitness_preserved': min(fitness_scores),
            'diversity_at_restart': self.diversity_history[-1] if self.diversity_history else 0.0
        }
        self.restart_history.append(restart_event)

        # Reset counters
        self.stagnation_counter = 0
        self.best_fitness_history.clear()

        logging.info(f"Population restarted: preserved {elite_count} elite, generated {len(new_individuals)} new")

        return new_population, new_fitness_scores

    def _check_fitness_stagnation(self) -> Dict[str, Any]:
        """Check if fitness has stagnated based on improvement history."""
        if len(self.best_fitness_history) < self.current_stagnation_threshold:
            return {'is_stagnant': False, 'generations': 0}

        recent_history = list(self.best_fitness_history)
        start_fitness = recent_history[-self.current_stagnation_threshold]
        current_fitness = recent_history[-1]

        improvement = start_fitness - current_fitness  # Assuming minimization

        is_stagnant = improvement < self.config.fitness_improvement_threshold

        return {
            'is_stagnant': is_stagnant,
            'generations': self.current_stagnation_threshold if is_stagnant else 0,
            'improvement': improvement,
            'threshold': self.config.fitness_improvement_threshold
        }

    def _check_convergence_rate(self) -> Dict[str, Any]:
        """Analyze convergence rate to detect problematic patterns."""
        if len(self.best_fitness_history) < 10:
            return {'is_problematic': False, 'rate': 0.0}

        recent_fitness = list(self.best_fitness_history)[-10:]

        # Calculate improvement rate
        improvements = []
        for i in range(1, len(recent_fitness)):
            improvement = recent_fitness[i-1] - recent_fitness[i]  # Assuming minimization
            improvements.append(improvement)

        avg_improvement_rate = np.mean(improvements)

        # Check if rate is too slow or negative (getting worse)
        is_problematic = avg_improvement_rate < self.config.fitness_improvement_threshold / 10

        return {
            'is_problematic': is_problematic,
            'rate': avg_improvement_rate,
            'recent_improvements': improvements
        }

    def _should_restart_by_interval(self, generation: int) -> bool:
        """Check if restart is due based on interval scheduling."""
        if not self.restart_history:
            return False

        last_restart_gen = self.restart_history[-1]['generation']
        generations_since_restart = generation - last_restart_gen

        # Adaptive interval based on restart success
        success_rate = self._calculate_restart_success_rate()
        if success_rate > 0.7:
            interval = self.config.restart_interval_max
        else:
            interval = self.config.restart_interval_min

        return generations_since_restart >= interval

    def _calculate_restart_success_rate(self) -> float:
        """Calculate success rate of previous restarts."""
        if len(self.restart_history) < 2:
            return 0.5  # Default neutral success rate

        successful_restarts = 0
        for i in range(1, len(self.restart_history)):
            prev_best = self.restart_history[i-1]['best_fitness_preserved']
            curr_best = self.restart_history[i]['best_fitness_preserved']
            if curr_best < prev_best:  # Improvement after restart
                successful_restarts += 1

        return successful_restarts / (len(self.restart_history) - 1)

    def _update_adaptive_thresholds(self, current_diversity: float, restart_triggered: bool) -> None:
        """Update thresholds based on algorithm performance."""
        # Adapt diversity threshold
        if restart_triggered and current_diversity > self.current_diversity_threshold:
            # Restart was triggered by other factors, can relax diversity threshold
            self.current_diversity_threshold = min(
                self.config.diversity_threshold * 1.5,
                self.current_diversity_threshold * 1.1
            )
        elif not restart_triggered and current_diversity < self.current_diversity_threshold * 0.8:
            # Diversity is very low but no restart - tighten threshold
            self.current_diversity_threshold = max(
                self.config.diversity_threshold * 0.5,
                self.current_diversity_threshold * 0.9
            )

    def _generate_diverse_restart_population(self, elite_individuals: List[np.ndarray],
                                           count: int, **kwargs) -> List[np.ndarray]:
        """
        Generate diverse new individuals for restart population.

        Args:
            elite_individuals: Elite individuals to use as reference
            count: Number of new individuals to generate
            **kwargs: Additional context (num_source_images, etc.)

        Returns:
            List of new diverse individuals
        """
        new_individuals = []
        num_source_images = kwargs.get('num_source_images', self.genome_length)

        # Strategy 1: Random individuals (30%)
        random_count = max(1, int(count * 0.3))
        for _ in range(random_count):
            if self.grid_size is not None:
                # Use coordinate validation for consistent extraction
                width, height = validate_grid_coordinates(self.grid_size, "IntelligentRestartManager._generate_diverse_restart_population")
                grid_width, grid_height = width, height
                individual = np.random.randint(0, num_source_images, size=(grid_height, grid_width))
            else:
                # Fallback to 1D if grid_size not available (shouldn't happen in practice)
                individual = np.random.randint(0, num_source_images, size=self.genome_length)
            new_individuals.append(individual)

        # Strategy 2: Mutated elite variants (40%)
        mutation_count = int(count * 0.4)
        for _ in range(mutation_count):
            if elite_individuals:
                base = random.choice(elite_individuals).copy()
                # Heavy mutation for diversity
                mutation_rate = 0.3
                for i in range(len(base)):
                    if random.random() < mutation_rate:
                        base[i] = random.randint(0, num_source_images - 1)
                new_individuals.append(base)
            else:
                # Fallback to random
                if self.grid_size is not None:
                    # Use coordinate validation for consistent extraction
                    width, height = validate_grid_coordinates(self.grid_size, "IntelligentRestartManager._generate_diverse_restart_population")
                    grid_width, grid_height = width, height
                    individual = np.random.randint(0, num_source_images, size=(grid_height, grid_width))
                else:
                    # Fallback to 1D if grid_size not available (shouldn't happen in practice)
                    individual = np.random.randint(0, num_source_images, size=self.genome_length)
                new_individuals.append(individual)

        # Strategy 3: Diverse sampling (30%)
        remaining_count = count - len(new_individuals)
        diverse_individuals = self._generate_diverse_sampling(
            elite_individuals, remaining_count, num_source_images
        )
        new_individuals.extend(diverse_individuals)

        return new_individuals[:count]  # Ensure exact count

    def _generate_diverse_sampling(self, elite_individuals: List[np.ndarray],
                                 count: int, num_source_images: int) -> List[np.ndarray]:
        """Generate individuals using diversity-aware sampling."""
        new_individuals = []

        # Analyze elite patterns to avoid them
        if elite_individuals:
            # Calculate position frequencies in elite
            position_frequencies = {}
            for pos in range(self.genome_length):
                frequencies = {}
                for individual in elite_individuals:
                    value = individual[pos]
                    frequencies[value] = frequencies.get(value, 0) + 1
                position_frequencies[pos] = frequencies

            # Generate individuals avoiding common elite patterns
            for _ in range(count):
                if self.grid_size is not None:
                    # Use coordinate validation for consistent extraction
                    width, height = validate_grid_coordinates(self.grid_size, "IntelligentRestartManager._generate_diverse_sampling")
                    grid_width, grid_height = width, height
                    individual = np.zeros((grid_height, grid_width), dtype=int)
                    # Flatten for processing, then reshape back
                    flat_individual = np.zeros(self.genome_length, dtype=int)
                else:
                    individual = np.zeros(self.genome_length, dtype=int)
                    flat_individual = individual

                for pos in range(self.genome_length):
                    elite_frequencies = position_frequencies[pos]
                    total_elite = len(elite_individuals)

                    # Create bias away from common elite values
                    weights = []
                    values = list(range(num_source_images))

                    for value in values:
                        elite_freq = elite_frequencies.get(value, 0)
                        # Higher weight for less frequent values in elite
                        weight = max(0.1, 1.0 - (elite_freq / total_elite))
                        weights.append(weight)

                    # Normalize weights
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                        flat_individual[pos] = np.random.choice(values, p=weights)
                    else:
                        flat_individual[pos] = random.randint(0, num_source_images - 1)

                # Reshape to proper 2D grid if needed
                if self.grid_size is not None:
                    # Use coordinate validation for consistent extraction
                    width, height = validate_grid_coordinates(self.grid_size, "IntelligentRestartManager._generate_diverse_sampling")
                    grid_width, grid_height = width, height
                    individual = flat_individual.reshape((grid_height, grid_width))

                new_individuals.append(individual)
        else:
            # No elite reference, generate random diverse individuals
            for _ in range(count):
                if self.grid_size is not None:
                    # Use coordinate validation for consistent extraction
                    width, height = validate_grid_coordinates(self.grid_size, "IntelligentRestartManager._generate_diverse_sampling")
                    grid_width, grid_height = width, height
                    individual = np.random.randint(0, num_source_images, size=(grid_height, grid_width))
                else:
                    # Fallback to 1D if grid_size not available (shouldn't happen in practice)
                    individual = np.random.randint(0, num_source_images, size=self.genome_length)
                new_individuals.append(individual)

        return new_individuals

    def get_restart_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about restart performance."""
        if not self.restart_history:
            return {
                'total_restarts': 0,
                'success_rate': 0.0,
                'average_interval': 0,
                'current_thresholds': {
                    'diversity': self.current_diversity_threshold,
                    'stagnation': self.current_stagnation_threshold
                }
            }

        intervals = []
        for i in range(1, len(self.restart_history)):
            interval = (self.restart_history[i]['generation'] -
                       self.restart_history[i-1]['generation'])
            intervals.append(interval)

        return {
            'total_restarts': len(self.restart_history),
            'success_rate': self._calculate_restart_success_rate(),
            'average_interval': np.mean(intervals) if intervals else 0,
            'average_elite_preserved': np.mean([r['elite_preserved'] for r in self.restart_history]),
            'best_fitness_trend': [r['best_fitness_preserved'] for r in self.restart_history],
            'current_thresholds': {
                'diversity': self.current_diversity_threshold,
                'stagnation': self.current_stagnation_threshold
            },
            'adaptive_enabled': self.config.enable_adaptive_thresholds
        }