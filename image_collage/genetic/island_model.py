"""
Island Model implementation for genetic algorithm with multiple populations.

Implements the multi-population island model strategy from DIVERSITY.md to prevent
convergence by maintaining multiple independent populations with periodic migration.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging
import time
from collections import defaultdict

# Import coordinate validation utilities
from ..utils.coordinate_validation import (
    validate_grid_coordinates,
    log_coordinate_interpretation
)


@dataclass
class Island:
    """Represents a single island in the island model."""
    population: List[np.ndarray] = field(default_factory=list)
    fitness_scores: List[float] = field(default_factory=list)
    best_individual: Optional[np.ndarray] = None
    best_fitness: float = float('inf')
    generation: int = 0
    diversity_score: float = 0.0

    # Performance optimization caches
    _fitness_cache: Dict[str, float] = field(default_factory=dict)
    _diversity_cache: float = -1.0
    _diversity_cache_generation: int = -1
    _sorted_fitness_indices: Optional[np.ndarray] = None
    _sorted_fitness_valid: bool = False


class IslandModelManager:
    """Manages multiple populations (islands) with migration between them."""

    def __init__(self, config, num_islands: int = 4, migration_interval: int = 10,
                 migration_rate: float = 0.1, migration_policy: str = "ring"):
        self.config = config
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.migration_policy = migration_policy

        # Calculate island size (distribute total population)
        total_population = config.genetic_params.population_size
        self.island_size = max(10, total_population // num_islands)

        # Initialize islands
        self.islands = [Island() for _ in range(num_islands)]

        # Global tracking
        self.global_best_individual = None
        self.global_best_fitness = float('inf')
        self.migration_history = []

        # Performance optimization settings
        self.diversity_sample_size = min(20, self.island_size // 2)
        self.enable_fitness_cache = True
        self.lazy_diversity_calculation = True

        # Migration policies - optimized versions
        self.migration_policies = {
            "best_to_worst": self._migrate_best_to_worst_optimized,
            "random": self._migrate_random_optimized,
            "ring": self._migrate_ring_optimized,
            "diversity_based": self._migrate_diversity_based_optimized
        }

        logging.info(f"Initialized Island Model with {num_islands} islands of size {self.island_size}")

    def initialize_islands(self, grid_size: Tuple[int, int], num_source_images: int,
                          allow_duplicates: bool = True) -> None:
        """Initialize all islands with random populations."""
        # Use coordinate validation for consistent extraction
        width, height = validate_grid_coordinates(grid_size, "IslandModelManager.initialize_islands")
        grid_width, grid_height = width, height

        # Log coordinate interpretation for debugging
        log_coordinate_interpretation(grid_size, "IslandModelManager")

        for island_idx, island in enumerate(self.islands):
            island.population = []
            island.fitness_scores = []

            for _ in range(self.island_size):
                if allow_duplicates:
                    individual = np.random.randint(0, num_source_images,
                                                 size=(grid_height, grid_width))
                else:
                    if num_source_images >= grid_height * grid_width:
                        indices = np.random.choice(num_source_images,
                                                 size=grid_height * grid_width, replace=False)
                        individual = indices.reshape((grid_height, grid_width))
                    else:
                        individual = np.random.randint(0, num_source_images,
                                                     size=(grid_height, grid_width))

                island.population.append(individual)
                island.fitness_scores.append(float('inf'))  # Will be evaluated

            logging.info(f"Initialized island {island_idx} with {len(island.population)} individuals")

    def evolve_generation(self, fitness_evaluator, ga_engine) -> Dict[str, Any]:
        """Evolve all islands for one generation with optimized performance."""
        start_time = time.time()
        generation_stats = {
            'island_stats': [],
            'migration_occurred': False,
            'global_improvement': False,
            'timing': {}
        }

        is_migration_generation = (self.islands[0].generation % self.migration_interval == 0 and self.islands[0].generation > 0)

        # Evolve each island independently with optimizations
        for island_idx, island in enumerate(self.islands):
            island_start = time.time()

            # Set GA engine population to current island
            ga_engine.population = island.population
            ga_engine.generation = island.generation

            # Evaluate fitness if needed (initial generation)
            if any(f == float('inf') for f in island.fitness_scores):
                island.fitness_scores = self._evaluate_population_fitness(island.population, fitness_evaluator)

            # Store pre-evolution population for incremental fitness evaluation
            pre_evolution_population = [ind.copy() for ind in island.population]
            pre_evolution_fitness = island.fitness_scores.copy()

            # Evolve the island
            ga_engine.evolve_population(island.fitness_scores)

            # Update island state
            island.population = ga_engine.population
            island.generation += 1

            # Incremental fitness evaluation - only evaluate new/changed individuals
            island.fitness_scores = self._incremental_fitness_evaluation(
                island, pre_evolution_population, pre_evolution_fitness, fitness_evaluator
            )

            # Update island best with fast min finding
            self._update_island_best(island)

            # Calculate diversity only when needed (migration generations or first time)
            if is_migration_generation or island.diversity_score == 0.0:
                island.diversity_score = self._calculate_island_diversity_optimized(island.population)
            # Otherwise use cached diversity score

            # Track island stats
            generation_stats['island_stats'].append({
                'island_id': island_idx,
                'best_fitness': island.best_fitness,
                'avg_fitness': np.mean(island.fitness_scores),
                'diversity': island.diversity_score,
                'population_size': len(island.population),
                'evolution_time': time.time() - island_start
            })

        # Check for global improvement
        current_global_best = self.get_global_best()
        if current_global_best[1] < self.global_best_fitness:
            self.global_best_fitness = current_global_best[1]
            self.global_best_individual = current_global_best[0].copy()
            generation_stats['global_improvement'] = True

        # Handle migration with timing
        if is_migration_generation:
            migration_start = time.time()
            migration_stats = self.perform_migration()
            generation_stats['migration_occurred'] = True
            generation_stats['migration_stats'] = migration_stats
            generation_stats['timing']['migration_time'] = time.time() - migration_start

        generation_stats['timing']['total_time'] = time.time() - start_time
        return generation_stats

    def perform_migration(self) -> Dict[str, Any]:
        """Perform migration between islands based on current policy."""
        if self.migration_policy in self.migration_policies:
            migration_stats = self.migration_policies[self.migration_policy]()
        else:
            logging.warning(f"Unknown migration policy: {self.migration_policy}, using random")
            migration_stats = self._migrate_random()

        # Record migration event
        self.migration_history.append({
            'generation': self.islands[0].generation,
            'policy': self.migration_policy,
            'stats': migration_stats
        })

        logging.info(f"Migration completed at generation {self.islands[0].generation}")
        return migration_stats








    def get_global_best(self) -> Tuple[np.ndarray, float]:
        """Get the best individual across all islands."""
        best_individual = None
        best_fitness = float('inf')

        for island in self.islands:
            if island.best_fitness < best_fitness:
                best_fitness = island.best_fitness
                best_individual = island.best_individual

        return best_individual, best_fitness

    def get_total_population(self) -> List[np.ndarray]:
        """Get combined population from all islands."""
        total_population = []
        for island in self.islands:
            total_population.extend(island.population)
        return total_population

    def get_total_fitness_scores(self) -> List[float]:
        """Get combined fitness scores from all islands."""
        total_fitness = []
        for island in self.islands:
            total_fitness.extend(island.fitness_scores)
        return total_fitness

    def update_populations_and_fitness(self, population: List[np.ndarray], fitness_scores: List[float]) -> None:
        """Update island populations with evolved individuals and fitness scores."""
        if len(population) == 0 or len(fitness_scores) == 0:
            return

        # Distribute population back to islands
        pop_per_island = len(population) // self.num_islands
        remaining = len(population) % self.num_islands

        start_idx = 0
        for island_idx, island in enumerate(self.islands):
            # Calculate how many individuals this island gets
            island_size = pop_per_island + (1 if island_idx < remaining else 0)
            end_idx = start_idx + island_size

            # Update island population and fitness
            island.population = population[start_idx:end_idx]
            island.fitness_scores = fitness_scores[start_idx:end_idx]

            # Update island's best individual and fitness
            if len(island.fitness_scores) > 0:
                best_idx = np.argmin(island.fitness_scores)
                island.best_fitness = island.fitness_scores[best_idx]
                island.best_individual = island.population[best_idx].copy()

            start_idx = end_idx

    def get_island_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all islands."""
        stats = {
            'num_islands': self.num_islands,
            'island_size': self.island_size,
            'total_population': sum(len(island.population) for island in self.islands),
            'global_best_fitness': self.global_best_fitness,
            'migration_interval': self.migration_interval,
            'migration_rate': self.migration_rate,
            'migration_policy': self.migration_policy
        }

        # Individual island stats
        island_stats = []
        for i, island in enumerate(self.islands):
            island_stat = {
                'island_id': i,
                'generation': island.generation,
                'population_size': len(island.population),
                'best_fitness': island.best_fitness,
                'avg_fitness': np.mean(island.fitness_scores) if len(island.fitness_scores) > 0 else float('inf'),
                'worst_fitness': max(island.fitness_scores) if len(island.fitness_scores) > 0 else float('inf'),
                'diversity_score': island.diversity_score
            }
            island_stats.append(island_stat)

        stats['islands'] = island_stats
        stats['migration_history_count'] = len(self.migration_history)

        return stats

    def set_migration_policy(self, policy: str) -> None:
        """Change migration policy during evolution."""
        if policy in self.migration_policies:
            self.migration_policy = policy
            logging.info(f"Migration policy changed to: {policy}")
        else:
            logging.warning(f"Unknown migration policy: {policy}")

    def adjust_migration_parameters(self, migration_interval: Optional[int] = None,
                                  migration_rate: Optional[float] = None) -> None:
        """Dynamically adjust migration parameters."""
        if migration_interval is not None:
            self.migration_interval = migration_interval
            logging.info(f"Migration interval changed to: {migration_interval}")

        if migration_rate is not None:
            self.migration_rate = migration_rate
            logging.info(f"Migration rate changed to: {migration_rate}")

    # ============================================================================
    # PHASE 2 & 3: PERFORMANCE OPTIMIZATION METHODS
    # ============================================================================

    def _evaluate_population_fitness(self, population: List[np.ndarray], fitness_evaluator) -> List[float]:
        """Evaluate fitness for entire population with optional caching."""
        fitness_scores = []
        for individual in population:
            if self.enable_fitness_cache:
                # Create a hash of the individual for caching
                ind_hash = hash(individual.tobytes())
                if ind_hash in self.islands[0]._fitness_cache:
                    fitness = self.islands[0]._fitness_cache[ind_hash]
                else:
                    fitness = fitness_evaluator.evaluate(individual)
                    self.islands[0]._fitness_cache[ind_hash] = fitness
            else:
                fitness = fitness_evaluator.evaluate(individual)
            fitness_scores.append(fitness)
        return fitness_scores

    def _incremental_fitness_evaluation(self, island: Island, pre_evolution_population: List[np.ndarray],
                                       pre_evolution_fitness: List[float], fitness_evaluator) -> List[float]:
        """Optimized fitness evaluation - only evaluate new/changed individuals."""
        new_fitness_scores = []

        for i, individual in enumerate(island.population):
            # Try to find this individual in the pre-evolution population
            found_match = False

            # Fast comparison using array equality
            for j, old_individual in enumerate(pre_evolution_population):
                if np.array_equal(individual, old_individual):
                    # Reuse existing fitness
                    new_fitness_scores.append(pre_evolution_fitness[j])
                    found_match = True
                    break

            if not found_match:
                # New individual - evaluate fitness
                if self.enable_fitness_cache:
                    ind_hash = hash(individual.tobytes())
                    if ind_hash in island._fitness_cache:
                        fitness = island._fitness_cache[ind_hash]
                    else:
                        fitness = fitness_evaluator.evaluate(individual)
                        island._fitness_cache[ind_hash] = fitness
                else:
                    fitness = fitness_evaluator.evaluate(individual)
                new_fitness_scores.append(fitness)

        return new_fitness_scores

    def _update_island_best(self, island: Island) -> None:
        """Fast update of island's best individual using numpy operations."""
        if not island.fitness_scores:
            return

        fitness_array = np.array(island.fitness_scores)
        best_idx = np.argmin(fitness_array)
        current_best_fitness = fitness_array[best_idx]

        if current_best_fitness < island.best_fitness:
            island.best_fitness = current_best_fitness
            island.best_individual = island.population[best_idx].copy()

        # Update sorted fitness indices cache
        island._sorted_fitness_indices = np.argsort(fitness_array)
        island._sorted_fitness_valid = True

    def _calculate_island_diversity_optimized(self, population: List[np.ndarray]) -> float:
        """Fast diversity calculation using sampling and vectorization."""
        if len(population) < 2:
            return 0.0

        # Use sampling for large populations
        if len(population) > self.diversity_sample_size:
            sample_indices = np.random.choice(len(population), self.diversity_sample_size, replace=False)
            sample_pop = [population[i] for i in sample_indices]
        else:
            sample_pop = population

        if len(sample_pop) < 2:
            return 0.0

        # Vectorized diversity calculation
        pop_matrix = np.array([ind.flatten() for ind in sample_pop])

        # Calculate pairwise hamming distances using broadcasting
        distances = []
        n_samples = len(pop_matrix)

        # For small samples, use all pairs; for larger samples, use random pairs
        if n_samples <= 10:
            # Use all pairs
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = np.sum(pop_matrix[i] != pop_matrix[j])
                    distances.append(dist)
        else:
            # Use random sampling of pairs for efficiency
            n_pairs = min(50, (n_samples * (n_samples - 1)) // 2)
            for _ in range(n_pairs):
                i, j = np.random.choice(n_samples, 2, replace=False)
                dist = np.sum(pop_matrix[i] != pop_matrix[j])
                distances.append(dist)

        if not distances:
            return 0.0

        avg_distance = np.mean(distances)
        max_possible_distance = len(pop_matrix[0]) if len(pop_matrix) > 0 else 1

        return avg_distance / max_possible_distance

    # ============================================================================
    # OPTIMIZED MIGRATION METHODS
    # ============================================================================

    def _migrate_ring_optimized(self) -> Dict[str, Any]:
        """Optimized ring topology migration using numpy operations."""
        num_migrants = max(1, int(self.island_size * self.migration_rate))
        migrations = []

        for island_idx in range(self.num_islands):
            target_idx = (island_idx + 1) % self.num_islands

            source_island = self.islands[island_idx]
            target_island = self.islands[target_idx]

            # Use cached sorted indices if available
            if source_island._sorted_fitness_valid and source_island._sorted_fitness_indices is not None:
                best_indices = source_island._sorted_fitness_indices[:num_migrants]
            else:
                fitness_array = np.array(source_island.fitness_scores)
                best_indices = np.argpartition(fitness_array, num_migrants)[:num_migrants]

            # Select migrants
            migrants = [source_island.population[i].copy() for i in best_indices]

            # Fast replacement using argpartition
            if target_island._sorted_fitness_valid and target_island._sorted_fitness_indices is not None:
                worst_indices = target_island._sorted_fitness_indices[-num_migrants:]
            else:
                target_fitness = np.array(target_island.fitness_scores)
                worst_indices = np.argpartition(target_fitness, -num_migrants)[-num_migrants:]

            # Perform migration
            for i, migrant in enumerate(migrants):
                if i < len(worst_indices):
                    replace_idx = worst_indices[i]
                    target_island.population[replace_idx] = migrant
                    target_island.fitness_scores[replace_idx] = source_island.fitness_scores[best_indices[i]]

            # Invalidate sorted cache for target island
            target_island._sorted_fitness_valid = False

            migrations.append({
                'source': island_idx,
                'target': target_idx,
                'migrants': len(migrants)
            })

        return {'migrations': migrations, 'total_migrants': len(migrations) * num_migrants}

    def _migrate_best_to_worst_optimized(self) -> Dict[str, Any]:
        """Optimized best-to-worst migration using pre-sorted fitness."""
        # Sort islands by their best fitness
        island_fitness = [(i, island.best_fitness) for i, island in enumerate(self.islands)]
        island_fitness.sort(key=lambda x: x[1])

        num_migrants = max(1, int(self.island_size * self.migration_rate))
        migrations = []

        # Migrate from best half to worst half
        n_islands = len(island_fitness)
        best_half = island_fitness[:n_islands//2]
        worst_half = island_fitness[n_islands//2:]

        for (best_idx, _), (worst_idx, _) in zip(best_half, worst_half):
            source_island = self.islands[best_idx]
            target_island = self.islands[worst_idx]

            # Fast selection using numpy operations
            source_fitness = np.array(source_island.fitness_scores)
            best_indices = np.argpartition(source_fitness, num_migrants)[:num_migrants]

            target_fitness = np.array(target_island.fitness_scores)
            worst_indices = np.argpartition(target_fitness, -num_migrants)[-num_migrants:]

            # Perform migration with fitness transfer
            migrants = [source_island.population[i].copy() for i in best_indices]

            for i, migrant in enumerate(migrants):
                if i < len(worst_indices):
                    replace_idx = worst_indices[i]
                    target_island.population[replace_idx] = migrant
                    target_island.fitness_scores[replace_idx] = source_island.fitness_scores[best_indices[i]]

            # Invalidate caches
            target_island._sorted_fitness_valid = False

            migrations.append({
                'source': best_idx,
                'target': worst_idx,
                'migrants': len(migrants)
            })

        return {'migrations': migrations, 'total_migrants': len(migrations) * num_migrants}

    def _migrate_random_optimized(self) -> Dict[str, Any]:
        """Optimized random migration with minimal overhead."""
        num_migrants = max(1, int(self.island_size * self.migration_rate))
        migrations = []

        for island_idx in range(self.num_islands):
            # Random target selection
            possible_targets = [i for i in range(self.num_islands) if i != island_idx]
            target_idx = random.choice(possible_targets)

            source_island = self.islands[island_idx]
            target_island = self.islands[target_idx]

            # Random selection with numpy
            migrant_indices = np.random.choice(len(source_island.population), num_migrants, replace=False)
            replace_indices = np.random.choice(len(target_island.population), num_migrants, replace=False)

            # Perform migration
            for i, migrant_idx in enumerate(migrant_indices):
                if i < len(replace_indices):
                    replace_idx = replace_indices[i]
                    target_island.population[replace_idx] = source_island.population[migrant_idx].copy()
                    target_island.fitness_scores[replace_idx] = source_island.fitness_scores[migrant_idx]

            # Invalidate caches
            target_island._sorted_fitness_valid = False

            migrations.append({
                'source': island_idx,
                'target': target_idx,
                'migrants': len(migrant_indices)
            })

        return {'migrations': migrations, 'total_migrants': len(migrations) * num_migrants}

    def _migrate_diversity_based_optimized(self) -> Dict[str, Any]:
        """Optimized diversity-based migration with sampling."""
        num_migrants = max(1, int(self.island_size * self.migration_rate))
        migrations = []

        # Calculate diversity for all islands (using cached values when possible)
        island_diversities = []
        for i, island in enumerate(self.islands):
            if island._diversity_cache_generation == island.generation and island._diversity_cache >= 0:
                diversity = island._diversity_cache
            else:
                diversity = self._calculate_island_diversity_optimized(island.population)
                island._diversity_cache = diversity
                island._diversity_cache_generation = island.generation
            island_diversities.append((i, diversity))

        # Sort by diversity
        island_diversities.sort(key=lambda x: x[1])

        # Low diversity islands receive from high diversity islands
        n_islands = len(island_diversities)
        low_diversity = island_diversities[:n_islands//2]
        high_diversity = island_diversities[n_islands//2:]

        for (low_idx, _), (high_idx, _) in zip(low_diversity, high_diversity):
            source_island = self.islands[high_idx]
            target_island = self.islands[low_idx]

            # Select diverse migrants using fast sampling
            migrants = self._select_diverse_migrants_optimized(
                source_island.population, source_island.fitness_scores, num_migrants
            )

            # Replace random individuals (simplified for performance)
            replace_indices = np.random.choice(len(target_island.population), len(migrants), replace=False)

            for i, migrant in enumerate(migrants):
                if i < len(replace_indices):
                    replace_idx = replace_indices[i]
                    target_island.population[replace_idx] = migrant
                    # Re-evaluate fitness for migrant (simplified)
                    target_island.fitness_scores[replace_idx] = float('inf')

            # Invalidate caches
            target_island._sorted_fitness_valid = False

            migrations.append({
                'source': high_idx,
                'target': low_idx,
                'migrants': len(migrants)
            })

        return {'migrations': migrations, 'total_migrants': len(migrations) * num_migrants}

    def _select_diverse_migrants_optimized(self, population: List[np.ndarray],
                                         fitness_scores: List[float], num_migrants: int) -> List[np.ndarray]:
        """Fast diverse migrant selection using sampling."""
        if len(population) <= num_migrants:
            return [ind.copy() for ind in population]

        selected = []

        # Start with best individual
        best_idx = np.argmin(fitness_scores)
        selected.append(population[best_idx].copy())

        if num_migrants == 1:
            return selected

        # For remaining slots, use random sampling for diversity (much faster)
        remaining_indices = [i for i in range(len(population)) if i != best_idx]
        additional_count = min(num_migrants - 1, len(remaining_indices))

        if additional_count > 0:
            additional_indices = np.random.choice(remaining_indices, additional_count, replace=False)
            for idx in additional_indices:
                selected.append(population[idx].copy())

        return selected

    # ============================================================================
    # PHASE 3: ADVANCED ARCHITECTURAL OPTIMIZATIONS
    # ============================================================================

    def enable_performance_mode(self, cache_size_limit: int = 10000) -> None:
        """Enable high-performance mode with aggressive caching and optimizations."""
        self.enable_fitness_cache = True
        self.lazy_diversity_calculation = True
        self.diversity_sample_size = min(15, self.island_size // 3)  # Even more aggressive sampling

        # Clear old caches to prevent memory issues
        for island in self.islands:
            if len(island._fitness_cache) > cache_size_limit:
                # Keep only most recent cache entries
                keys_to_remove = list(island._fitness_cache.keys())[:-cache_size_limit//2]
                for key in keys_to_remove:
                    del island._fitness_cache[key]

        logging.info("Performance mode enabled with aggressive optimizations")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics for optimization analysis."""
        total_cache_size = sum(len(island._fitness_cache) for island in self.islands)
        cache_hit_estimates = {}

        for i, island in enumerate(self.islands):
            cache_hit_estimates[f'island_{i}'] = len(island._fitness_cache)

        return {
            'total_fitness_cache_entries': total_cache_size,
            'cache_hit_estimates': cache_hit_estimates,
            'diversity_sample_size': self.diversity_sample_size,
            'lazy_diversity_enabled': self.lazy_diversity_calculation,
            'performance_mode_enabled': self.enable_fitness_cache,
            'migration_policy': self.migration_policy,
            'optimization_level': 'Phase 3 - Full Architectural Optimization'
        }

    def optimize_for_population_size(self, population_size: int) -> None:
        """Automatically adjust optimization parameters based on population size."""
        if population_size <= 50:
            # Small population - can afford more precision
            self.diversity_sample_size = min(population_size // 2, 15)
            self.enable_fitness_cache = True
        elif population_size <= 200:
            # Medium population - balanced approach
            self.diversity_sample_size = min(population_size // 4, 20)
            self.enable_fitness_cache = True
        else:
            # Large population - aggressive optimization
            self.diversity_sample_size = min(population_size // 6, 25)
            self.enable_fitness_cache = True

        logging.info(f"Optimized parameters for population size {population_size}: "
                    f"diversity_sample_size={self.diversity_sample_size}")

    def clear_performance_caches(self) -> None:
        """Clear all performance caches to free memory."""
        for island in self.islands:
            island._fitness_cache.clear()
            island._diversity_cache = -1.0
            island._diversity_cache_generation = -1
            island._sorted_fitness_indices = None
            island._sorted_fitness_valid = False

        logging.info("All performance caches cleared")