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


@dataclass
class Island:
    """Represents a single island in the island model."""
    population: List[np.ndarray] = field(default_factory=list)
    fitness_scores: List[float] = field(default_factory=list)
    best_individual: Optional[np.ndarray] = None
    best_fitness: float = float('inf')
    generation: int = 0
    diversity_score: float = 0.0


class IslandModelManager:
    """Manages multiple populations (islands) with migration between them."""

    def __init__(self, config, num_islands: int = 4, migration_interval: int = 10,
                 migration_rate: float = 0.1, migration_policy: str = "best_to_worst"):
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

        # Migration policies
        self.migration_policies = {
            "best_to_worst": self._migrate_best_to_worst,
            "random": self._migrate_random,
            "ring": self._migrate_ring,
            "diversity_based": self._migrate_diversity_based
        }

        logging.info(f"Initialized Island Model with {num_islands} islands of size {self.island_size}")

    def initialize_islands(self, grid_size: Tuple[int, int], num_source_images: int,
                          allow_duplicates: bool = True) -> None:
        """Initialize all islands with random populations."""
        grid_height, grid_width = grid_size

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
        """Evolve all islands for one generation, then handle migration."""
        generation_stats = {
            'island_stats': [],
            'migration_occurred': False,
            'global_improvement': False
        }

        # Evolve each island independently
        for island_idx, island in enumerate(self.islands):
            # Set GA engine population to current island
            ga_engine.population = island.population
            ga_engine.generation = island.generation

            # Evaluate fitness if needed
            if any(f == float('inf') for f in island.fitness_scores):
                island.fitness_scores = []
                for individual in island.population:
                    fitness = fitness_evaluator.evaluate(individual)
                    island.fitness_scores.append(fitness)

            # Evolve the island
            ga_engine.evolve_population(island.fitness_scores)

            # Update island state
            island.population = ga_engine.population
            island.generation += 1

            # Re-evaluate fitness for new individuals
            island.fitness_scores = []
            for individual in island.population:
                fitness = fitness_evaluator.evaluate(individual)
                island.fitness_scores.append(fitness)

            # Update island best
            best_idx = np.argmin(island.fitness_scores)
            current_best_fitness = island.fitness_scores[best_idx]

            if current_best_fitness < island.best_fitness:
                island.best_fitness = current_best_fitness
                island.best_individual = island.population[best_idx].copy()

            # Calculate island diversity (simplified)
            island.diversity_score = self._calculate_island_diversity(island.population)

            # Track island stats
            generation_stats['island_stats'].append({
                'island_id': island_idx,
                'best_fitness': island.best_fitness,
                'avg_fitness': np.mean(island.fitness_scores),
                'diversity': island.diversity_score,
                'population_size': len(island.population)
            })

        # Check for global improvement
        current_global_best = self.get_global_best()
        if current_global_best[1] < self.global_best_fitness:
            self.global_best_fitness = current_global_best[1]
            self.global_best_individual = current_global_best[0].copy()
            generation_stats['global_improvement'] = True

        # Handle migration
        if self.islands[0].generation % self.migration_interval == 0:
            migration_stats = self.perform_migration()
            generation_stats['migration_occurred'] = True
            generation_stats['migration_stats'] = migration_stats

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

    def _migrate_best_to_worst(self) -> Dict[str, Any]:
        """Migrate best individuals from fit islands to least fit islands."""
        # Sort islands by fitness
        island_fitness = [(i, island.best_fitness) for i, island in enumerate(self.islands)]
        island_fitness.sort(key=lambda x: x[1])  # Sort by fitness (ascending = better first)

        num_migrants = max(1, int(self.island_size * self.migration_rate))
        migrations = []

        # Migrate from best half to worst half
        best_half = island_fitness[:len(island_fitness)//2]
        worst_half = island_fitness[len(island_fitness)//2:]

        for (best_idx, _), (worst_idx, _) in zip(best_half, worst_half):
            source_island = self.islands[best_idx]
            target_island = self.islands[worst_idx]

            # Select best individuals from source
            source_fitness = source_island.fitness_scores
            best_indices = np.argsort(source_fitness)[:num_migrants]

            migrants = [source_island.population[i].copy() for i in best_indices]

            # Replace worst individuals in target
            target_fitness = target_island.fitness_scores
            worst_indices = np.argsort(target_fitness)[-num_migrants:]

            for i, migrant in enumerate(migrants):
                replace_idx = worst_indices[i]
                target_island.population[replace_idx] = migrant
                target_island.fitness_scores[replace_idx] = float('inf')  # Will be re-evaluated

            migrations.append({
                'source': best_idx,
                'target': worst_idx,
                'migrants': len(migrants)
            })

        return {'migrations': migrations, 'total_migrants': len(migrations) * num_migrants}

    def _migrate_random(self) -> Dict[str, Any]:
        """Random migration between islands."""
        num_migrants = max(1, int(self.island_size * self.migration_rate))
        migrations = []

        for island_idx in range(self.num_islands):
            # Select random target island (different from source)
            possible_targets = [i for i in range(self.num_islands) if i != island_idx]
            target_idx = random.choice(possible_targets)

            source_island = self.islands[island_idx]
            target_island = self.islands[target_idx]

            # Select random individuals for migration
            migrant_indices = random.sample(range(len(source_island.population)), num_migrants)
            migrants = [source_island.population[i].copy() for i in migrant_indices]

            # Replace random individuals in target
            replace_indices = random.sample(range(len(target_island.population)), num_migrants)

            for i, migrant in enumerate(migrants):
                replace_idx = replace_indices[i]
                target_island.population[replace_idx] = migrant
                target_island.fitness_scores[replace_idx] = float('inf')

            migrations.append({
                'source': island_idx,
                'target': target_idx,
                'migrants': len(migrants)
            })

        return {'migrations': migrations, 'total_migrants': len(migrations) * num_migrants}

    def _migrate_ring(self) -> Dict[str, Any]:
        """Ring topology migration - each island sends to next island in ring."""
        num_migrants = max(1, int(self.island_size * self.migration_rate))
        migrations = []

        for island_idx in range(self.num_islands):
            target_idx = (island_idx + 1) % self.num_islands

            source_island = self.islands[island_idx]
            target_island = self.islands[target_idx]

            # Select best individuals from source
            source_fitness = source_island.fitness_scores
            best_indices = np.argsort(source_fitness)[:num_migrants]
            migrants = [source_island.population[i].copy() for i in best_indices]

            # Replace worst individuals in target
            target_fitness = target_island.fitness_scores
            worst_indices = np.argsort(target_fitness)[-num_migrants:]

            for i, migrant in enumerate(migrants):
                replace_idx = worst_indices[i]
                target_island.population[replace_idx] = migrant
                target_island.fitness_scores[replace_idx] = float('inf')

            migrations.append({
                'source': island_idx,
                'target': target_idx,
                'migrants': len(migrants)
            })

        return {'migrations': migrations, 'total_migrants': len(migrations) * num_migrants}

    def _migrate_diversity_based(self) -> Dict[str, Any]:
        """Migration based on diversity levels - low diversity islands receive immigrants."""
        num_migrants = max(1, int(self.island_size * self.migration_rate))
        migrations = []

        # Calculate diversity for all islands
        island_diversities = [(i, island.diversity_score) for i, island in enumerate(self.islands)]

        # Sort by diversity (ascending = less diverse first)
        island_diversities.sort(key=lambda x: x[1])

        # Low diversity islands receive from high diversity islands
        low_diversity = island_diversities[:len(island_diversities)//2]
        high_diversity = island_diversities[len(island_diversities)//2:]

        for (low_idx, _), (high_idx, _) in zip(low_diversity, high_diversity):
            source_island = self.islands[high_idx]
            target_island = self.islands[low_idx]

            # Select diverse individuals from source (not just best)
            migrants = self._select_diverse_migrants(source_island.population,
                                                   source_island.fitness_scores,
                                                   num_migrants)

            # Replace similar individuals in target
            replace_indices = self._select_similar_individuals(target_island.population, num_migrants)

            for i, migrant in enumerate(migrants):
                if i < len(replace_indices):
                    replace_idx = replace_indices[i]
                    target_island.population[replace_idx] = migrant
                    target_island.fitness_scores[replace_idx] = float('inf')

            migrations.append({
                'source': high_idx,
                'target': low_idx,
                'migrants': len(migrants)
            })

        return {'migrations': migrations, 'total_migrants': len(migrations) * num_migrants}

    def _select_diverse_migrants(self, population: List[np.ndarray], fitness_scores: List[float],
                               num_migrants: int) -> List[np.ndarray]:
        """Select diverse individuals for migration."""
        if len(population) <= num_migrants:
            return [ind.copy() for ind in population]

        selected = []
        remaining_indices = list(range(len(population)))

        # Select first individual (best fitness)
        best_idx = np.argmin(fitness_scores)
        selected.append(population[best_idx].copy())
        remaining_indices.remove(best_idx)

        # Select remaining individuals to maximize diversity
        for _ in range(num_migrants - 1):
            if not remaining_indices:
                break

            best_candidate_idx = None
            best_diversity_score = -1

            for candidate_idx in remaining_indices:
                candidate = population[candidate_idx]

                # Calculate diversity relative to already selected individuals
                diversity_score = 0
                for selected_ind in selected:
                    hamming_dist = np.sum(candidate != selected_ind)
                    diversity_score += hamming_dist

                # Normalize by number of comparisons and individual length
                diversity_score /= (len(selected) * len(candidate))

                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_candidate_idx = candidate_idx

            if best_candidate_idx is not None:
                selected.append(population[best_candidate_idx].copy())
                remaining_indices.remove(best_candidate_idx)

        return selected

    def _select_similar_individuals(self, population: List[np.ndarray], num_to_select: int) -> List[int]:
        """Select most similar individuals for replacement."""
        if len(population) <= num_to_select:
            return list(range(len(population)))

        # Calculate pairwise similarities
        similarity_scores = []
        for i, individual in enumerate(population):
            total_similarity = 0
            for j, other in enumerate(population):
                if i != j:
                    similarity = 1.0 - (np.sum(individual != other) / len(individual))
                    total_similarity += similarity

            avg_similarity = total_similarity / (len(population) - 1) if len(population) > 1 else 0
            similarity_scores.append((i, avg_similarity))

        # Sort by similarity (descending = most similar first)
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in similarity_scores[:num_to_select]]

    def _calculate_island_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate simple diversity measure for an island."""
        if len(population) < 2:
            return 0.0

        total_distance = 0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.sum(population[i] != population[j])
                total_distance += distance
                comparisons += 1

        avg_distance = total_distance / comparisons if comparisons > 0 else 0
        max_possible_distance = len(population[0]) if population else 1

        return avg_distance / max_possible_distance

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