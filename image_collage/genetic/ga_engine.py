import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any
import logging
from collections import defaultdict

from ..config.settings import CollageConfig
from .comprehensive_diversity import ComprehensiveDiversityManager, DiversityMetrics
from .spatial_diversity import SpatialDiversityManager
from .island_model import IslandModelManager


class GeneticAlgorithmEngine:
    def __init__(self, config: CollageConfig):
        self.config = config
        self.population: List[np.ndarray] = []
        self.population_size = config.genetic_params.population_size
        self.num_source_images = 0
        self.grid_size = config.grid_size
        self.generation = 0

        # Advanced evolution tracking
        self.fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        self.last_improvement_generation = 0
        self.best_fitness_ever = float('inf')

        # Adaptive parameters
        self.original_mutation_rate = config.genetic_params.mutation_rate
        self.original_crossover_rate = config.genetic_params.crossover_rate

        # Advanced diversity management
        self.comprehensive_diversity_manager = None
        self.spatial_diversity_manager = None
        self.island_model_manager = None

        # Multi-population support
        self.use_island_model = config.genetic_params.enable_island_model if hasattr(config.genetic_params, 'enable_island_model') else False
        self.subpopulations = []
        self.migration_interval = 20
        self.migration_rate = 0.1
        
    def update_config(self, config: CollageConfig) -> None:
        self.config = config
        self.population_size = config.genetic_params.population_size
        self.grid_size = config.grid_size
    
    def initialize_population(self, num_source_images: int) -> None:
        self.num_source_images = num_source_images
        self.population = []
        self.generation = 0

        grid_width, grid_height = self.grid_size
        total_tiles = grid_width * grid_height

        # Initialize diversity managers
        self.comprehensive_diversity_manager = ComprehensiveDiversityManager(
            self.config, self.grid_size, num_source_images
        )
        self.spatial_diversity_manager = SpatialDiversityManager(self.grid_size, num_source_images)

        # Initialize island model if enabled
        if self.use_island_model:
            self.island_model_manager = IslandModelManager(
                self.config, num_islands=4, migration_interval=20, migration_rate=0.1
            )
            self.island_model_manager.initialize_islands(self.grid_size, num_source_images, self.config.allow_duplicate_tiles)
            # Get initial population from island model
            self.population = self.island_model_manager.get_total_population()[:self.population_size]
        else:
            # Standard population initialization
            for _ in range(self.population_size):
                if self.config.allow_duplicate_tiles:
                    individual = np.random.randint(0, num_source_images, size=(grid_height, grid_width))
                else:
                    if num_source_images < total_tiles:
                        logging.warning(f"Not enough source images ({num_source_images}) for grid size ({total_tiles}). Using duplicates.")
                        individual = np.random.randint(0, num_source_images, size=(grid_height, grid_width))
                    else:
                        indices = np.random.choice(num_source_images, size=total_tiles, replace=False)
                        individual = indices.reshape((grid_height, grid_width))

                self.population.append(individual)
    
    def get_population(self) -> List[np.ndarray]:
        return self.population.copy()

    def set_population(self, population: List[np.ndarray]) -> None:
        """Set the population, used for checkpoint restoration."""
        self.population = [individual.copy() for individual in population]
        self.population_size = len(self.population)

    def get_state(self) -> Dict[str, Any]:
        """Get the complete state of the GA engine for checkpointing."""
        return {
            'generation': self.generation,
            'num_source_images': self.num_source_images,
            'fitness_history': self.fitness_history.copy(),
            'diversity_history': self.diversity_history.copy(),
            'stagnation_counter': self.stagnation_counter,
            'last_improvement_generation': self.last_improvement_generation,
            'best_fitness_ever': self.best_fitness_ever,
            'original_mutation_rate': self.original_mutation_rate,
            'original_crossover_rate': self.original_crossover_rate,
            'use_island_model': self.use_island_model,
            'migration_interval': self.migration_interval,
            'migration_rate': self.migration_rate
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the GA engine state from checkpoint."""
        self.generation = state.get('generation', 0)
        # Only update num_source_images if it's in the saved state
        # If not, keep the current value (from loaded source images)
        if 'num_source_images' in state and state['num_source_images'] > 0:
            self.num_source_images = state['num_source_images']
        self.fitness_history = state.get('fitness_history', []).copy()
        self.diversity_history = state.get('diversity_history', []).copy()
        self.stagnation_counter = state.get('stagnation_counter', 0)
        self.last_improvement_generation = state.get('last_improvement_generation', 0)
        self.best_fitness_ever = state.get('best_fitness_ever', float('inf'))
        self.original_mutation_rate = state.get('original_mutation_rate', self.config.genetic_params.mutation_rate)
        self.original_crossover_rate = state.get('original_crossover_rate', self.config.genetic_params.crossover_rate)
        self.use_island_model = state.get('use_island_model', False)
        self.migration_interval = state.get('migration_interval', 20)
        self.migration_rate = state.get('migration_rate', 0.1)

        # Re-initialize diversity managers if needed
        if self.num_source_images > 0:
            self.comprehensive_diversity_manager = ComprehensiveDiversityManager(
                self.config, self.grid_size, self.num_source_images
            )
            self.spatial_diversity_manager = SpatialDiversityManager(self.grid_size, self.num_source_images)

            # Re-initialize island model if it was used
            if self.use_island_model:
                self.island_model_manager = IslandModelManager(
                    self.config, num_islands=4, migration_interval=self.migration_interval,
                    migration_rate=self.migration_rate
                )
    
    def evolve_population(self, fitness_scores: List[float]) -> None:
        if len(fitness_scores) != len(self.population):
            raise ValueError("Number of fitness scores must match population size")

        fitness_scores = np.array(fitness_scores)

        # Track fitness and diversity evolution
        best_fitness = np.min(fitness_scores)
        current_diversity = self.get_population_diversity()

        self.fitness_history.append(best_fitness)
        self.diversity_history.append(current_diversity)

        # Check for improvement
        if best_fitness < self.best_fitness_ever:
            self.best_fitness_ever = best_fitness
            self.last_improvement_generation = self.generation
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # Apply adaptive strategies based on population state
        if self.config.genetic_params.enable_adaptive_parameters:
            self._adaptive_parameter_control(fitness_scores, current_diversity)

        new_population = []

        # Dynamic elitism based on diversity
        elite_count = self._adaptive_elitism_count(current_diversity)
        elite_indices = np.argsort(fitness_scores)[:elite_count]

        # Add elite individuals with potential diversity injection
        for idx in elite_indices:
            diversity_threshold = self.config.genetic_params.diversity_threshold
            if current_diversity < diversity_threshold and random.random() < 0.3:
                # Inject diversity by mutating elite individuals
                mutated_elite = self._diversify_mutate(self.population[idx].copy())
                new_population.append(mutated_elite)
            else:
                new_population.append(self.population[idx].copy())

        # Fill remaining population with advanced breeding strategies
        while len(new_population) < self.population_size:
            if random.random() < self.config.genetic_params.crossover_rate:
                if self.config.genetic_params.enable_adaptive_parameters:
                    parent1, parent2 = self._advanced_parent_selection(fitness_scores, current_diversity)
                else:
                    parent1 = self._tournament_selection(fitness_scores)
                    parent2 = self._tournament_selection(fitness_scores)

                if self.config.genetic_params.enable_advanced_crossover:
                    child1, child2 = self._enhanced_crossover(parent1, parent2)
                else:
                    child1, child2 = self._crossover(parent1, parent2)

                if self.config.genetic_params.enable_advanced_mutation:
                    child1 = self._adaptive_mutate(child1, current_diversity)
                    child2 = self._adaptive_mutate(child2, current_diversity)
                else:
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)

                new_population.extend([child1, child2])
            else:
                parent = self._tournament_selection(fitness_scores)
                if self.config.genetic_params.enable_advanced_mutation:
                    child = self._adaptive_mutate(parent.copy(), current_diversity)
                else:
                    child = self._mutate(parent.copy())
                new_population.append(child)

        # Apply diversity preservation techniques
        new_population = self._ensure_population_diversity(new_population[:self.population_size])

        # Restart mechanism for severe stagnation
        restart_threshold = self.config.genetic_params.restart_threshold
        restart_ratio = self.config.genetic_params.restart_ratio
        if self.stagnation_counter > restart_threshold:
            new_population = self._population_restart(new_population, restart_ratio)
            self.stagnation_counter = 0

        self.population = new_population
        self.generation += 1
    
    def _tournament_selection(self, fitness_scores: np.ndarray) -> np.ndarray:
        tournament_size = self.config.genetic_params.tournament_size
        tournament_indices = np.random.choice(len(self.population), size=tournament_size, replace=False)
        tournament_fitness = fitness_scores[tournament_indices]
        
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        grid_width, grid_height = self.grid_size  # grid_size is (width, height)
        
        if random.random() < 0.5:
            crossover_point1 = random.randint(0, grid_height - 1)
            crossover_point2 = random.randint(crossover_point1, grid_height - 1)
            
            child1[crossover_point1:crossover_point2] = parent2[crossover_point1:crossover_point2]
            child2[crossover_point1:crossover_point2] = parent1[crossover_point1:crossover_point2]
        else:
            crossover_point1 = random.randint(0, grid_width - 1)
            crossover_point2 = random.randint(crossover_point1, grid_width - 1)
            
            child1[:, crossover_point1:crossover_point2] = parent2[:, crossover_point1:crossover_point2]
            child2[:, crossover_point1:crossover_point2] = parent1[:, crossover_point1:crossover_point2]
        
        return child1, child2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        grid_width, grid_height = self.grid_size  # grid_size is (width, height)

        for i in range(grid_height):
            for j in range(grid_width):
                if random.random() < self.config.genetic_params.mutation_rate:
                    if random.random() < 0.7:
                        mutated[i, j] = random.randint(0, self.num_source_images - 1)
                    else:
                        if random.random() < 0.5 and j > 0:
                            mutated[i, j], mutated[i, j-1] = mutated[i, j-1], mutated[i, j]
                        elif i > 0:
                            mutated[i, j], mutated[i-1, j] = mutated[i-1, j], mutated[i, j]
        
        return mutated
    
    def get_generation(self) -> int:
        return self.generation
    
    def get_population_diversity(self) -> float:
        if not self.population:
            return 0.0
        
        unique_individuals = set()
        for individual in self.population:
            unique_individuals.add(tuple(individual.flatten()))
        
        return len(unique_individuals) / len(self.population)
    
    def adaptive_mutation_rate(self, fitness_scores: List[float], target_diversity: float = 0.7) -> None:
        current_diversity = self.get_population_diversity()
        
        if current_diversity < target_diversity:
            self.config.genetic_params.mutation_rate = min(
                0.15, self.config.genetic_params.mutation_rate * 1.1
            )
        else:
            self.config.genetic_params.mutation_rate = max(
                0.01, self.config.genetic_params.mutation_rate * 0.95
            )
    
    def get_stats(self) -> dict:
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'diversity': self.get_population_diversity(),
            'mutation_rate': self.config.genetic_params.mutation_rate,
            'crossover_rate': self.config.genetic_params.crossover_rate,
            'stagnation_counter': self.stagnation_counter,
            'generations_since_improvement': self.generation - self.last_improvement_generation,
        }

    def _adaptive_parameter_control(self, fitness_scores: np.ndarray, diversity: float) -> None:
        """Dynamically adjust genetic parameters based on population state."""
        fitness_std = np.std(fitness_scores)
        stagnation_ratio = self.stagnation_counter / max(self.generation, 1)

        # Increase mutation when diversity is low or stagnation is high
        if diversity < 0.4 or stagnation_ratio > 0.2:
            self.config.genetic_params.mutation_rate = min(
                0.25, self.original_mutation_rate * (2.0 + stagnation_ratio)
            )
        elif diversity > 0.7 and stagnation_ratio < 0.1:
            # Reduce mutation when diversity is high and making progress
            self.config.genetic_params.mutation_rate = max(
                0.01, self.original_mutation_rate * 0.5
            )
        else:
            # Gradual return to original rate
            target_rate = self.original_mutation_rate
            current_rate = self.config.genetic_params.mutation_rate
            self.config.genetic_params.mutation_rate = 0.9 * current_rate + 0.1 * target_rate

        # Adjust crossover rate inversely to mutation
        crossover_adjustment = 1.0 - (self.config.genetic_params.mutation_rate / self.original_mutation_rate - 1.0) * 0.2
        self.config.genetic_params.crossover_rate = np.clip(
            self.original_crossover_rate * crossover_adjustment, 0.5, 0.95
        )

    def _adaptive_elitism_count(self, diversity: float) -> int:
        """Adjust number of elite individuals based on diversity."""
        base_count = max(1, int(self.population_size * self.config.genetic_params.elitism_rate))

        if diversity < 0.3:
            # Reduce elitism when diversity is low
            return max(1, int(base_count * 0.7))
        elif diversity > 0.7:
            # Increase elitism when diversity is high
            return min(int(self.population_size * 0.3), int(base_count * 1.5))
        else:
            return base_count

    def _advanced_parent_selection(self, fitness_scores: np.ndarray, diversity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced parent selection promoting both fitness and diversity."""
        if diversity < 0.4:
            # When diversity is low, encourage selection of diverse parents
            parent1 = self._tournament_selection(fitness_scores)

            # Find a parent that's different from parent1
            attempts = 0
            while attempts < 10:
                parent2 = self._tournament_selection(fitness_scores)
                if not np.array_equal(parent1, parent2):
                    break
                attempts += 1
            else:
                # If can't find different parent, use random selection
                parent2 = self.population[random.randint(0, len(self.population) - 1)].copy()

            return parent1, parent2
        else:
            # Normal tournament selection when diversity is adequate
            return (self._tournament_selection(fitness_scores),
                   self._tournament_selection(fitness_scores))

    def _enhanced_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced crossover with multiple strategies."""
        strategy = random.choice(['uniform', 'block', 'adaptive'])

        if strategy == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        elif strategy == 'block':
            return self._block_crossover(parent1, parent2)
        else:
            return self._crossover(parent1, parent2)  # Original method

    def _uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover - each gene has equal probability from either parent."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        mask = np.random.random(parent1.shape) < 0.5
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]

        return child1, child2

    def _block_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Block crossover - exchange rectangular regions."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        grid_height, grid_width = parent1.shape

        # Random block size (10-30% of grid)
        block_height = random.randint(max(1, grid_height // 10), grid_height // 3)
        block_width = random.randint(max(1, grid_width // 10), grid_width // 3)

        # Random block position
        start_row = random.randint(0, max(0, grid_height - block_height))
        start_col = random.randint(0, max(0, grid_width - block_width))

        end_row = min(start_row + block_height, grid_height)
        end_col = min(start_col + block_width, grid_width)

        # Swap blocks
        child1[start_row:end_row, start_col:end_col] = parent2[start_row:end_row, start_col:end_col]
        child2[start_row:end_row, start_col:end_col] = parent1[start_row:end_row, start_col:end_col]

        return child1, child2

    def _adaptive_mutate(self, individual: np.ndarray, diversity: float) -> np.ndarray:
        """Adaptive mutation based on population diversity and stagnation."""
        base_rate = self.config.genetic_params.mutation_rate

        # Increase mutation intensity when diversity is low
        if diversity < 0.3:
            mutation_rate = min(0.3, base_rate * 2.0)
        elif self.stagnation_counter > 20:
            mutation_rate = min(0.4, base_rate * (1.0 + self.stagnation_counter / 50.0))
        else:
            mutation_rate = base_rate

        return self._enhanced_mutate(individual, mutation_rate)

    def _enhanced_mutate(self, individual: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Enhanced mutation with multiple strategies."""
        mutated = individual.copy()
        grid_height, grid_width = individual.shape

        for i in range(grid_height):
            for j in range(grid_width):
                if random.random() < mutation_rate:
                    mutation_type = random.choice(['replacement', 'swap', 'block_shuffle'])

                    if mutation_type == 'replacement':
                        # Standard replacement mutation
                        mutated[i, j] = random.randint(0, self.num_source_images - 1)

                    elif mutation_type == 'swap' and (j > 0 or i > 0):
                        # Swap with neighbor
                        if random.random() < 0.5 and j > 0:
                            mutated[i, j], mutated[i, j-1] = mutated[i, j-1], mutated[i, j]
                        elif i > 0:
                            mutated[i, j], mutated[i-1, j] = mutated[i-1, j], mutated[i, j]

                    elif mutation_type == 'block_shuffle':
                        # Shuffle a small local region
                        self._local_shuffle_mutation(mutated, i, j)

        return mutated

    def _local_shuffle_mutation(self, individual: np.ndarray, center_i: int, center_j: int) -> None:
        """Shuffle genes in a small local region."""
        grid_height, grid_width = individual.shape
        radius = random.randint(1, 3)

        min_i = max(0, center_i - radius)
        max_i = min(grid_height, center_i + radius + 1)
        min_j = max(0, center_j - radius)
        max_j = min(grid_width, center_j + radius + 1)

        # Check if region is valid (non-empty)
        if max_i > min_i and max_j > min_j:
            # Extract region and shuffle
            region = individual[min_i:max_i, min_j:max_j].flatten()
            # Additional check: ensure region is not empty
            if region.size > 0:
                np.random.shuffle(region)
                individual[min_i:max_i, min_j:max_j] = region.reshape(max_i - min_i, max_j - min_j)

    def _diversify_mutate(self, individual: np.ndarray) -> np.ndarray:
        """High-intensity mutation for diversity injection."""
        mutated = individual.copy()
        grid_height, grid_width = individual.shape

        # Mutate 20-40% of genes with random values
        mutation_count = random.randint(int(0.2 * grid_height * grid_width),
                                       int(0.4 * grid_height * grid_width))

        positions = [(i, j) for i in range(grid_height) for j in range(grid_width)]
        random.shuffle(positions)

        for pos in positions[:mutation_count]:
            i, j = pos
            mutated[i, j] = random.randint(0, self.num_source_images - 1)

        return mutated

    def _ensure_population_diversity(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """Ensure minimum diversity by replacing duplicate individuals."""
        unique_individuals = {}
        diverse_population = []

        for individual in population:
            individual_key = tuple(individual.flatten())
            if individual_key not in unique_individuals:
                unique_individuals[individual_key] = individual
                diverse_population.append(individual)
            else:
                # Replace duplicate with mutated version
                diverse_individual = self._diversify_mutate(individual.copy())
                diverse_population.append(diverse_individual)

        return diverse_population

    def _population_restart(self, current_population: List[np.ndarray], restart_ratio: float) -> List[np.ndarray]:
        """Restart portion of population to escape local optima."""
        restart_count = int(len(current_population) * restart_ratio)
        keep_count = len(current_population) - restart_count

        # Keep the best individuals
        new_population = current_population[:keep_count]

        # Generate new random individuals
        grid_height, grid_width = self.grid_size
        for _ in range(restart_count):
            if self.config.allow_duplicate_tiles:
                individual = np.random.randint(0, self.num_source_images,
                                             size=(grid_height, grid_width))
            else:
                if self.num_source_images >= grid_height * grid_width:
                    indices = np.random.choice(self.num_source_images,
                                             size=grid_height * grid_width, replace=False)
                    individual = indices.reshape((grid_height, grid_width))
                else:
                    individual = np.random.randint(0, self.num_source_images,
                                                 size=(grid_height, grid_width))

            new_population.append(individual)

        return new_population

    def set_population(self, population: List[np.ndarray]) -> None:
        """Set the population from checkpoint restoration.

        Args:
            population: List of individuals to restore
        """
        self.population = [individual.copy() for individual in population]
        self.population_size = len(self.population)

    def get_population(self) -> List[np.ndarray]:
        """Get the current population.

        Returns:
            List of current population individuals
        """
        return self.population.copy()

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore GA engine internal state from checkpoint.

        Args:
            state: Dictionary containing GA engine state
        """
        if 'generation' in state:
            self.generation = state['generation']
        if 'fitness_history' in state:
            self.fitness_history = state['fitness_history'].copy()
        if 'diversity_history' in state:
            self.diversity_history = state['diversity_history'].copy()
        if 'stagnation_counter' in state:
            self.stagnation_counter = state['stagnation_counter']
        if 'last_improvement_generation' in state:
            self.last_improvement_generation = state['last_improvement_generation']
        if 'best_fitness_ever' in state:
            self.best_fitness_ever = state['best_fitness_ever']

    def get_state(self) -> Dict[str, Any]:
        """Get GA engine internal state for checkpoint saving.

        Returns:
            Dictionary containing GA engine state
        """
        return {
            'generation': self.generation,
            'fitness_history': self.fitness_history.copy(),
            'diversity_history': self.diversity_history.copy(),
            'stagnation_counter': self.stagnation_counter,
            'last_improvement_generation': self.last_improvement_generation,
            'best_fitness_ever': self.best_fitness_ever
        }