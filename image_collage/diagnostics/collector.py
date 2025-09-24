import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenerationStats:
    """Enhanced statistics for a single generation with comprehensive diversity metrics."""
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    fitness_std: float
    best_individual: np.ndarray
    processing_time: float
    beneficial_mutations: int = 0
    beneficial_crossovers: int = 0
    population_diversity: float = 0.0
    convergence_metric: float = 0.0

    # Enhanced diversity metrics from comprehensive diversity manager
    hamming_diversity: float = 0.0
    entropy_diversity: float = 0.0
    spatial_diversity: float = 0.0
    cluster_diversity: float = 0.0

    # New comprehensive diversity metrics
    hamming_distance_avg: float = 0.0
    hamming_distance_std: float = 0.0
    position_wise_entropy: float = 0.0
    unique_individuals_ratio: float = 0.0
    fitness_coefficient_variation: float = 0.0
    cluster_count: int = 0
    population_entropy: float = 0.0
    normalized_diversity: float = 0.0

    # Spatial diversity metrics
    local_pattern_entropy: float = 0.0
    spatial_clustering: float = 0.0
    edge_pattern_diversity: float = 0.0
    quadrant_diversity: float = 0.0
    neighbor_similarity: float = 0.0
    tile_distribution_variance: float = 0.0
    contiguous_regions: float = 0.0
    spatial_autocorrelation: float = 0.0
    spatial_diversity_score: float = 0.0

    # Advanced evolution metrics
    selection_pressure: float = 0.0
    fitness_variance: float = 0.0
    age_distribution: Dict[int, int] = field(default_factory=dict)
    birth_method_counts: Dict[str, int] = field(default_factory=dict)

    # Island model metrics (if enabled)
    island_diversities: List[float] = field(default_factory=list)
    migration_events: int = 0
    inter_island_diversity: float = 0.0

    # Adaptive parameters (if enabled)
    current_mutation_rate: float = 0.0
    current_crossover_rate: float = 0.0
    restart_events: int = 0
    stagnation_counter: int = 0


@dataclass
class DiagnosticsData:
    """Complete diagnostics data for an evolution run."""
    generations: List[GenerationStats] = field(default_factory=list)
    total_mutations: int = 0
    total_crossovers: int = 0
    beneficial_mutation_rate: float = 0.0
    beneficial_crossover_rate: float = 0.0
    final_convergence: float = 0.0
    total_processing_time: float = 0.0
    config_used: Dict[str, Any] = field(default_factory=dict)
    migration_events: List[Dict[str, Any]] = field(default_factory=list)


class DiagnosticsCollector:
    """Collects detailed diagnostics during genetic algorithm evolution."""

    def __init__(self, config):
        self.config = config
        self.data = DiagnosticsData()
        self.current_generation = 0
        self.start_time = 0.0
        self.generation_start_time = 0.0

        # Track genetic operations
        self.mutation_attempts = 0
        self.crossover_attempts = 0
        self.beneficial_mutations = 0
        self.beneficial_crossovers = 0

        # Previous generation data for comparison
        self.previous_population = None
        self.previous_fitness_scores = None

    def start_evolution(self):
        """Called at the start of evolution."""
        self.start_time = time.time()
        self.data.config_used = {
            'grid_size': self.config.grid_size,
            'max_generations': self.config.genetic_params.max_generations,
            'population_size': self.config.genetic_params.population_size,
            'mutation_rate': self.config.genetic_params.mutation_rate,
            'crossover_rate': self.config.genetic_params.crossover_rate,
            'fitness_weights': {
                'color': self.config.fitness_weights.color,
                'luminance': self.config.fitness_weights.luminance,
                'texture': self.config.fitness_weights.texture,
                'edges': self.config.fitness_weights.edges,
            }
        }

    def start_generation(self, generation: int):
        """Called at the start of each generation."""
        self.current_generation = generation
        self.generation_start_time = time.time()

    def record_generation(self, population: List[np.ndarray], fitness_scores: List[float]):
        """Record statistics for the current generation."""
        generation_time = time.time() - self.generation_start_time

        # Basic fitness statistics
        best_idx = np.argmin(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        average_fitness = np.mean(fitness_scores)
        worst_fitness = np.max(fitness_scores)
        fitness_std = np.std(fitness_scores)

        # Population diversity (average pairwise differences)
        diversity = self._calculate_population_diversity(population)

        # Enhanced diversity metrics (backward compatibility)
        hamming_div = self._calculate_hamming_diversity(population)
        entropy_div = self._calculate_entropy_diversity(population)
        spatial_div = self._calculate_spatial_diversity(population)
        cluster_div = self._calculate_cluster_diversity(population)

        # New comprehensive diversity metrics
        comprehensive_metrics = None
        spatial_metrics = None

        try:
            # Try to get comprehensive diversity metrics if manager available
            from ..genetic.comprehensive_diversity import ComprehensiveDiversityManager
            if hasattr(self.config, 'grid_size') and hasattr(self.config, 'num_source_images'):
                comp_manager = ComprehensiveDiversityManager(
                    self.config,
                    self.config.grid_size,
                    getattr(self.config, 'num_source_images', 1000)
                )
                comprehensive_metrics = comp_manager.calculate_comprehensive_diversity(population, fitness_scores)
        except (ImportError, AttributeError):
            pass

        try:
            # Try to get spatial diversity metrics if manager available
            from ..genetic.spatial_diversity import SpatialDiversityManager
            if hasattr(self.config, 'grid_size'):
                spatial_manager = SpatialDiversityManager(
                    self.config.grid_size,
                    getattr(self.config, 'num_source_images', 1000)
                )
                # Flatten 2D population arrays to 1D for spatial manager
                flattened_population = [individual.flatten() for individual in population]
                spatial_metrics = spatial_manager.calculate_spatial_diversity(flattened_population)
        except (ImportError, AttributeError):
            pass

        # Advanced evolution metrics - use standardized selection pressure definition
        from ..cli.helpers import calculate_selection_pressure
        selection_pressure = calculate_selection_pressure(fitness_scores)
        fitness_variance = np.var(fitness_scores)

        # Convergence metric (improvement rate)
        convergence = 0.0
        if len(self.data.generations) > 0:
            prev_best = self.data.generations[-1].best_fitness
            convergence = (prev_best - best_fitness) / max(abs(prev_best), 1e-6)

        # Count beneficial operations this generation
        gen_beneficial_mutations = 0
        gen_beneficial_crossovers = 0

        if self.previous_fitness_scores is not None:
            gen_beneficial_mutations = self._count_beneficial_mutations(
                population, fitness_scores
            )
            gen_beneficial_crossovers = self._count_beneficial_crossovers(
                population, fitness_scores
            )

        stats = GenerationStats(
            generation=self.current_generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            worst_fitness=worst_fitness,
            fitness_std=fitness_std,
            best_individual=population[best_idx].copy(),
            processing_time=generation_time,
            beneficial_mutations=gen_beneficial_mutations,
            beneficial_crossovers=gen_beneficial_crossovers,
            population_diversity=diversity,
            convergence_metric=convergence,
            # Enhanced diversity metrics (backward compatibility)
            hamming_diversity=hamming_div,
            entropy_diversity=entropy_div,
            spatial_diversity=spatial_div,
            cluster_diversity=cluster_div,
            # New comprehensive diversity metrics
            hamming_distance_avg=comprehensive_metrics.hamming_distance_avg if comprehensive_metrics else hamming_div,
            hamming_distance_std=comprehensive_metrics.hamming_distance_std if comprehensive_metrics else 0.0,
            position_wise_entropy=comprehensive_metrics.position_wise_entropy if comprehensive_metrics else entropy_div,
            unique_individuals_ratio=comprehensive_metrics.unique_individuals_ratio if comprehensive_metrics else (1.0 if population else 0.0),
            fitness_coefficient_variation=comprehensive_metrics.fitness_coefficient_variation if comprehensive_metrics else 0.0,
            cluster_count=comprehensive_metrics.cluster_count if comprehensive_metrics else max(1, len(set(tuple(ind.flatten()) for ind in population[:10]))),
            population_entropy=comprehensive_metrics.population_entropy if comprehensive_metrics else 0.0,
            normalized_diversity=comprehensive_metrics.normalized_diversity if comprehensive_metrics else diversity,
            # Spatial diversity metrics
            local_pattern_entropy=spatial_metrics['local_pattern_entropy'] if spatial_metrics else 0.0,
            spatial_clustering=spatial_metrics['spatial_clustering'] if spatial_metrics else 0.0,
            edge_pattern_diversity=spatial_metrics['edge_pattern_diversity'] if spatial_metrics else 0.0,
            quadrant_diversity=spatial_metrics['quadrant_diversity'] if spatial_metrics else 0.0,
            neighbor_similarity=spatial_metrics['neighbor_similarity'] if spatial_metrics else 0.0,
            tile_distribution_variance=spatial_metrics['tile_distribution_variance'] if spatial_metrics else 0.0,
            contiguous_regions=spatial_metrics['contiguous_regions'] if spatial_metrics else 0.0,
            spatial_autocorrelation=spatial_metrics['spatial_autocorrelation'] if spatial_metrics else 0.0,
            spatial_diversity_score=spatial_metrics['spatial_diversity_score'] if spatial_metrics else spatial_div,
            # Advanced evolution metrics
            selection_pressure=selection_pressure,
            fitness_variance=fitness_variance,
            # Current parameters (to be filled by GA engine if available)
            current_mutation_rate=getattr(self.config.genetic_params, 'mutation_rate', 0.0),
            current_crossover_rate=getattr(self.config.genetic_params, 'crossover_rate', 0.0)
        )

        self.data.generations.append(stats)

        # Update totals
        self.beneficial_mutations += gen_beneficial_mutations
        self.beneficial_crossovers += gen_beneficial_crossovers

        # Store for next generation comparison
        self.previous_population = [ind.copy() for ind in population]
        self.previous_fitness_scores = fitness_scores.copy()

    def record_genetic_operation(self, operation: str, beneficial: bool):
        """Record the result of a genetic operation."""
        if operation == 'mutation':
            self.mutation_attempts += 1
            if beneficial:
                self.beneficial_mutations += 1
        elif operation == 'crossover':
            self.crossover_attempts += 1
            if beneficial:
                self.beneficial_crossovers += 1

    def finish_evolution(self):
        """Called when evolution completes.

        Calculates accurate success rates based on estimated total operations
        across all generations rather than individual tracked attempts.
        Computes final convergence using fitness variance over recent generations.
        """
        self.data.total_processing_time = time.time() - self.start_time
        self.data.total_mutations = self.mutation_attempts
        self.data.total_crossovers = self.crossover_attempts

        # Calculate rates based on total beneficial operations across all generations
        total_beneficial_mutations = sum(g.beneficial_mutations for g in self.data.generations)
        total_beneficial_crossovers = sum(g.beneficial_crossovers for g in self.data.generations)

        # Estimate total operations attempted across all generations
        total_generations = len(self.data.generations)
        if total_generations > 0:
            population_size = self.data.config_used.get('population_size', 100)
            mutation_rate = self.data.config_used.get('mutation_rate', 0.05)
            crossover_rate = self.data.config_used.get('crossover_rate', 0.8)

            # Estimate total operations attempted
            estimated_mutations = total_generations * population_size * mutation_rate
            estimated_crossovers = total_generations * population_size * crossover_rate / 2

            if estimated_mutations > 0:
                self.data.beneficial_mutation_rate = total_beneficial_mutations / estimated_mutations
            if estimated_crossovers > 0:
                self.data.beneficial_crossover_rate = total_beneficial_crossovers / estimated_crossovers

        if len(self.data.generations) > 10:
            # Calculate final convergence as improvement over last 10 generations
            recent_best = [g.best_fitness for g in self.data.generations[-10:]]
            self.data.final_convergence = (max(recent_best) - min(recent_best)) / max(abs(max(recent_best)), 1e-6)
        elif len(self.data.generations) > 1:
            # Use all generations if fewer than 10
            all_best = [g.best_fitness for g in self.data.generations]
            self.data.final_convergence = (max(all_best) - min(all_best)) / max(abs(max(all_best)), 1e-6)

    def save_to_folder(self, output_folder: str):
        """Save all diagnostics data to a folder."""
        folder_path = Path(output_folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        # Save raw data as JSON
        self._save_json_data(folder_path)

        # Save summary statistics as text
        self._save_summary_text(folder_path)

        # Save generation-by-generation CSV
        self._save_csv_data(folder_path)

    def _calculate_population_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate average pairwise diversity in population."""
        from ..utils.diversity_metrics import calculate_population_diversity
        return calculate_population_diversity(population)

    def _count_beneficial_mutations(self, population: List[np.ndarray],
                                  fitness_scores: List[float]) -> int:
        """Estimate beneficial mutations based on fitness improvements."""
        if self.previous_fitness_scores is None:
            return 0

        improvements = 0
        for current, previous in zip(fitness_scores, self.previous_fitness_scores):
            if current < previous:  # Lower fitness is better
                improvements += 1

        # Rough estimate: assume mutations contributed to improvements
        mutation_rate = self.config.genetic_params.mutation_rate
        estimated_mutations = len(population) * mutation_rate
        return min(improvements, int(estimated_mutations))

    def _count_beneficial_crossovers(self, population: List[np.ndarray],
                                   fitness_scores: List[float]) -> int:
        """Estimate beneficial crossovers based on fitness improvements."""
        if self.previous_fitness_scores is None:
            return 0

        improvements = 0
        for current, previous in zip(fitness_scores, self.previous_fitness_scores):
            if current < previous:
                improvements += 1

        # Rough estimate: assume crossovers contributed to improvements
        crossover_rate = self.config.genetic_params.crossover_rate
        estimated_crossovers = len(population) * crossover_rate / 2  # Two parents per crossover
        return min(improvements, int(estimated_crossovers))

    def _save_json_data(self, folder_path: Path):
        """Save complete diagnostics data as JSON."""
        # Convert numpy arrays to lists for JSON serialization
        json_data = {
            'config': self.data.config_used,
            'summary': {
                'total_generations': len(self.data.generations),
                'total_processing_time': self.data.total_processing_time,
                'beneficial_mutation_rate': self.data.beneficial_mutation_rate,
                'beneficial_crossover_rate': self.data.beneficial_crossover_rate,
                'final_convergence': self.data.final_convergence,
            },
            'generations': []
        }

        for gen in self.data.generations:
            gen_data = {
                'generation': gen.generation,
                'best_fitness': gen.best_fitness,
                'average_fitness': gen.average_fitness,
                'worst_fitness': gen.worst_fitness,
                'fitness_std': gen.fitness_std,
                'processing_time': gen.processing_time,
                'beneficial_mutations': gen.beneficial_mutations,
                'beneficial_crossovers': gen.beneficial_crossovers,
                'population_diversity': gen.population_diversity,
                'convergence_metric': gen.convergence_metric,
                # Enhanced diversity metrics
                'hamming_diversity': gen.hamming_diversity,
                'entropy_diversity': gen.entropy_diversity,
                'spatial_diversity': gen.spatial_diversity,
                'cluster_diversity': gen.cluster_diversity,
                # Comprehensive diversity metrics
                'hamming_distance_avg': gen.hamming_distance_avg,
                'hamming_distance_std': gen.hamming_distance_std,
                'position_wise_entropy': gen.position_wise_entropy,
                'unique_individuals_ratio': gen.unique_individuals_ratio,
                'fitness_coefficient_variation': gen.fitness_coefficient_variation,
                'cluster_count': gen.cluster_count,
                'population_entropy': gen.population_entropy,
                'normalized_diversity': gen.normalized_diversity,
                # Spatial diversity metrics
                'local_pattern_entropy': gen.local_pattern_entropy,
                'spatial_clustering': gen.spatial_clustering,
                'edge_pattern_diversity': gen.edge_pattern_diversity,
                'quadrant_diversity': gen.quadrant_diversity,
                'neighbor_similarity': gen.neighbor_similarity,
                'tile_distribution_variance': gen.tile_distribution_variance,
                'contiguous_regions': gen.contiguous_regions,
                'spatial_autocorrelation': gen.spatial_autocorrelation,
                'spatial_diversity_score': gen.spatial_diversity_score,
                # Advanced metrics
                'selection_pressure': gen.selection_pressure,
                'fitness_variance': gen.fitness_variance,
                'current_mutation_rate': gen.current_mutation_rate,
                'current_crossover_rate': gen.current_crossover_rate
            }
            json_data['generations'].append(gen_data)

        # Add migration events to the JSON output
        json_data['migration_events'] = self.data.migration_events

        with open(folder_path / 'diagnostics_data.json', 'w') as f:
            json.dump(json_data, f, indent=2)

    def _save_summary_text(self, folder_path: Path):
        """Save human-readable summary."""
        summary = []
        summary.append("=== GENETIC ALGORITHM DIAGNOSTICS SUMMARY ===\n")

        summary.append(f"Configuration:")
        summary.append(f"  Grid Size: {self.data.config_used['grid_size']}")
        summary.append(f"  Population Size: {self.data.config_used['population_size']}")
        summary.append(f"  Mutation Rate: {self.data.config_used['mutation_rate']:.3f}")
        summary.append(f"  Crossover Rate: {self.data.config_used['crossover_rate']:.3f}")
        summary.append("")

        summary.append(f"Evolution Results:")
        summary.append(f"  Generations Completed: {len(self.data.generations)}")
        summary.append(f"  Total Processing Time: {self.data.total_processing_time:.2f} seconds")
        summary.append(f"  Average Time per Generation: {self.data.total_processing_time / max(len(self.data.generations), 1):.3f} seconds")
        summary.append("")

        if self.data.generations:
            final_gen = self.data.generations[-1]
            initial_fitness = self.data.generations[0].best_fitness
            summary.append(f"Fitness Evolution:")
            summary.append(f"  Initial Best Fitness: {initial_fitness:.6f}")
            summary.append(f"  Final Best Fitness: {final_gen.best_fitness:.6f}")
            summary.append(f"  Total Improvement: {initial_fitness - final_gen.best_fitness:.6f}")
            summary.append(f"  Improvement Percentage: {((initial_fitness - final_gen.best_fitness) / initial_fitness * 100):.2f}%")
            summary.append("")

        summary.append(f"Genetic Operations:")
        summary.append(f"  Beneficial Mutation Rate: {self.data.beneficial_mutation_rate:.3f}")
        summary.append(f"  Beneficial Crossover Rate: {self.data.beneficial_crossover_rate:.3f}")
        summary.append(f"  Final Convergence Rate: {self.data.final_convergence:.6f}")

        if self.data.generations:
            avg_diversity = np.mean([g.population_diversity for g in self.data.generations])
            summary.append(f"  Average Population Diversity: {avg_diversity:.3f}")

        with open(folder_path / 'summary.txt', 'w') as f:
            f.write('\n'.join(summary))

    def _save_csv_data(self, folder_path: Path):
        """Save generation data as CSV for analysis."""
        import csv

        with open(folder_path / 'generation_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # Extended CSV headers with all diversity metrics
            writer.writerow([
                'Generation', 'Best_Fitness', 'Average_Fitness', 'Worst_Fitness',
                'Fitness_Std', 'Processing_Time', 'Beneficial_Mutations',
                'Beneficial_Crossovers', 'Population_Diversity', 'Convergence_Metric',
                'Hamming_Diversity', 'Entropy_Diversity', 'Spatial_Diversity', 'Cluster_Diversity',
                'Hamming_Distance_Avg', 'Hamming_Distance_Std', 'Position_Wise_Entropy',
                'Unique_Individuals_Ratio', 'Fitness_Coefficient_Variation', 'Cluster_Count',
                'Population_Entropy', 'Normalized_Diversity', 'Local_Pattern_Entropy',
                'Spatial_Clustering', 'Edge_Pattern_Diversity', 'Quadrant_Diversity',
                'Neighbor_Similarity', 'Tile_Distribution_Variance', 'Contiguous_Regions',
                'Spatial_Autocorrelation', 'Spatial_Diversity_Score', 'Selection_Pressure',
                'Fitness_Variance', 'Current_Mutation_Rate', 'Current_Crossover_Rate'
            ])

            for gen in self.data.generations:
                writer.writerow([
                    gen.generation, gen.best_fitness, gen.average_fitness,
                    gen.worst_fitness, gen.fitness_std, gen.processing_time,
                    gen.beneficial_mutations, gen.beneficial_crossovers,
                    gen.population_diversity, gen.convergence_metric,
                    gen.hamming_diversity, gen.entropy_diversity, gen.spatial_diversity, gen.cluster_diversity,
                    gen.hamming_distance_avg, gen.hamming_distance_std, gen.position_wise_entropy,
                    gen.unique_individuals_ratio, gen.fitness_coefficient_variation, gen.cluster_count,
                    gen.population_entropy, gen.normalized_diversity, gen.local_pattern_entropy,
                    gen.spatial_clustering, gen.edge_pattern_diversity, gen.quadrant_diversity,
                    gen.neighbor_similarity, gen.tile_distribution_variance, gen.contiguous_regions,
                    gen.spatial_autocorrelation, gen.spatial_diversity_score, gen.selection_pressure,
                    gen.fitness_variance, gen.current_mutation_rate, gen.current_crossover_rate
                ])

    def _calculate_hamming_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate Hamming distance-based diversity."""
        from ..utils.diversity_metrics import calculate_hamming_distance_average
        return calculate_hamming_distance_average(population)

    def _calculate_entropy_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate entropy-based diversity."""
        from ..utils.diversity_metrics import calculate_position_wise_entropy
        return calculate_position_wise_entropy(population)

    def _calculate_spatial_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate spatial pattern diversity for image collages."""
        if len(population) < 2:
            return 0.0

        try:
            # Import spatial diversity calculator if available
            from ..genetic.spatial_diversity import SpatialDiversityManager

            # Extract grid_size from population shape and estimate num_source_images
            if population:
                height, width = population[0].shape  # (height, width)
                grid_size = (width, height)  # SpatialDiversityManager expects (width, height)
                # Estimate num_source_images from the maximum value in the population
                max_source_idx = max(np.max(individual) for individual in population)
                num_source_images = int(max_source_idx) + 1  # +1 because indices are 0-based

                spatial_manager = SpatialDiversityManager(grid_size, num_source_images)
                # Flatten 2D population arrays to 1D for spatial manager
                flattened_population = [individual.flatten() for individual in population]
                result = spatial_manager.calculate_spatial_diversity(flattened_population)
                return result.get('spatial_diversity_score', 0.0)
            else:
                return 0.0
        except ImportError:
            # Fallback to simple spatial variance
            spatial_vars = []
            for individual in population:
                # Calculate variance in spatial neighborhoods
                h, w = individual.shape
                local_vars = []
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        neighborhood = individual[i-1:i+2, j-1:j+2]
                        local_vars.append(np.var(neighborhood))

                if local_vars:
                    spatial_vars.append(np.mean(local_vars))

            return np.std(spatial_vars) if spatial_vars else 0.0

    def _calculate_cluster_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate clustering-based diversity."""
        from ..utils.diversity_metrics import calculate_cluster_diversity
        return calculate_cluster_diversity(population)

    def record_migration_event(self, generation: int, source_island: int, target_island: int,
                              migrant_fitness: float, num_migrants: int = 1):
        """
        Record a migration event for island model genetic algorithms.

        Args:
            generation: Current generation when migration occurred
            source_island: ID of the island sending migrants
            target_island: ID of the island receiving migrants
            migrant_fitness: Fitness of the migrating individual(s)
            num_migrants: Number of individuals migrating (default: 1)
        """
        migration_event = {
            'generation': generation,
            'source_island': source_island,
            'target_island': target_island,
            'migrant_fitness': migrant_fitness,
            'num_migrants': num_migrants,
            'timestamp': time.time() - self.start_time  # Time since evolution started
        }

        self.data.migration_events.append(migration_event)