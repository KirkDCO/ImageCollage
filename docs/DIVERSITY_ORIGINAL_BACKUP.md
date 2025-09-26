# Population Diversity and Premature Convergence Prevention

This document outlines comprehensive strategies for maintaining population diversity and preventing premature convergence in the Image Collage Generator's genetic algorithm. Premature convergence to local optima is one of the most critical challenges in evolutionary computation, and these techniques provide robust solutions.

## Implementation Status

**Legend:**
- ✅ **Fully Implemented** - Feature is complete and available
- 🚧 **In Progress** - Feature is partially implemented
- 📋 **Planned** - Feature is documented but not implemented
- ❌ **Not Implemented** - Feature is neither documented nor implemented

**Overall Status:** The Image Collage Generator includes a comprehensive diversity management system with advanced spatial awareness and real-time monitoring capabilities.

## Table of Contents

1. [Implementation Status Overview](#implementation-status-overview)
2. [Overview](#overview)
3. [Selection Pressure Management](#selection-pressure-management)
4. [Population Structure Strategies](#population-structure-strategies)
5. [Diversity-Preserving Operators](#diversity-preserving-operators)
6. [Explicit Diversity Maintenance](#explicit-diversity-maintenance)
7. [Restart and Injection Strategies](#restart-and-injection-strategies)
8. [Advanced Techniques](#advanced-techniques)
9. [Problem-Specific Strategies](#problem-specific-strategies-for-image-collage)
10. [Spatial Diversity Management](#spatial-diversity-management)
11. [Monitoring and Adaptive Control](#monitoring-and-adaptive-control)
12. [Configuration and Usage](#configuration-and-usage)
13. [Code Integration](#code-integration)

## Implementation Status Overview

### ✅ **Fully Implemented Features**

#### Core Diversity Metrics (`comprehensive_diversity.py`)
- **Hamming Distance Analysis**: Average and standard deviation of genetic distances
- **Position-wise Entropy**: Shannon entropy for each gene position
- **Unique Individuals Ratio**: Proportion of genetically distinct individuals
- **Fitness Diversity**: Variance, range, and coefficient of variation
- **Cluster Analysis**: Genetic clustering estimation using distance thresholds
- **Normalized Diversity Score**: Combined metric for overall population diversity

#### Spatial Diversity Management (`spatial_diversity.py`)
- **Local Pattern Entropy**: 2x2 spatial pattern analysis for tile arrangements
- **Spatial Clustering**: Analysis of similar tile neighborhoods
- **Edge Pattern Diversity**: Boundary tile arrangement analysis
- **Quadrant Diversity**: Grid section diversity comparison
- **Neighbor Similarity**: Adjacent tile correlation analysis
- **Spatial Autocorrelation**: Spatial pattern correlation metrics
- **Contiguous Regions**: Connected component analysis

#### Island Model Implementation (`island_model.py`)
- **Multi-Population Evolution**: Multiple isolated populations
- **Migration Management**: Configurable inter-island migration
- **Population Synchronization**: Coordinated evolution across islands
- **Migration Event Tracking**: Complete migration history for lineage analysis

#### Integration Points
- **Lineage Tracking Integration**: Diversity metrics included in genealogy analysis
- **Diagnostics Integration**: Real-time diversity monitoring
- **Configuration Support**: YAML and CLI configuration options
- **Checkpoint Integration**: Diversity state preserved in crash recovery

### 🚧 **Partially Implemented Features**

#### Adaptive Parameters
- **Basic Adaptation**: Population size and mutation rate adjustment
- **Diversity-Based Triggers**: Parameter changes based on diversity thresholds
- **Missing**: Advanced parameter history and predictive adjustment

### 📋 **Planned Features** (Documented but Not Implemented)

#### Selection Pressure Management
- **Fitness Scaling**: Linear scaling, sigma scaling, rank-based selection
- **Multi-Modal Selection**: Dynamic selection method combination
- **Adaptive Selection Pressure**: Real-time pressure adjustment

#### Advanced Replacement Strategies
- **Crowding Replacement**: Replace similar individuals rather than worst
- **Fitness Sharing**: Population density-based fitness adjustment
- **Speciation Support**: Species-based replacement strategies

#### Intelligent Restart Systems
- **Stagnation Detection**: Advanced convergence monitoring
- **Elite Preservation**: Selective population restart with top individuals
- **Diversity-Based Triggers**: Restart when diversity falls below thresholds

#### Real-Time Monitoring
- **Diversity Dashboard**: Live diversity monitoring during evolution
- **Alert System**: Warnings for premature convergence
- **Intervention Recommendations**: Suggested parameter adjustments

### ❌ **Future Enhancements** (Not Currently Planned)

#### Machine Learning Integration
- **Predictive Diversity Models**: ML-based diversity trend prediction
- **Automatic Parameter Tuning**: Learned parameter optimization
- **Pattern Recognition**: Automated successful pattern identification

## Overview

Population diversity is essential for:
- **Exploration vs. Exploitation Balance**: Maintaining ability to explore new solution regions
- **Local Optima Escape**: Preventing entrapment in suboptimal solutions
- **Sustained Evolution**: Keeping genetic material available for continued improvement
- **Robust Solutions**: Finding solutions that work across different problem variations

### Key Principles
- **Measure Diversity**: Use multiple metrics to assess population state
- **Adaptive Control**: Adjust algorithm behavior based on diversity levels
- **Multiple Strategies**: Combine complementary diversity preservation techniques
- **Problem-Aware**: Leverage domain knowledge for effective diversity measures

## Selection Pressure Management 📋

**Status**: Documented but not implemented - these are planned features for enhanced diversity control.

Selection pressure determines how strongly the algorithm favors high-fitness individuals. Too much pressure causes rapid convergence; too little prevents improvement.

### Fitness Scaling Techniques

#### Linear Scaling
```python
def linear_scaling(fitness_scores: List[float], scaling_factor: float = 2.0) -> List[float]:
    """Scale fitness values to control selection pressure."""
    min_fitness = min(fitness_scores)
    max_fitness = max(fitness_scores)

    if max_fitness == min_fitness:
        return [1.0] * len(fitness_scores)

    # Scale so best individual has scaling_factor times average fitness
    avg_fitness = sum(fitness_scores) / len(fitness_scores)

    scaled_scores = []
    for fitness in fitness_scores:
        scaled = avg_fitness + (fitness - avg_fitness) * scaling_factor / (max_fitness - avg_fitness)
        scaled_scores.append(max(0.1, scaled))  # Ensure minimum selection chance

    return scaled_scores
```

#### Sigma Scaling
```python
def sigma_scaling(fitness_scores: List[float], c: float = 2.0) -> List[float]:
    """Normalize fitness based on population standard deviation."""
    mean_fitness = statistics.mean(fitness_scores)
    std_fitness = statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 1.0

    if std_fitness == 0:
        return [1.0] * len(fitness_scores)

    scaled_scores = []
    for fitness in fitness_scores:
        if std_fitness > 0:
            scaled = 1.0 + (fitness - mean_fitness) / (c * std_fitness)
        else:
            scaled = 1.0
        scaled_scores.append(max(0.1, scaled))

    return scaled_scores
```

#### Rank-Based Selection
```python
def rank_based_selection_weights(population_size: int, selection_pressure: float = 1.5) -> List[float]:
    """Generate selection weights based on fitness rank rather than absolute values."""
    # Linear ranking: worst gets weight 2-s, best gets weight s
    weights = []
    for rank in range(population_size):
        weight = 2 - selection_pressure + 2 * (selection_pressure - 1) * rank / (population_size - 1)
        weights.append(weight)
    return weights
```

### Adaptive Selection Pressure
```python
class AdaptiveSelectionManager:
    def __init__(self, base_pressure: float = 2.0, diversity_threshold: float = 0.3):
        self.base_pressure = base_pressure
        self.diversity_threshold = diversity_threshold
        self.current_pressure = base_pressure

    def adjust_pressure(self, diversity_metric: float) -> float:
        """Adjust selection pressure based on population diversity."""
        if diversity_metric < self.diversity_threshold:
            # Low diversity - reduce selection pressure
            self.current_pressure = max(1.1, self.current_pressure * 0.9)
        elif diversity_metric > 2 * self.diversity_threshold:
            # High diversity - can increase selection pressure
            self.current_pressure = min(self.base_pressure, self.current_pressure * 1.05)

        return self.current_pressure
```

### Multi-Modal Selection
```python
class MultiModalSelector:
    def __init__(self, methods: List[str] = None):
        self.methods = methods or ["tournament", "roulette", "rank"]
        self.method_weights = [1.0] * len(self.methods)

    def select_parents(self, population: List[np.ndarray], fitness_scores: List[float],
                      num_parents: int) -> List[np.ndarray]:
        """Use multiple selection methods to maintain diversity."""
        parents = []

        for i in range(num_parents):
            method = random.choices(self.methods, weights=self.method_weights)[0]

            if method == "tournament":
                parent = self.tournament_selection(population, fitness_scores)
            elif method == "roulette":
                parent = self.roulette_selection(population, fitness_scores)
            elif method == "rank":
                parent = self.rank_selection(population, fitness_scores)

            parents.append(parent)

        return parents
```

## Population Structure Strategies ✅

**Status**: Fully implemented in `island_model.py` with comprehensive migration and synchronization.

### Island Model (Multiple Populations) ✅
```python
class IslandPopulationManager:
    def __init__(self, num_islands: int = 4, island_size: int = 50,
                 migration_rate: float = 0.1, migration_interval: int = 10):
        self.num_islands = num_islands
        self.islands = [Population(size=island_size) for _ in range(num_islands)]
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        self.generation = 0

    def evolve_generation(self):
        """Evolve all islands independently, then migrate."""
        # Evolve each island
        for island in self.islands:
            island.evolve_one_generation()

        # Periodic migration
        if self.generation % self.migration_interval == 0:
            self.migrate_individuals()

        self.generation += 1

    def migrate_individuals(self):
        """Exchange individuals between islands."""
        num_migrants = int(self.migration_rate * self.islands[0].size)

        for i, source_island in enumerate(self.islands):
            target_island_idx = (i + 1) % self.num_islands
            target_island = self.islands[target_island_idx]

            # Select best individuals for migration
            migrants = source_island.get_best_individuals(num_migrants)

            # Replace worst individuals in target island
            target_island.replace_worst_individuals(migrants)

    def get_global_best(self) -> np.ndarray:
        """Return best individual across all islands."""
        island_bests = [island.get_best_individual() for island in self.islands]
        island_best_fitness = [island.get_best_fitness() for island in self.islands]

        best_idx = island_best_fitness.index(min(island_best_fitness))
        return island_bests[best_idx]
```

### Spatial Population Structure
```python
class SpatialPopulation:
    def __init__(self, grid_width: int, grid_height: int):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.population_grid = [[Individual() for _ in range(grid_width)]
                               for _ in range(grid_height)]

    def get_neighbors(self, x: int, y: int, radius: int = 1) -> List[Individual]:
        """Get spatial neighbors for local competition."""
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = (x + dx) % self.grid_width, (y + dy) % self.grid_height
                if (dx, dy) != (0, 0):
                    neighbors.append(self.population_grid[ny][nx])
        return neighbors

    def evolve_spatially(self):
        """Evolve using only local neighborhoods."""
        new_grid = [[None for _ in range(self.grid_width)]
                   for _ in range(self.grid_height)]

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                current = self.population_grid[y][x]
                neighbors = self.get_neighbors(x, y)

                # Local tournament selection
                parents = self.local_tournament_selection(neighbors, 2)
                offspring = self.crossover(parents[0], parents[1])

                # Replace if offspring is better than current
                if offspring.fitness < current.fitness:
                    new_grid[y][x] = offspring
                else:
                    new_grid[y][x] = current

        self.population_grid = new_grid
```

## Diversity-Preserving Operators

### Smart Mutation Strategies

#### Adaptive Mutation Rates
```python
class AdaptiveMutationManager:
    def __init__(self, base_rate: float = 0.05, min_rate: float = 0.01, max_rate: float = 0.3):
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.current_rate = base_rate

    def update_mutation_rate(self, diversity_metrics: Dict[str, float]) -> float:
        """Adjust mutation rate based on population diversity."""
        diversity_score = diversity_metrics.get('normalized_diversity', 0.5)

        if diversity_score < 0.2:
            # Very low diversity - increase mutation significantly
            self.current_rate = min(self.max_rate, self.current_rate * 1.5)
        elif diversity_score < 0.4:
            # Low diversity - increase mutation moderately
            self.current_rate = min(self.max_rate, self.current_rate * 1.2)
        elif diversity_score > 0.8:
            # High diversity - can reduce mutation
            self.current_rate = max(self.min_rate, self.current_rate * 0.9)

        return self.current_rate
```

#### Distance-Based Mutation
```python
def distance_based_mutation(individual: np.ndarray, population: List[np.ndarray],
                           grid_size: Tuple[int, int]) -> np.ndarray:
    """Mutate toward unexplored regions of solution space."""
    # Calculate average distance to all other individuals
    avg_distances = []
    for pos in range(len(individual)):
        distances = []
        for other in population:
            if not np.array_equal(individual, other):
                distances.append(abs(individual[pos] - other[pos]))
        avg_distances.append(np.mean(distances) if distances else 0)

    # Bias mutation toward positions with high average distance
    # (indicating this individual is unique in those positions)
    mutation_weights = [1.0 / (1.0 + dist) for dist in avg_distances]
    mutation_weights = np.array(mutation_weights)
    mutation_weights /= mutation_weights.sum()

    # Select positions to mutate based on weights
    num_mutations = max(1, int(len(individual) * 0.05))
    mutation_positions = np.random.choice(
        len(individual), size=num_mutations, replace=False, p=mutation_weights
    )

    mutated = individual.copy()
    for pos in mutation_positions:
        # Mutate to a value different from neighbors
        current_values = [ind[pos] for ind in population]
        available_values = list(set(range(len(population))) - set(current_values))
        if available_values:
            mutated[pos] = random.choice(available_values)
        else:
            mutated[pos] = random.randint(0, len(population) - 1)

    return mutated
```

#### Multi-Point Guided Mutation
```python
def guided_multi_point_mutation(individual: np.ndarray, population: List[np.ndarray],
                               mutation_rate: float = 0.1) -> np.ndarray:
    """Perform multiple mutations guided by population patterns."""
    mutated = individual.copy()

    # Identify common patterns in population
    position_frequencies = {}
    for pos in range(len(individual)):
        frequencies = {}
        for other in population:
            value = other[pos]
            frequencies[value] = frequencies.get(value, 0) + 1
        position_frequencies[pos] = frequencies

    # Mutate multiple positions
    for pos in range(len(individual)):
        if random.random() < mutation_rate:
            current_value = individual[pos]
            freq_dist = position_frequencies[pos]

            # Bias away from common values
            total_pop = len(population)
            common_threshold = total_pop * 0.3

            if freq_dist.get(current_value, 0) > common_threshold:
                # Current value is too common, change it
                rare_values = [v for v, f in freq_dist.items() if f < common_threshold]
                if rare_values:
                    mutated[pos] = random.choice(rare_values)
                else:
                    # All values are common, choose randomly
                    available = list(set(range(len(population))) - {current_value})
                    if available:
                        mutated[pos] = random.choice(available)

    return mutated
```

### Diversity-Aware Crossover

#### Dissimilar Parent Selection
```python
def select_dissimilar_parents(population: List[np.ndarray], fitness_scores: List[float],
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
    for _ in range(min(100, len(population) * (len(population) - 1) // 2)):
        idx1, idx2 = random.sample(range(len(population)), 2)
        score = combined_score(idx1, idx2)

        if score < best_score:
            best_score = score
            best_parents = (population[idx1], population[idx2])

    return best_parents
```

#### Multi-Parent Crossover
```python
def multi_parent_crossover(parents: List[np.ndarray], num_offspring: int = 2) -> List[np.ndarray]:
    """Create offspring by combining genetic material from multiple parents."""
    if len(parents) < 3:
        # Fall back to standard two-parent crossover
        return standard_crossover(parents[0], parents[1])

    offspring = []
    individual_length = len(parents[0])

    for _ in range(num_offspring):
        child = np.zeros_like(parents[0])

        for pos in range(individual_length):
            # For each position, select value from one of the parents
            # Bias toward less common values among parents
            parent_values = [parent[pos] for parent in parents]
            value_counts = {v: parent_values.count(v) for v in set(parent_values)}

            # Weight inversely to frequency (prefer rare values)
            weights = [1.0 / value_counts[v] for v in parent_values]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            # Select parent based on weights
            selected_parent_idx = np.random.choice(len(parents), p=weights)
            child[pos] = parents[selected_parent_idx][pos]

        offspring.append(child)

    return offspring
```

## Explicit Diversity Maintenance

### Comprehensive Diversity Metrics
```python
class DiversityMetricsCalculator:
    def __init__(self):
        self.metrics_history = []

    def calculate_comprehensive_diversity(self, population: List[np.ndarray],
                                        fitness_scores: List[float]) -> Dict[str, float]:
        """Calculate multiple diversity measures for comprehensive assessment."""
        metrics = {}

        # Genetic diversity measures
        metrics['hamming_distance_avg'] = self._average_hamming_distance(population)
        metrics['hamming_distance_std'] = self._hamming_distance_std(population)
        metrics['position_wise_entropy'] = self._position_wise_entropy(population)
        metrics['unique_individuals_ratio'] = self._unique_individuals_ratio(population)

        # Fitness diversity measures
        metrics['fitness_variance'] = np.var(fitness_scores)
        metrics['fitness_range'] = max(fitness_scores) - min(fitness_scores)
        metrics['fitness_coefficient_variation'] = np.std(fitness_scores) / np.mean(fitness_scores) if np.mean(fitness_scores) > 0 else 0

        # Structural diversity measures
        metrics['cluster_count'] = self._estimate_genetic_clusters(population)
        metrics['population_entropy'] = self._population_entropy(population)

        # Normalized combined diversity score
        metrics['normalized_diversity'] = self._normalize_diversity_score(metrics)

        self.metrics_history.append(metrics)
        return metrics

    def _average_hamming_distance(self, population: List[np.ndarray]) -> float:
        """Calculate average pairwise Hamming distance."""
        if len(population) < 2:
            return 0.0

        total_distance = 0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.sum(population[i] != population[j])
                total_distance += distance
                comparisons += 1

        return total_distance / comparisons if comparisons > 0 else 0

    def _position_wise_entropy(self, population: List[np.ndarray]) -> float:
        """Calculate entropy for each position and return average."""
        if not population:
            return 0.0

        individual_length = len(population[0])
        total_entropy = 0

        for pos in range(individual_length):
            values = [individual[pos] for individual in population]
            value_counts = {}
            for value in values:
                value_counts[value] = value_counts.get(value, 0) + 1

            # Calculate Shannon entropy
            entropy = 0
            total_count = len(values)
            for count in value_counts.values():
                if count > 0:
                    probability = count / total_count
                    entropy -= probability * math.log2(probability)

            total_entropy += entropy

        return total_entropy / individual_length

    def _estimate_genetic_clusters(self, population: List[np.ndarray],
                                 max_clusters: int = 10) -> int:
        """Estimate number of genetic clusters using elbow method."""
        if len(population) < 2:
            return 1

        # Calculate distance matrix
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.sum(population[i] != population[j])
                distances.append(dist)

        if not distances:
            return 1

        # Use simple threshold-based clustering
        avg_distance = np.mean(distances)
        threshold = avg_distance * 0.5

        # Count clusters by connected components
        clusters = []
        assigned = [False] * len(population)

        for i in range(len(population)):
            if not assigned[i]:
                cluster = [i]
                assigned[i] = True

                # Find all individuals within threshold distance
                for j in range(len(population)):
                    if not assigned[j]:
                        dist = np.sum(population[i] != population[j])
                        if dist <= threshold:
                            cluster.append(j)
                            assigned[j] = True

                clusters.append(cluster)

        return len(clusters)

    def _normalize_diversity_score(self, metrics: Dict[str, float]) -> float:
        """Combine multiple metrics into normalized diversity score (0-1)."""
        # Weight different metrics based on importance
        weights = {
            'hamming_distance_avg': 0.25,
            'position_wise_entropy': 0.20,
            'unique_individuals_ratio': 0.20,
            'fitness_coefficient_variation': 0.15,
            'cluster_count': 0.10,
            'population_entropy': 0.10
        }

        # Normalize each metric to 0-1 range (problem-specific)
        normalized = {}
        normalized['hamming_distance_avg'] = min(1.0, metrics['hamming_distance_avg'] / len(population[0]))
        normalized['position_wise_entropy'] = metrics['position_wise_entropy'] / math.log2(len(population)) if len(population) > 1 else 0
        normalized['unique_individuals_ratio'] = metrics['unique_individuals_ratio']
        normalized['fitness_coefficient_variation'] = min(1.0, metrics['fitness_coefficient_variation'])
        normalized['cluster_count'] = min(1.0, metrics['cluster_count'] / (len(population) * 0.1))

        # Calculate weighted average
        total_score = sum(weights[key] * normalized[key] for key in weights.keys())
        return total_score
```

### Diversity-Based Replacement Strategies

#### Crowding Replacement
```python
class CrowdingReplacement:
    def __init__(self, crowding_factor: float = 2.0):
        self.crowding_factor = crowding_factor

    def replace_individual(self, population: List[np.ndarray], fitness_scores: List[float],
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
            # Fall back to worst replacement if not better than similar
            return fitness_scores.index(max(fitness_scores))
```

#### Fitness Sharing
```python
class FitnessSharing:
    def __init__(self, sharing_radius: float = 5.0, alpha: float = 1.0):
        self.sharing_radius = sharing_radius
        self.alpha = alpha

    def calculate_shared_fitness(self, population: List[np.ndarray],
                               fitness_scores: List[float]) -> List[float]:
        """Adjust fitness based on local population density."""
        shared_fitness = []

        for i, individual in enumerate(population):
            # Calculate sharing function values
            sharing_sum = 0
            for j, other in enumerate(population):
                distance = np.sum(individual != other)
                if distance < self.sharing_radius:
                    sharing_value = 1 - (distance / self.sharing_radius) ** self.alpha
                    sharing_sum += sharing_value

            # Adjust fitness by sharing sum
            if sharing_sum > 0:
                shared = fitness_scores[i] / sharing_sum
            else:
                shared = fitness_scores[i]

            shared_fitness.append(shared)

        return shared_fitness
```

## Restart and Injection Strategies

### Intelligent Restart Mechanisms
```python
class IntelligentRestartManager:
    def __init__(self, diversity_threshold: float = 0.1, stagnation_threshold: int = 50,
                 elite_preservation_ratio: float = 0.1):
        self.diversity_threshold = diversity_threshold
        self.stagnation_threshold = stagnation_threshold
        self.elite_preservation_ratio = elite_preservation_ratio
        self.stagnation_counter = 0
        self.best_fitness_history = []

    def should_restart(self, diversity_score: float, current_best_fitness: float) -> bool:
        """Determine if population should be restarted."""
        self.best_fitness_history.append(current_best_fitness)

        # Check diversity-based restart
        if diversity_score < self.diversity_threshold:
            return True

        # Check stagnation-based restart
        if len(self.best_fitness_history) >= self.stagnation_threshold:
            recent_improvement = (
                self.best_fitness_history[-self.stagnation_threshold] -
                self.best_fitness_history[-1]
            )
            if recent_improvement < 0.001:  # Minimal improvement threshold
                return True

        return False

    def perform_restart(self, population: List[np.ndarray], fitness_scores: List[float],
                       population_size: int) -> Tuple[List[np.ndarray], List[float]]:
        """Restart population while preserving elite individuals."""
        # Sort by fitness and preserve elite
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        elite_count = max(1, int(population_size * self.elite_preservation_ratio))

        new_population = []
        new_fitness_scores = []

        # Keep elite individuals
        for i in range(elite_count):
            idx = sorted_indices[i]
            new_population.append(population[idx].copy())
            new_fitness_scores.append(fitness_scores[idx])

        # Fill rest with random individuals
        for _ in range(population_size - elite_count):
            random_individual = self._generate_random_individual(len(population[0]))
            new_population.append(random_individual)
            new_fitness_scores.append(float('inf'))  # Will be evaluated later

        self.stagnation_counter = 0
        self.best_fitness_history = self.best_fitness_history[-10:]  # Keep recent history

        return new_population, new_fitness_scores
```

### Adaptive Immigration Policies
```python
class AdaptiveImmigrationManager:
    def __init__(self, base_immigration_rate: float = 0.05,
                 max_immigration_rate: float = 0.2):
        self.base_immigration_rate = base_immigration_rate
        self.max_immigration_rate = max_immigration_rate
        self.current_rate = base_immigration_rate

    def update_immigration_rate(self, diversity_metrics: Dict[str, float]) -> float:
        """Adjust immigration rate based on population state."""
        diversity_score = diversity_metrics.get('normalized_diversity', 0.5)

        if diversity_score < 0.2:
            # Very low diversity - increase immigration
            self.current_rate = min(self.max_immigration_rate, self.current_rate * 1.5)
        elif diversity_score > 0.7:
            # High diversity - reduce immigration
            self.current_rate = max(self.base_immigration_rate, self.current_rate * 0.8)

        return self.current_rate

    def perform_immigration(self, population: List[np.ndarray], fitness_scores: List[float],
                           diversity_metrics: Dict[str, float]) -> Tuple[List[np.ndarray], List[float]]:
        """Inject new individuals to increase diversity."""
        immigration_count = int(len(population) * self.current_rate)

        if immigration_count == 0:
            return population, fitness_scores

        # Identify underexplored regions
        immigrants = self._generate_diverse_immigrants(population, immigration_count)

        # Replace worst individuals
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)

        new_population = population.copy()
        new_fitness_scores = fitness_scores.copy()

        for i in range(immigration_count):
            replace_idx = sorted_indices[i]
            new_population[replace_idx] = immigrants[i]
            new_fitness_scores[replace_idx] = float('inf')  # Will be evaluated later

        return new_population, new_fitness_scores

    def _generate_diverse_immigrants(self, population: List[np.ndarray],
                                   count: int) -> List[np.ndarray]:
        """Generate immigrants that explore underrepresented regions."""
        immigrants = []
        individual_length = len(population[0])

        # Analyze position frequencies
        position_frequencies = {}
        for pos in range(individual_length):
            frequencies = {}
            for individual in population:
                value = individual[pos]
                frequencies[value] = frequencies.get(value, 0) + 1
            position_frequencies[pos] = frequencies

        for _ in range(count):
            immigrant = np.zeros(individual_length, dtype=int)

            for pos in range(individual_length):
                frequencies = position_frequencies[pos]
                total_pop = len(population)

                # Bias toward underrepresented values
                available_values = list(range(len(population)))  # Assuming tile indices
                weights = []

                for value in available_values:
                    current_freq = frequencies.get(value, 0)
                    # Higher weight for less frequent values
                    weight = max(0.1, 1.0 - (current_freq / total_pop))
                    weights.append(weight)

                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                # Select value based on weights
                immigrant[pos] = np.random.choice(available_values, p=weights)

            immigrants.append(immigrant)

        return immigrants
```

## Advanced Techniques

### Speciation and Niching
```python
class SpeciationManager:
    def __init__(self, compatibility_threshold: float = 3.0,
                 min_species_size: int = 3):
        self.compatibility_threshold = compatibility_threshold
        self.min_species_size = min_species_size
        self.species_representatives = []
        self.species_history = []

    def organize_species(self, population: List[np.ndarray],
                        fitness_scores: List[float]) -> List[List[int]]:
        """Group population into species based on genetic similarity."""
        species = []
        assigned = [False] * len(population)

        # Update species representatives
        self._update_species_representatives(population)

        # Assign individuals to species
        for i, individual in enumerate(population):
            if assigned[i]:
                continue

            # Try to assign to existing species
            assigned_to_species = False
            for species_idx, representative in enumerate(self.species_representatives):
                if self._genetic_distance(individual, representative) < self.compatibility_threshold:
                    # Create or extend species
                    if species_idx >= len(species):
                        species.extend([[] for _ in range(species_idx - len(species) + 1)])
                    species[species_idx].append(i)
                    assigned[i] = True
                    assigned_to_species = True
                    break

            # Create new species if not assigned
            if not assigned_to_species:
                species.append([i])
                assigned[i] = True
                self.species_representatives.append(individual.copy())

        # Filter out small species
        species = [s for s in species if len(s) >= self.min_species_size]

        return species

    def _genetic_distance(self, individual1: np.ndarray, individual2: np.ndarray) -> float:
        """Calculate genetic distance between two individuals."""
        hamming_distance = np.sum(individual1 != individual2)
        return hamming_distance / len(individual1)

    def apply_fitness_sharing_within_species(self, species: List[List[int]],
                                           fitness_scores: List[float]) -> List[float]:
        """Apply fitness sharing within each species."""
        shared_fitness = fitness_scores.copy()

        for species_members in species:
            species_size = len(species_members)
            for member_idx in species_members:
                shared_fitness[member_idx] = fitness_scores[member_idx] / species_size

        return shared_fitness
```

### Multi-Objective Diversity Maintenance
```python
class MultiObjectiveDiversityManager:
    def __init__(self, diversity_weight: float = 0.3):
        self.diversity_weight = diversity_weight

    def calculate_multi_objective_fitness(self, individual: np.ndarray,
                                        population: List[np.ndarray],
                                        original_fitness: float) -> Tuple[float, float]:
        """Calculate fitness with diversity as second objective."""
        # Original fitness (minimize)
        fitness_objective = original_fitness

        # Diversity objective (maximize, so negate for minimization)
        diversity_objective = -self._calculate_individual_diversity(individual, population)

        return fitness_objective, diversity_objective

    def _calculate_individual_diversity(self, individual: np.ndarray,
                                      population: List[np.ndarray]) -> float:
        """Calculate how diverse an individual is relative to population."""
        if len(population) <= 1:
            return 1.0

        total_distance = 0
        for other in population:
            if not np.array_equal(individual, other):
                distance = np.sum(individual != other)
                total_distance += distance

        avg_distance = total_distance / (len(population) - 1)
        max_possible_distance = len(individual)

        return avg_distance / max_possible_distance

    def pareto_selection(self, population: List[np.ndarray],
                        fitness_objectives: List[Tuple[float, float]],
                        selection_size: int) -> List[int]:
        """Select individuals based on Pareto dominance with crowding distance."""
        # Identify Pareto fronts
        fronts = self._fast_non_dominated_sort(fitness_objectives)

        selected_indices = []
        front_idx = 0

        while len(selected_indices) < selection_size and front_idx < len(fronts):
            front = fronts[front_idx]

            if len(selected_indices) + len(front) <= selection_size:
                # Add entire front
                selected_indices.extend(front)
            else:
                # Add part of front based on crowding distance
                remaining_slots = selection_size - len(selected_indices)
                crowding_distances = self._calculate_crowding_distance(front, fitness_objectives)

                # Sort by crowding distance (descending)
                front_with_distances = list(zip(front, crowding_distances))
                front_with_distances.sort(key=lambda x: x[1], reverse=True)

                selected_indices.extend([idx for idx, _ in front_with_distances[:remaining_slots]])

            front_idx += 1

        return selected_indices
```

## Problem-Specific Strategies for Image Collage

### Spatial Diversity Preservation
```python
class SpatialDiversityManager:
    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_width, self.grid_height = grid_size
        self.total_positions = self.grid_width * self.grid_height

    def calculate_spatial_diversity(self, population: List[np.ndarray]) -> Dict[str, float]:
        """Calculate diversity metrics specific to spatial tile arrangements."""
        metrics = {}

        # Local pattern diversity
        metrics['local_pattern_entropy'] = self._calculate_local_pattern_entropy(population)

        # Spatial clustering diversity
        metrics['spatial_clustering'] = self._calculate_spatial_clustering(population)

        # Edge pattern diversity
        metrics['edge_pattern_diversity'] = self._calculate_edge_pattern_diversity(population)

        # Quadrant diversity
        metrics['quadrant_diversity'] = self._calculate_quadrant_diversity(population)

        return metrics

    def _calculate_local_pattern_entropy(self, population: List[np.ndarray]) -> float:
        """Calculate entropy of local 2x2 patterns."""
        pattern_counts = {}

        for individual in population:
            grid = individual.reshape(self.grid_height, self.grid_width)

            # Extract all 2x2 patterns
            for y in range(self.grid_height - 1):
                for x in range(self.grid_width - 1):
                    pattern = tuple(grid[y:y+2, x:x+2].flatten())
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Calculate Shannon entropy
        total_patterns = sum(pattern_counts.values())
        entropy = 0
        for count in pattern_counts.values():
            if count > 0:
                prob = count / total_patterns
                entropy -= prob * math.log2(prob)

        return entropy

    def _calculate_spatial_clustering(self, population: List[np.ndarray]) -> float:
        """Calculate diversity in spatial clustering patterns."""
        clustering_scores = []

        for individual in population:
            grid = individual.reshape(self.grid_height, self.grid_width)

            # Calculate clustering using local similarity
            similarity_sum = 0
            comparisons = 0

            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    # Check neighbors
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                                (dy != 0 or dx != 0)):
                                neighbors.append(grid[ny, nx])

                    # Calculate similarity to neighbors
                    if neighbors:
                        current_tile = grid[y, x]
                        similar_neighbors = sum(1 for n in neighbors if n == current_tile)
                        similarity = similar_neighbors / len(neighbors)
                        similarity_sum += similarity
                        comparisons += 1

            clustering_score = similarity_sum / comparisons if comparisons > 0 else 0
            clustering_scores.append(clustering_score)

        # Return variance in clustering scores (higher = more diverse)
        return np.var(clustering_scores)

    def spatial_aware_mutation(self, individual: np.ndarray,
                              population: List[np.ndarray]) -> np.ndarray:
        """Perform mutation that considers spatial diversity."""
        mutated = individual.copy()
        grid = individual.reshape(self.grid_height, self.grid_width)

        # Identify regions with low spatial diversity
        diversity_map = self._calculate_position_diversity(individual, population)

        # Bias mutation toward low-diversity regions
        flat_diversity = diversity_map.flatten()
        mutation_weights = 1.0 / (flat_diversity + 0.1)  # Avoid division by zero
        mutation_weights /= mutation_weights.sum()

        # Select positions for mutation
        num_mutations = max(1, int(len(individual) * 0.05))
        mutation_positions = np.random.choice(
            len(individual), size=num_mutations, replace=False, p=mutation_weights
        )

        for pos in mutation_positions:
            y, x = pos // self.grid_width, pos % self.grid_width

            # Find tiles that would increase local diversity
            current_neighbors = self._get_neighbor_tiles(grid, x, y)

            # Choose tile different from neighbors
            available_tiles = list(range(len(population)))
            diverse_tiles = [t for t in available_tiles if t not in current_neighbors]

            if diverse_tiles:
                mutated[pos] = random.choice(diverse_tiles)
            else:
                mutated[pos] = random.choice(available_tiles)

        return mutated
```

### Multi-Modal Fitness Landscape Management
```python
class MultiModalFitnessManager:
    def __init__(self, fitness_components: List[str] = None):
        self.fitness_components = fitness_components or ['color', 'luminance', 'texture', 'edges']
        self.component_diversity_history = {comp: [] for comp in self.fitness_components}

    def maintain_component_diversity(self, population: List[np.ndarray],
                                   component_fitness_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """Maintain diversity in each fitness component separately."""
        diversity_adjustments = {}

        for component in self.fitness_components:
            scores = component_fitness_scores[component]

            # Calculate component-specific diversity
            component_diversity = np.var(scores) / (np.mean(scores) ** 2) if np.mean(scores) > 0 else 0
            self.component_diversity_history[component].append(component_diversity)

            # Determine if this component needs diversity boost
            recent_diversity = self.component_diversity_history[component][-10:]
            if len(recent_diversity) >= 3:
                diversity_trend = np.polyfit(range(len(recent_diversity)), recent_diversity, 1)[0]

                if diversity_trend < -0.01:  # Declining diversity
                    diversity_adjustments[component] = 'increase'
                elif component_diversity < 0.1:  # Low absolute diversity
                    diversity_adjustments[component] = 'increase'
                else:
                    diversity_adjustments[component] = 'maintain'
            else:
                diversity_adjustments[component] = 'maintain'

        return diversity_adjustments

    def component_aware_selection(self, population: List[np.ndarray],
                                 component_fitness_scores: Dict[str, List[float]],
                                 diversity_adjustments: Dict[str, str],
                                 selection_size: int) -> List[int]:
        """Select individuals considering diversity in each fitness component."""

        # Calculate selection scores for each individual
        selection_scores = []

        for i in range(len(population)):
            score = 0

            for component in self.fitness_components:
                component_score = component_fitness_scores[component][i]
                adjustment = diversity_adjustments[component]

                if adjustment == 'increase':
                    # Favor individuals that are diverse in this component
                    other_scores = [component_fitness_scores[component][j]
                                  for j in range(len(population)) if j != i]

                    if other_scores:
                        # Diversity bonus based on distance from others
                        avg_other = np.mean(other_scores)
                        diversity_bonus = abs(component_score - avg_other)
                        score += component_score - 0.5 * diversity_bonus  # Assuming minimization
                    else:
                        score += component_score
                else:
                    score += component_score

            selection_scores.append(score)

        # Select best individuals based on adjusted scores
        selected_indices = sorted(range(len(selection_scores)),
                                key=lambda i: selection_scores[i])[:selection_size]

        return selected_indices
```

## Spatial Diversity Management ✅

**Status**: Fully implemented in `spatial_diversity.py` with comprehensive spatial awareness for image collage generation.

The spatial diversity system understands the unique characteristics of tile-based image generation and implements diversity preservation techniques that account for spatial relationships between tiles.

### Core Spatial Metrics ✅

```python
class SpatialDiversityManager:
    """Implemented spatial diversity analysis for image collage generation."""

    def calculate_spatial_diversity(self, population: List[np.ndarray]) -> Dict[str, float]:
        """Calculate comprehensive spatial diversity metrics."""
        return {
            'local_pattern_entropy': self._calculate_local_pattern_entropy(population),
            'spatial_clustering': self._calculate_spatial_clustering(population),
            'edge_pattern_diversity': self._calculate_edge_pattern_diversity(population),
            'quadrant_diversity': self._calculate_quadrant_diversity(population),
            'neighbor_similarity': self._calculate_neighbor_similarity(population),
            'position_wise_entropy': self._calculate_position_wise_entropy(population),
            'tile_distribution_variance': self._calculate_tile_distribution_variance(population),
            'contiguous_regions': self._calculate_contiguous_regions(population),
            'spatial_autocorrelation': self._calculate_spatial_autocorrelation(population),
            'spatial_diversity_score': self._calculate_combined_spatial_score(metrics)
        }
```

### Spatial Analysis Features ✅

#### Local Pattern Analysis
- **2x2 Pattern Entropy**: Analyzes diversity in small local tile arrangements
- **Edge Transition Patterns**: Examines tile transitions at grid boundaries
- **Corner Pattern Analysis**: Special handling for grid corner positions

#### Neighborhood Analysis
- **8-Connected Neighbors**: Analyzes all adjacent tiles for similarity patterns
- **Distance-Weighted Similarity**: Considers tile similarity at different distances
- **Clustering Coefficients**: Measures local clustering tendencies

#### Grid Structure Analysis
- **Quadrant Comparison**: Analyzes diversity differences across grid sections
- **Spatial Autocorrelation**: Measures spatial correlation patterns
- **Contiguous Region Detection**: Identifies connected areas of similar tiles

### Integration with Core System ✅

#### Configuration Support
```yaml
# YAML configuration for spatial diversity
genetic_algorithm:
  enable_comprehensive_diversity: true
  enable_spatial_diversity: true
  spatial_diversity_weight: 0.3
```

#### Diagnostics Integration
- **Spatial diversity metrics included in diagnostics reports**
- **Visualization of spatial patterns and diversity trends**
- **Integration with lineage tracking for spatial inheritance analysis**

## Monitoring and Adaptive Control

### Real-Time Diversity Dashboard
```python
class DiversityDashboard:
    def __init__(self, update_interval: int = 10):
        self.update_interval = update_interval
        self.diversity_history = []
        self.intervention_history = []
        self.alert_thresholds = {
            'critical_diversity': 0.1,
            'low_diversity': 0.2,
            'stagnation_generations': 30
        }

    def update_dashboard(self, generation: int, diversity_metrics: Dict[str, float],
                        population_state: Dict[str, Any]):
        """Update diversity monitoring dashboard."""

        # Record metrics
        self.diversity_history.append({
            'generation': generation,
            'metrics': diversity_metrics.copy(),
            'population_state': population_state.copy()
        })

        # Check for alerts
        alerts = self._check_diversity_alerts(diversity_metrics, generation)

        # Display dashboard (if in verbose mode)
        if generation % self.update_interval == 0:
            self._display_dashboard(generation, diversity_metrics, alerts)

        return alerts

    def _check_diversity_alerts(self, metrics: Dict[str, float],
                               generation: int) -> List[str]:
        """Check for diversity-related issues requiring intervention."""
        alerts = []

        diversity_score = metrics.get('normalized_diversity', 0.5)

        if diversity_score < self.alert_thresholds['critical_diversity']:
            alerts.append('CRITICAL: Extremely low population diversity')
        elif diversity_score < self.alert_thresholds['low_diversity']:
            alerts.append('WARNING: Low population diversity')

        # Check for stagnation
        if len(self.diversity_history) >= self.alert_thresholds['stagnation_generations']:
            recent_diversity = [h['metrics']['normalized_diversity']
                              for h in self.diversity_history[-self.alert_thresholds['stagnation_generations']:]]

            diversity_change = max(recent_diversity) - min(recent_diversity)
            if diversity_change < 0.05:
                alerts.append('WARNING: Diversity stagnation detected')

        return alerts

    def _display_dashboard(self, generation: int, metrics: Dict[str, float],
                          alerts: List[str]):
        """Display diversity dashboard information."""
        print(f"\n=== Diversity Dashboard - Generation {generation} ===")
        print(f"Overall Diversity Score: {metrics.get('normalized_diversity', 0):.3f}")
        print(f"Hamming Distance (avg): {metrics.get('hamming_distance_avg', 0):.2f}")
        print(f"Position Entropy: {metrics.get('position_wise_entropy', 0):.3f}")
        print(f"Unique Individuals: {metrics.get('unique_individuals_ratio', 0):.2%}")
        print(f"Genetic Clusters: {metrics.get('cluster_count', 0)}")

        if alerts:
            print("\nALERTS:")
            for alert in alerts:
                print(f"  ⚠ {alert}")

        print("=" * 50)
```

### Adaptive Parameter Controller
```python
class AdaptiveParameterController:
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config.copy()
        self.current_config = base_config.copy()
        self.adaptation_history = []
        self.parameter_bounds = {
            'mutation_rate': (0.01, 0.5),
            'crossover_rate': (0.3, 0.95),
            'tournament_size': (2, 10),
            'elitism_rate': (0.05, 0.3)
        }

    def adapt_parameters(self, diversity_metrics: Dict[str, float],
                        performance_metrics: Dict[str, float],
                        generation: int) -> Dict[str, Any]:
        """Dynamically adjust GA parameters based on population state."""

        diversity_score = diversity_metrics.get('normalized_diversity', 0.5)
        fitness_improvement = performance_metrics.get('fitness_improvement_rate', 0.0)

        adaptations = {}

        # Mutation rate adaptation
        if diversity_score < 0.2:
            # Low diversity - increase mutation
            new_rate = min(self.parameter_bounds['mutation_rate'][1],
                          self.current_config['mutation_rate'] * 1.3)
            adaptations['mutation_rate'] = new_rate
        elif diversity_score > 0.7 and fitness_improvement > 0:
            # High diversity with good progress - can reduce mutation
            new_rate = max(self.parameter_bounds['mutation_rate'][0],
                          self.current_config['mutation_rate'] * 0.9)
            adaptations['mutation_rate'] = new_rate

        # Selection pressure adaptation
        if diversity_score < 0.3:
            # Reduce selection pressure
            new_size = max(self.parameter_bounds['tournament_size'][0],
                          self.current_config['tournament_size'] - 1)
            adaptations['tournament_size'] = new_size
        elif diversity_score > 0.6 and fitness_improvement < 0.001:
            # Can increase selection pressure for better convergence
            new_size = min(self.parameter_bounds['tournament_size'][1],
                          self.current_config['tournament_size'] + 1)
            adaptations['tournament_size'] = new_size

        # Update configuration
        for param, value in adaptations.items():
            self.current_config[param] = value

        # Record adaptation
        self.adaptation_history.append({
            'generation': generation,
            'adaptations': adaptations.copy(),
            'diversity_score': diversity_score,
            'fitness_improvement': fitness_improvement
        })

        return adaptations
```

## Configuration and Usage ✅

**Status**: Fully integrated with YAML configuration and CLI options.

### YAML Configuration

```yaml
# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  # Core diversity settings
  enable_comprehensive_diversity: true
  enable_spatial_diversity: true

  # Island model for population diversity
  enable_island_model: true
  island_model_num_islands: 4
  island_model_migration_interval: 10
  island_model_migration_rate: 0.2

  # Advanced evolution features
  enable_adaptive_parameters: true
  enable_advanced_crossover: true
  enable_advanced_mutation: true

  # Diversity thresholds
  diversity_threshold: 0.5
  stagnation_threshold: 25
  restart_threshold: 20
  restart_ratio: 0.4
```

### CLI Usage

```bash
# Enable comprehensive diversity tracking
image-collage generate target.jpg sources/ result.png \
  --preset advanced --diagnostics diagnostics/ --verbose

# With island model and migration
image-collage generate target.jpg sources/ result.png \
  --config island_config.yaml --track-lineage lineage/

# Monitor diversity in real-time
image-collage generate target.jpg sources/ result.png \
  --preset gpu --save-checkpoints --verbose
```

### Integration with Other Features

#### Lineage Tracking Integration
- Diversity metrics automatically included in lineage analysis
- Spatial diversity inheritance tracking
- Migration event diversity impact analysis

#### Diagnostics Integration
- Real-time diversity monitoring in diagnostics dashboard
- Diversity trend visualization
- Spatial pattern evolution tracking

#### Checkpoint Integration
- Diversity state preserved across crash recovery
- Population diversity restoration on resume
- Adaptive parameter state preservation

## Implementation Plan

### Phase 1: Core Diversity Infrastructure (Weeks 1-2)
1. **Diversity Metrics Calculator**
   - Implement comprehensive diversity measurement
   - Add to existing diagnostics system
   - Create baseline monitoring

2. **Adaptive Mutation Manager**
   - Basic mutation rate adaptation
   - Integration with GA engine
   - Simple diversity threshold responses

3. **Basic Immigration Policy**
   - Random individual injection
   - Worst individual replacement
   - Configurable immigration rates

### Phase 2: Advanced Selection and Replacement (Weeks 3-4)
1. **Multi-Modal Selection**
   - Tournament, roulette, rank combination
   - Adaptive selection pressure
   - Fitness scaling implementations

2. **Diversity-Based Replacement**
   - Crowding replacement strategy
   - Fitness sharing implementation
   - Similar individual identification

3. **Restart Mechanisms**
   - Intelligent restart triggers
   - Elite preservation during restart
   - Stagnation detection

### Phase 3: Spatial and Problem-Specific Features (Weeks 5-6)
1. **Spatial Diversity Management**
   - Position-specific diversity metrics
   - Spatial pattern analysis
   - Context-aware mutation

2. **Multi-Component Fitness Diversity**
   - Component-wise diversity tracking
   - Balanced evolution across fitness aspects
   - Component-aware selection

### Phase 4: Advanced Techniques (Weeks 7-8)
1. **Speciation and Niching**
   - Species identification algorithms
   - Within-species fitness sharing
   - Species population management

2. **Island Model Implementation**
   - Multiple population management
   - Migration policies
   - Inter-island communication

3. **Adaptive Parameter Control**
   - Real-time parameter adjustment
   - Performance-based adaptation
   - Parameter history tracking

## Code Integration

### Main GA Engine Integration
```python
# genetic/ga_engine.py
class GeneticAlgorithmEngine:
    def __init__(self, config: CollageConfig, diversity_manager: Optional[DiversityManager] = None):
        self.config = config
        self.diversity_manager = diversity_manager or DiversityManager(config)
        self.population = []
        self.generation = 0

    def evolve_population(self, fitness_scores: List[float]):
        """Enhanced evolution with diversity maintenance."""

        # Assess current diversity state
        diversity_metrics = self.diversity_manager.calculate_diversity(
            self.population, fitness_scores
        )

        # Check for intervention needs
        interventions = self.diversity_manager.check_intervention_needs(
            diversity_metrics, self.generation
        )

        # Apply interventions if needed
        if interventions:
            self.population, fitness_scores = self.diversity_manager.apply_interventions(
                self.population, fitness_scores, interventions
            )

        # Adapt parameters based on diversity state
        adapted_params = self.diversity_manager.adapt_parameters(
            diversity_metrics, self.generation
        )

        # Perform selection with diversity awareness
        parents = self.diversity_manager.select_parents_with_diversity(
            self.population, fitness_scores, adapted_params
        )

        # Generate offspring with diversity-preserving operators
        offspring = self.diversity_manager.generate_diverse_offspring(
            parents, adapted_params
        )

        # Replace population with diversity-based replacement
        self.population = self.diversity_manager.replace_with_diversity(
            self.population, offspring, fitness_scores, adapted_params
        )

        self.generation += 1
```

### Diagnostics Integration
```python
# diagnostics/collector.py
class DiagnosticsCollector:
    def __init__(self, config: CollageConfig, diversity_manager: Optional[DiversityManager] = None):
        self.diversity_manager = diversity_manager
        self.diversity_history = []

    def record_generation(self, population: List[np.ndarray], fitness_scores: List[float]):
        """Enhanced generation recording with diversity tracking."""

        # Existing diagnostics recording...

        # Add diversity tracking
        if self.diversity_manager:
            diversity_metrics = self.diversity_manager.calculate_diversity(
                population, fitness_scores
            )
            self.diversity_history.append(diversity_metrics)

            # Add diversity data to generation stats
            self.current_generation.diversity_metrics = diversity_metrics
```

### CLI Integration
```python
# cli/main.py
@click.option('--diversity-strategy', type=click.Choice(['basic', 'adaptive', 'advanced']),
              default='basic', help='Diversity preservation strategy')
@click.option('--diversity-threshold', type=float, default=0.2,
              help='Diversity threshold for interventions')
def generate(target_image, source_directory, output_path, diversity_strategy, diversity_threshold, **kwargs):
    """Enhanced generate command with diversity options."""

    # Create diversity manager based on strategy
    if diversity_strategy == 'basic':
        diversity_manager = BasicDiversityManager(threshold=diversity_threshold)
    elif diversity_strategy == 'adaptive':
        diversity_manager = AdaptiveDiversityManager(threshold=diversity_threshold)
    elif diversity_strategy == 'advanced':
        diversity_manager = AdvancedDiversityManager(threshold=diversity_threshold)

    # Create generator with diversity manager
    generator = CollageGenerator(config, diversity_manager=diversity_manager)

    # Rest of generation logic...
```

This comprehensive diversity management system provides robust protection against premature convergence while maintaining the genetic algorithm's ability to find high-quality solutions. The modular design allows for gradual implementation and testing of different strategies.