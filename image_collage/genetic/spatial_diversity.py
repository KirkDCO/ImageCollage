"""
Spatial diversity management specifically designed for image collage generation.

This module implements diversity preservation techniques that understand the spatial
nature of tile arrangements in photomosaic generation.
"""

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Any
from collections import defaultdict


class SpatialDiversityManager:
    """Manages diversity preservation using spatial awareness for image collage."""

    def __init__(self, grid_size: Tuple[int, int], num_source_images: int):
        self.grid_width, self.grid_height = grid_size
        self.num_source_images = num_source_images
        self.total_positions = self.grid_width * self.grid_height

        # Spatial diversity tracking
        self.spatial_diversity_history = []
        self.pattern_frequency_cache = {}

    def calculate_spatial_diversity(self, population: List[np.ndarray]) -> Dict[str, float]:
        """Calculate comprehensive spatial diversity metrics for tile arrangements."""
        metrics = {}

        # Core spatial diversity measures
        metrics['local_pattern_entropy'] = self._calculate_local_pattern_entropy(population)
        metrics['spatial_clustering'] = self._calculate_spatial_clustering(population)
        metrics['edge_pattern_diversity'] = self._calculate_edge_pattern_diversity(population)
        metrics['quadrant_diversity'] = self._calculate_quadrant_diversity(population)
        metrics['neighbor_similarity'] = self._calculate_neighbor_similarity(population)

        # Position-wise analysis
        metrics['position_wise_entropy'] = self._calculate_position_wise_entropy(population)
        metrics['tile_distribution_variance'] = self._calculate_tile_distribution_variance(population)

        # Spatial structure analysis
        metrics['contiguous_regions'] = self._calculate_contiguous_regions(population)
        metrics['spatial_autocorrelation'] = self._calculate_spatial_autocorrelation(population)

        # Combined spatial diversity score
        metrics['spatial_diversity_score'] = self._calculate_combined_spatial_score(metrics, len(population))

        self.spatial_diversity_history.append(metrics)
        return metrics

    def _calculate_local_pattern_entropy(self, population: List[np.ndarray]) -> float:
        """Calculate entropy of local 2x2 and 3x3 patterns."""
        pattern_counts_2x2 = defaultdict(int)
        pattern_counts_3x3 = defaultdict(int)

        for individual in population:
            grid = individual.reshape(self.grid_height, self.grid_width)

            # Extract all 2x2 patterns
            for y in range(self.grid_height - 1):
                for x in range(self.grid_width - 1):
                    pattern = tuple(grid[y:y+2, x:x+2].flatten())
                    pattern_counts_2x2[pattern] += 1

            # Extract all 3x3 patterns (center + neighbors)
            for y in range(1, self.grid_height - 1):
                for x in range(1, self.grid_width - 1):
                    pattern = tuple(grid[y-1:y+2, x-1:x+2].flatten())
                    pattern_counts_3x3[pattern] += 1

        # Calculate Shannon entropy for both pattern types
        entropy_2x2 = self._shannon_entropy(list(pattern_counts_2x2.values()))
        entropy_3x3 = self._shannon_entropy(list(pattern_counts_3x3.values()))

        # Weighted combination (3x3 patterns are more informative)
        return 0.4 * entropy_2x2 + 0.6 * entropy_3x3

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
                    neighbors = self._get_neighbor_tiles(grid, x, y)

                    if neighbors:
                        current_tile = grid[y, x]
                        similar_neighbors = sum(1 for n in neighbors if n == current_tile)
                        similarity = similar_neighbors / len(neighbors)
                        similarity_sum += similarity
                        comparisons += 1

            clustering_score = similarity_sum / comparisons if comparisons > 0 else 0
            clustering_scores.append(clustering_score)

        # Return variance in clustering scores (higher = more diverse)
        return np.var(clustering_scores) if len(clustering_scores) > 1 else 0

    def _calculate_edge_pattern_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate diversity in edge/border patterns."""
        edge_patterns = defaultdict(int)

        for individual in population:
            grid = individual.reshape(self.grid_height, self.grid_width)

            # Top and bottom edges
            for x in range(self.grid_width - 1):
                top_pair = (grid[0, x], grid[0, x + 1])
                bottom_pair = (grid[-1, x], grid[-1, x + 1])
                edge_patterns[('top', top_pair)] += 1
                edge_patterns[('bottom', bottom_pair)] += 1

            # Left and right edges
            for y in range(self.grid_height - 1):
                left_pair = (grid[y, 0], grid[y + 1, 0])
                right_pair = (grid[y, -1], grid[y + 1, -1])
                edge_patterns[('left', left_pair)] += 1
                edge_patterns[('right', right_pair)] += 1

        return self._shannon_entropy(list(edge_patterns.values()))

    def _calculate_quadrant_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate diversity between different quadrants of the grid."""
        quadrant_diversities = []

        mid_h, mid_w = self.grid_height // 2, self.grid_width // 2

        for individual in population:
            grid = individual.reshape(self.grid_height, self.grid_width)

            # Extract quadrants
            quadrants = [
                grid[:mid_h, :mid_w],          # Top-left
                grid[:mid_h, mid_w:],          # Top-right
                grid[mid_h:, :mid_w],          # Bottom-left
                grid[mid_h:, mid_w:]           # Bottom-right
            ]

            # Calculate diversity between quadrants
            inter_quadrant_distances = []
            for i in range(len(quadrants)):
                for j in range(i + 1, len(quadrants)):
                    q1_flat = quadrants[i].flatten()
                    q2_flat = quadrants[j].flatten()

                    # Handle different quadrant sizes by using minimum length
                    min_length = min(len(q1_flat), len(q2_flat))
                    q1_trimmed = q1_flat[:min_length]
                    q2_trimmed = q2_flat[:min_length]

                    # Hamming distance normalized by comparison size
                    distance = np.sum(q1_trimmed != q2_trimmed) / min_length if min_length > 0 else 0
                    inter_quadrant_distances.append(distance)

            avg_quadrant_diversity = np.mean(inter_quadrant_distances)
            quadrant_diversities.append(avg_quadrant_diversity)

        return np.mean(quadrant_diversities)

    def _calculate_neighbor_similarity(self, population: List[np.ndarray]) -> float:
        """Calculate average similarity between adjacent tiles."""
        similarity_scores = []

        for individual in population:
            grid = individual.reshape(self.grid_height, self.grid_width)

            total_similarity = 0
            total_pairs = 0

            # Horizontal neighbors
            for y in range(self.grid_height):
                for x in range(self.grid_width - 1):
                    if grid[y, x] == grid[y, x + 1]:
                        total_similarity += 1
                    total_pairs += 1

            # Vertical neighbors
            for y in range(self.grid_height - 1):
                for x in range(self.grid_width):
                    if grid[y, x] == grid[y + 1, x]:
                        total_similarity += 1
                    total_pairs += 1

            similarity_score = total_similarity / total_pairs if total_pairs > 0 else 0
            similarity_scores.append(similarity_score)

        # Return complement of average similarity (higher diversity = lower similarity)
        return 1.0 - np.mean(similarity_scores)

    def _calculate_position_wise_entropy(self, population: List[np.ndarray]) -> float:
        """Calculate entropy for each grid position and return average."""
        from ..utils.diversity_metrics import calculate_position_wise_entropy
        # The centralized function handles both 2D and flattened arrays
        return calculate_position_wise_entropy(population)

    def _calculate_tile_distribution_variance(self, population: List[np.ndarray]) -> float:
        """Calculate variance in tile usage across population."""
        tile_usage_counts = defaultdict(int)

        for individual in population:
            for tile_id in individual:
                # Convert numpy arrays to scalars for hashing
                if hasattr(tile_id, 'item'):
                    scalar_tile_id = tile_id.item()
                elif isinstance(tile_id, np.ndarray):
                    scalar_tile_id = int(tile_id.flatten()[0])
                else:
                    scalar_tile_id = tile_id
                tile_usage_counts[scalar_tile_id] += 1

        usage_values = list(tile_usage_counts.values())
        return np.var(usage_values) if len(usage_values) > 1 else 0

    def _calculate_contiguous_regions(self, population: List[np.ndarray]) -> float:
        """Calculate diversity in contiguous regions of same tiles."""
        region_size_diversities = []

        for individual in population:
            grid = individual.reshape(self.grid_height, self.grid_width)
            region_sizes = self._find_contiguous_regions(grid)

            # Calculate diversity in region sizes
            if len(region_sizes) > 1:
                region_diversity = np.var(region_sizes) / (np.mean(region_sizes) ** 2)
            else:
                region_diversity = 0

            region_size_diversities.append(region_diversity)

        return np.mean(region_size_diversities)

    def _calculate_spatial_autocorrelation(self, population: List[np.ndarray]) -> float:
        """Calculate spatial autocorrelation diversity."""
        autocorr_scores = []

        for individual in population:
            grid = individual.reshape(self.grid_height, self.grid_width)

            # Moran's I approximation for discrete data
            autocorr = self._morans_i_discrete(grid)
            autocorr_scores.append(autocorr)

        # Return variance in autocorrelation (more diverse = more varied spatial patterns)
        return np.var(autocorr_scores) if len(autocorr_scores) > 1 else 0

    def _calculate_combined_spatial_score(self, metrics: Dict[str, float], population_size: int) -> float:
        """Combine spatial metrics into a single diversity score."""
        weights = {
            'local_pattern_entropy': 0.25,
            'spatial_clustering': 0.20,
            'position_wise_entropy': 0.15,
            'quadrant_diversity': 0.15,
            'neighbor_similarity': 0.10,
            'edge_pattern_diversity': 0.10,
            'contiguous_regions': 0.05
        }

        # Normalize each metric to 0-1 range
        normalized_metrics = {}
        normalized_metrics['local_pattern_entropy'] = min(1.0, metrics['local_pattern_entropy'] / 10.0)
        normalized_metrics['spatial_clustering'] = min(1.0, metrics['spatial_clustering'] * 4.0)
        normalized_metrics['position_wise_entropy'] = min(1.0, metrics['position_wise_entropy'] / math.log2(population_size) if population_size > 1 else 0)
        normalized_metrics['quadrant_diversity'] = metrics['quadrant_diversity']
        normalized_metrics['neighbor_similarity'] = metrics['neighbor_similarity']
        normalized_metrics['edge_pattern_diversity'] = min(1.0, metrics['edge_pattern_diversity'] / 8.0)
        normalized_metrics['contiguous_regions'] = min(1.0, metrics['contiguous_regions'])

        # Calculate weighted score
        total_score = sum(weights[key] * normalized_metrics.get(key, 0) for key in weights.keys())
        return total_score

    def spatial_aware_mutation(self, individual: np.ndarray, population: List[np.ndarray]) -> np.ndarray:
        """Perform mutation considering spatial diversity patterns."""
        mutated = individual.copy()
        grid = individual.reshape(self.grid_height, self.grid_width)

        # Calculate position-wise diversity scores
        position_diversity = self._calculate_position_diversity_scores(individual, population)

        # Create mutation probability map (higher for low-diversity positions)
        mutation_weights = 1.0 / (position_diversity + 0.1)
        mutation_weights /= mutation_weights.sum()

        # Select positions for mutation based on spatial diversity
        num_mutations = max(1, int(len(individual) * 0.08))  # Slightly higher for spatial awareness
        mutation_positions = np.random.choice(
            len(individual), size=num_mutations, replace=False, p=mutation_weights
        )

        for pos in mutation_positions:
            y, x = pos // self.grid_width, pos % self.grid_width

            # Get current spatial context
            current_neighbors = self._get_neighbor_tiles(grid, x, y)

            # Choose tile that increases local diversity
            available_tiles = list(range(self.num_source_images))

            # Bias away from neighbor tiles and common tiles in this region
            local_region = self._get_local_region_tiles(grid, x, y, radius=2)
            common_in_region = self._get_common_tiles_in_region(local_region)

            # Weight tiles inversely to their frequency in local area
            tile_weights = []
            for tile_id in available_tiles:
                local_freq = local_region.count(tile_id) if local_region else 0
                neighbor_freq = current_neighbors.count(tile_id) if current_neighbors else 0

                # Higher weight for tiles that are rare in local context
                weight = 1.0 / (1.0 + local_freq + 2 * neighbor_freq)
                tile_weights.append(weight)

            # Normalize weights
            total_weight = sum(tile_weights)
            if total_weight > 0:
                tile_weights = [w / total_weight for w in tile_weights]

                # Select tile based on spatial diversity weights
                selected_tile = np.random.choice(available_tiles, p=tile_weights)
                mutated[pos] = selected_tile
            else:
                # Fallback to random selection
                mutated[pos] = random.choice(available_tiles)

        return mutated

    def _get_neighbor_tiles(self, grid: np.ndarray, x: int, y: int, include_diagonals: bool = True) -> List[int]:
        """Get tiles in neighboring positions."""
        neighbors = []

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected
        if include_diagonals:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # 8-connected

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.grid_height and 0 <= nx < self.grid_width:
                neighbors.append(grid[ny, nx])

        return neighbors

    def _get_local_region_tiles(self, grid: np.ndarray, x: int, y: int, radius: int = 2) -> List[int]:
        """Get all tiles in a local region around position."""
        tiles = []

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_height and 0 <= nx < self.grid_width:
                    tiles.append(grid[ny, nx])

        return tiles

    def _get_common_tiles_in_region(self, region_tiles: List[int]) -> List[int]:
        """Get tiles that appear frequently in a region."""
        if not region_tiles:
            return []

        tile_counts = defaultdict(int)
        for tile in region_tiles:
            tile_counts[tile] += 1

        threshold = len(region_tiles) * 0.3  # Tiles appearing in >30% of region
        common_tiles = [tile for tile, count in tile_counts.items() if count > threshold]

        return common_tiles

    def _shannon_entropy(self, counts: List[int]) -> float:
        """Calculate Shannon entropy from count values."""
        if not counts:
            return 0.0

        total = sum(counts)
        if total == 0:
            return 0.0

        entropy = 0
        for count in counts:
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)

        return entropy

    def _find_contiguous_regions(self, grid: np.ndarray) -> List[int]:
        """Find sizes of contiguous regions of same tiles."""
        visited = np.zeros_like(grid, dtype=bool)
        region_sizes = []

        def flood_fill(start_y: int, start_x: int, target_tile: int) -> int:
            """Flood fill to find contiguous region size."""
            stack = [(start_y, start_x)]
            size = 0

            while stack:
                y, x = stack.pop()

                if (y < 0 or y >= self.grid_height or x < 0 or x >= self.grid_width or
                    visited[y, x] or grid[y, x] != target_tile):
                    continue

                visited[y, x] = True
                size += 1

                # Add 4-connected neighbors
                stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])

            return size

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if not visited[y, x]:
                    region_size = flood_fill(y, x, grid[y, x])
                    if region_size > 0:
                        region_sizes.append(region_size)

        return region_sizes

    def _morans_i_discrete(self, grid: np.ndarray) -> float:
        """Calculate Moran's I for spatial autocorrelation (discrete version)."""
        n = grid.size

        # Calculate spatial weights matrix (binary adjacency)
        total_similarity = 0
        total_weights = 0

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                neighbors = []

                # 4-connected neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.grid_height and 0 <= nx < self.grid_width:
                        neighbors.append((ny, nx))

                # Calculate local similarity
                current_tile = grid[y, x]
                for ny, nx in neighbors:
                    neighbor_tile = grid[ny, nx]
                    similarity = 1.0 if current_tile == neighbor_tile else 0.0
                    total_similarity += similarity
                    total_weights += 1

        # Moran's I approximation
        if total_weights > 0:
            return total_similarity / total_weights
        else:
            return 0.0

    def _calculate_position_diversity_scores(self, individual: np.ndarray, population: List[np.ndarray]) -> np.ndarray:
        """Calculate diversity score for each position in the grid."""
        position_scores = np.zeros(len(individual))

        for pos in range(len(individual)):
            # Calculate how diverse this position is across the population
            position_values = [ind[pos] for ind in population]
            value_counts = defaultdict(int)
            for value in position_values:
                value_counts[value] += 1

            # Entropy-based diversity score
            entropy = self._shannon_entropy(list(value_counts.values()))
            position_scores[pos] = entropy

        return position_scores