"""
CLI Helper Functions

Common functionality shared across CLI commands to eliminate code duplication.
"""

from pathlib import Path
from typing import Optional, Callable
import click
import numpy as np

from ..core.collage_generator import CollageGenerator
from ..config.settings import CollageConfig


def create_generator(config: Optional[CollageConfig] = None) -> CollageGenerator:
    """Create and return a CollageGenerator instance.

    Args:
        config: Optional configuration. If None, uses default config.

    Returns:
        CollageGenerator instance
    """
    return CollageGenerator(config) if config else CollageGenerator()


def load_images(generator: CollageGenerator, target_image: Path, source_directory: Path) -> bool:
    """Load target and source images into the generator.

    Args:
        generator: CollageGenerator instance
        target_image: Path to target image
        source_directory: Path to source images directory

    Returns:
        True on success, False on failure
    """
    # Load target image
    if not generator.load_target(str(target_image)):
        click.echo("Error: Failed to load target image", err=True)
        return False

    # Load source images
    source_count = generator.load_sources(str(source_directory))
    if source_count == 0:
        click.echo("Error: No source images loaded", err=True)
        return False

    return True


def create_progress_callback(verbose: bool, interval: int = 10) -> Optional[Callable]:
    """Create a standardized progress callback function.

    Args:
        verbose: Whether to enable progress reporting
        interval: Generation interval for progress reports

    Returns:
        Progress callback function or None if not verbose
    """
    if not verbose:
        return None

    def progress(generation: int, fitness: float, preview: np.ndarray) -> None:
        if generation % interval == 0:
            click.echo(f"Generation {generation}: Fitness = {fitness:.6f}")

    return progress


def handle_error(message: str, error: Exception = None) -> None:
    """Standardized error handling and reporting.

    Args:
        message: Error message
        error: Optional exception object
    """
    if error:
        click.echo(f"Error: {message}: {error}", err=True)
    else:
        click.echo(f"Error: {message}", err=True)


def handle_generation_error(error: Exception, output_dir: Optional[str] = None,
                          save_checkpoints: bool = False) -> None:
    """Handle errors during generation with checkpoint recovery instructions.

    Args:
        error: The exception that occurred
        output_dir: Output directory path for checkpoint recovery
        save_checkpoints: Whether checkpoints were enabled
    """
    click.echo(f"❌ Error during generation: {error}", err=True)

    if save_checkpoints and output_dir:
        click.echo(f"💾 If checkpoints were saved, you can resume with:")
        click.echo(f"   image-collage resume <target_image> <source_directory> {output_dir}/")


def validate_positive_int(value: int, name: str) -> bool:
    """Validate that an integer value is positive.

    Args:
        value: Value to validate
        name: Name of the parameter for error messages

    Returns:
        True if valid, False otherwise
    """
    if value <= 0:
        click.echo(f"Error: {name} must be greater than 0", err=True)
        return False
    return True


def calculate_selection_pressure(fitness_scores: np.ndarray) -> float:
    """Calculate selection pressure using the standard definition.

    This provides a consistent definition across all components:
    Selection Pressure = (Average Fitness - Best Fitness) / Average Fitness

    This normalized metric indicates how much room for improvement exists:
    - 0.0: Population has converged (all individuals have same fitness)
    - Higher values: More diversity in fitness, more room for improvement

    Args:
        fitness_scores: Array of fitness values

    Returns:
        Selection pressure value
    """
    if len(fitness_scores) == 0:
        return 0.0

    avg_fitness = np.mean(fitness_scores)
    best_fitness = np.min(fitness_scores)  # Lower fitness is better

    if avg_fitness == 0:
        return 0.0

    return (avg_fitness - best_fitness) / avg_fitness


def calculate_unique_individuals(population) -> int:
    """Calculate the number of unique individuals in a population.

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Number of unique individuals
    """
    if not population:
        return 0

    return len(set(tuple(ind.flatten()) for ind in population))


def calculate_population_diversity(population) -> float:
    """Calculate population diversity as ratio of unique individuals.

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Diversity score (0.0 to 1.0)
    """
    from ..utils.diversity_metrics import calculate_unique_individuals_ratio
    return calculate_unique_individuals_ratio(population)


def calculate_hamming_diversity(population) -> float:
    """Calculate average Hamming distance between all pairs in population.

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Average Hamming distance (0.0 to 1.0)
    """
    from ..utils.diversity_metrics import calculate_hamming_distance_average
    return calculate_hamming_distance_average(population)


def calculate_position_wise_entropy(population) -> float:
    """Calculate position-wise entropy across population.

    Args:
        population: List of numpy arrays representing individuals

    Returns:
        Average entropy per position
    """
    from ..utils.diversity_metrics import calculate_position_wise_entropy as calc_entropy
    return calc_entropy(population)