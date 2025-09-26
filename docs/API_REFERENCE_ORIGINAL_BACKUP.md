# Python API Reference

This guide provides comprehensive documentation for using the Image Collage Generator as a Python library.

## 📦 Installation and Imports

```python
# Standard installation
pip install -e .

# With all features
pip install -e ".[gpu,visualization,dev]"

# Core imports
from image_collage import CollageGenerator
from image_collage.config import (
    CollageConfig, PresetConfigs, FitnessWeights,
    GeneticParams, GPUConfig, PerformanceConfig
)
from image_collage.core.result import CollageResult
from image_collage.rendering.renderer import CollageRenderer
```

## 🏗️ Core Classes

### CollageGenerator

The main class for creating collages.

```python
class CollageGenerator:
    """Main interface for generating image collages using genetic algorithms."""

    def __init__(self, config: CollageConfig):
        """Initialize generator with configuration.

        Args:
            config: CollageConfig object with all settings
        """

    def load_target(self, target_path: str) -> None:
        """Load target image to recreate as mosaic.

        Args:
            target_path: Path to target image file

        Raises:
            FileNotFoundError: Target image not found
            ValueError: Unsupported image format
        """

    def load_sources(self, source_directory: str) -> int:
        """Load source images from directory (recursive).

        Args:
            source_directory: Path to directory containing source images

        Returns:
            Number of source images loaded

        Raises:
            FileNotFoundError: Source directory not found
            ValueError: No valid images found
        """

    def generate(self,
                callback: Optional[Callable] = None,
                save_evolution: bool = False,
                evolution_interval: int = 50,
                diagnostics_folder: Optional[str] = None) -> CollageResult:
        """Generate collage using genetic algorithm.

        Args:
            callback: Progress callback function(generation, fitness, preview)
            save_evolution: Whether to save evolution frames for animation
            evolution_interval: Generations between evolution frames
            diagnostics_folder: Path to save diagnostic analysis

        Returns:
            CollageResult object with generated collage and metadata
        """

    def export(self, result: CollageResult, output_path: str,
              format: str = "PNG") -> None:
        """Export generated collage to file.

        Args:
            result: CollageResult from generate()
            output_path: Output file path
            format: Image format (PNG, JPEG, TIFF)
        """

    @property
    def target_image(self) -> np.ndarray:
        """Get loaded target image as numpy array."""

    @property
    def source_count(self) -> int:
        """Get number of loaded source images."""

    @property
    def renderer(self) -> CollageRenderer:
        """Get renderer for creating animations and comparisons."""
```

### CollageConfig

Configuration class containing all parameters.

```python
class CollageConfig:
    """Complete configuration for collage generation."""

    def __init__(self):
        """Initialize with default settings."""
        self.grid_size: Tuple[int, int] = (50, 50)
        self.tile_size: Tuple[int, int] = (32, 32)
        self.allow_duplicate_tiles: bool = True
        self.enable_diagnostics: bool = False
        self.diagnostics_output_dir: Optional[str] = None
        self.enable_lineage_tracking: bool = False
        self.lineage_output_dir: Optional[str] = None
        self.fitness_weights: FitnessWeights = FitnessWeights()
        self.genetic_params: GeneticParams = GeneticParams()
        self.gpu_config: GPUConfig = GPUConfig()
        self.performance_config: PerformanceConfig = PerformanceConfig()
        # ... other parameters

    @classmethod
    def load_from_file(cls, filepath: str) -> 'CollageConfig':
        """Load configuration from YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            CollageConfig object
        """

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to YAML file.

        Args:
            filepath: Output file path
        """

    def validate(self) -> List[str]:
        """Validate configuration parameters.

        Returns:
            List of validation error messages (empty if valid)
        """
```

### PresetConfigs

Factory class for predefined configurations.

```python
class PresetConfigs:
    """Factory for preset configurations."""

    @staticmethod
    def demo() -> CollageConfig:
        """Ultra-fast demo configuration (30-60 seconds).

        Returns:
            CollageConfig with demo settings
        """

    @staticmethod
    def quick() -> CollageConfig:
        """Quick preview configuration (2-5 minutes)."""

    @staticmethod
    def balanced() -> CollageConfig:
        """Balanced quality/speed configuration (10-20 minutes)."""

    @staticmethod
    def high() -> CollageConfig:
        """High quality configuration (30-60 minutes)."""

    @staticmethod
    def gpu() -> CollageConfig:
        """GPU-accelerated configuration (variable timing)."""

    @staticmethod
    def extreme() -> CollageConfig:
        """Maximum quality configuration (hours)."""
```

### CollageResult

Result object containing generated collage and metadata.

```python
class CollageResult:
    """Result of collage generation process."""

    def __init__(self):
        self.collage_image: np.ndarray           # Final collage as numpy array
        self.tile_assignments: np.ndarray        # Grid of tile assignments
        self.fitness_score: float                # Final fitness value
        self.generations_used: int               # Generations completed
        self.processing_time: float              # Total processing time (seconds)
        self.evolution_frames: List[np.ndarray]  # Animation frames (if saved)
        self.evolution_generation_numbers: List[int]  # Generation numbers for frames
        self.fitness_history: List[float]        # Fitness progression
        self.diversity_history: List[float]      # Diversity progression
        self.convergence_achieved: bool          # Whether convergence was reached
        self.config_used: CollageConfig          # Configuration used for generation
```

## ⚙️ Configuration Classes

### FitnessWeights

Controls the importance of different fitness components.

```python
class FitnessWeights:
    """Weights for fitness evaluation components."""

    def __init__(self,
                 color: float = 0.4,
                 luminance: float = 0.25,
                 texture: float = 0.2,
                 edges: float = 0.15):
        """Initialize fitness weights.

        Args:
            color: Color similarity weight (0.0-1.0)
            luminance: Brightness matching weight (0.0-1.0)
            texture: Texture correlation weight (0.0-1.0)
            edges: Edge preservation weight (0.0-1.0)

        Note:
            Weights must sum to 1.0
        """
        self.color = color
        self.luminance = luminance
        self.texture = texture
        self.edges = edges

    def validate(self) -> bool:
        """Check if weights sum to 1.0."""
        return abs(sum([self.color, self.luminance, self.texture, self.edges]) - 1.0) < 1e-6
```

### GeneticParams

Genetic algorithm parameters.

```python
class GeneticParams:
    """Genetic algorithm parameters."""

    def __init__(self):
        # Core GA Parameters
        self.population_size: int = 100          # Population size
        self.max_generations: int = 1000         # Maximum generations
        self.crossover_rate: float = 0.8         # Crossover probability
        self.mutation_rate: float = 0.05         # Mutation probability
        self.elitism_rate: float = 0.1           # Elite preservation rate
        self.tournament_size: int = 5            # Tournament selection size

        # Termination Criteria
        self.convergence_threshold: float = 0.001 # Convergence threshold
        self.early_stopping_patience: int = 50   # Early stopping patience
        self.target_fitness: float = 0.0         # Target fitness value

        # Advanced features
        self.enable_adaptive_parameters: bool = True
        self.enable_advanced_crossover: bool = True
        self.enable_advanced_mutation: bool = True
        self.enable_comprehensive_diversity: bool = True
        self.enable_spatial_diversity: bool = True
        self.enable_island_model: bool = False
        self.island_model_num_islands: int = 4
        self.island_model_migration_interval: int = 20
        self.island_model_migration_rate: float = 0.1
```

### GPUConfig

GPU acceleration settings.

```python
class GPUConfig:
    """GPU acceleration configuration."""

    def __init__(self):
        self.enable_gpu: bool = False            # Enable GPU acceleration
        self.gpu_devices: List[int] = [0]        # GPU device IDs
        self.gpu_batch_size: int = 256           # Batch size for GPU
        self.gpu_memory_limit_gb: float = 20.0   # Memory limit per GPU
        self.auto_mixed_precision: bool = True   # Enable FP16

    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import cupy
            return cupy.cuda.get_device_count() > 0
        except ImportError:
            return False
```

## 🎨 Rendering and Visualization

### CollageRenderer

Handles output rendering and visualization.

```python
class CollageRenderer:
    """Renders collages and creates visualizations."""

    def __init__(self, source_images: List[np.ndarray], tile_size: Tuple[int, int]):
        """Initialize renderer with source images and tile size."""

    def render_collage(self, tile_assignments: np.ndarray) -> np.ndarray:
        """Render collage from tile assignments.

        Args:
            tile_assignments: 2D array of source image indices

        Returns:
            Rendered collage as numpy array
        """

    def create_evolution_animation(self,
                                  frames: List[np.ndarray],
                                  output_path: str,
                                  generation_numbers: Optional[List[int]] = None,
                                  fps: int = 2) -> None:
        """Create evolution animation GIF.

        Args:
            frames: List of collage frames
            output_path: Output GIF path
            generation_numbers: Generation numbers for frame titles
            fps: Frames per second
        """

    def create_comparison_image(self,
                               target: np.ndarray,
                               result: np.ndarray,
                               output_path: str) -> None:
        """Create side-by-side comparison image.

        Args:
            target: Target image
            result: Generated collage
            output_path: Output image path
        """

    def create_preview_grid(self,
                           individuals: List[np.ndarray],
                           output_path: str,
                           grid_size: Tuple[int, int] = (3, 3)) -> None:
        """Create grid of multiple individuals for preview.

        Args:
            individuals: List of collage arrays
            output_path: Output image path
            grid_size: Preview grid dimensions
        """
```

## 📊 Analysis and Diagnostics

### DiagnosticsCollector

Collects and analyzes evolution data.

```python
class DiagnosticsCollector:
    """Collects diagnostic data during evolution."""

    def __init__(self, output_directory: str):
        """Initialize diagnostics collector."""

    def record_generation(self,
                         generation: int,
                         population: List,
                         fitness_scores: List[float],
                         diversity_metrics: dict) -> None:
        """Record data for one generation."""

    def generate_reports(self) -> None:
        """Generate all diagnostic reports and visualizations."""

    def export_data(self, format: str = "json") -> str:
        """Export collected data in specified format.

        Args:
            format: Export format ("json", "csv", "pickle")

        Returns:
            Path to exported data file
        """
```

### LineageTracker

Tracks genealogical information during evolution.

```python
class LineageTracker:
    """Tracks genealogical relationships between individuals."""

    def __init__(self, output_directory: str):
        """Initialize lineage tracker."""

    def record_birth(self,
                    individual_id: str,
                    parent_ids: List[str],
                    birth_method: str,
                    generation: int) -> None:
        """Record birth of new individual."""

    def generate_lineage_analysis(self) -> None:
        """Generate complete lineage analysis and visualizations."""

    def get_family_tree(self, individual_id: str) -> dict:
        """Get family tree for specific individual."""
```

## 🔧 Utility Functions

### Image Processing Utilities

```python
from image_collage.preprocessing import ImageLoader, FeatureExtractor

class ImageLoader:
    """Loads and preprocesses images."""

    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """Load image from file.

        Args:
            filepath: Path to image file

        Returns:
            Image as numpy array (RGB)
        """

    @staticmethod
    def load_images_from_directory(directory: str,
                                  max_images: Optional[int] = None) -> List[np.ndarray]:
        """Load all images from directory recursively."""

    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to specified size."""

    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """Validate image array format."""

class FeatureExtractor:
    """Extracts features from images for fitness evaluation."""

    @staticmethod
    def extract_color_features(image: np.ndarray) -> np.ndarray:
        """Extract color histogram features."""

    @staticmethod
    def extract_texture_features(image: np.ndarray) -> np.ndarray:
        """Extract texture features using Local Binary Patterns."""

    @staticmethod
    def extract_edge_features(image: np.ndarray) -> np.ndarray:
        """Extract edge features using Sobel operator."""
```

### Performance Monitoring

```python
from image_collage.monitoring import PerformanceMonitor, ResourceMonitor

class PerformanceMonitor:
    """Monitors algorithm performance."""

    def __init__(self):
        self.start_time: float
        self.generation_times: List[float]
        self.fitness_evaluations: int

    def start_timing(self) -> None:
        """Start performance timing."""

    def record_generation(self, generation_time: float) -> None:
        """Record timing for one generation."""

    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics."""

class ResourceMonitor:
    """Monitors system resource usage."""

    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage statistics."""

    @staticmethod
    def get_gpu_memory_usage() -> dict:
        """Get GPU memory usage (if available)."""

    @staticmethod
    def check_system_resources() -> dict:
        """Check available system resources."""
```

## 📝 Examples and Usage Patterns

### Basic Usage Pattern

```python
from image_collage import CollageGenerator
from image_collage.config import PresetConfigs

# Create generator with preset
config = PresetConfigs.balanced()
generator = CollageGenerator(config)

# Load images
generator.load_target("portrait.jpg")
source_count = generator.load_sources("family_photos/")
print(f"Loaded {source_count} source images")

# Generate with progress callback
def progress_callback(generation, fitness, preview):
    print(f"Generation {generation}: Fitness = {fitness:.6f}")

result = generator.generate(callback=progress_callback)

# Export result
generator.export(result, "family_collage.png")
print(f"Generated in {result.generations_used} generations")
print(f"Final fitness: {result.fitness_score:.6f}")
```

### Custom Configuration

```python
from image_collage.config import CollageConfig, FitnessWeights, GeneticParams

# Create custom configuration
config = CollageConfig()
config.grid_size = (80, 80)

# Custom fitness weights for portraits
config.fitness_weights = FitnessWeights(
    color=0.6,      # Emphasize color matching
    luminance=0.25, # Important for faces
    texture=0.05,   # Minimize texture artifacts
    edges=0.1       # Preserve facial edges
)

# Custom genetic parameters
config.genetic_params = GeneticParams()
config.genetic_params.population_size = 150
config.genetic_params.max_generations = 1500
config.genetic_params.mutation_rate = 0.04  # Gentler for portraits

# Enable GPU if available
if config.gpu_config.is_available():
    config.gpu_config.enable_gpu = True
    config.gpu_config.gpu_batch_size = 1024

generator = CollageGenerator(config)
```

### Advanced Features

```python
# Enable comprehensive analysis
config = PresetConfigs.high()
config.enable_lineage_tracking = True
config.lineage_output_dir = "genealogy_analysis"

generator = CollageGenerator(config)
generator.load_target("target.jpg")
generator.load_sources("sources/")

# Generate with evolution animation and diagnostics
result = generator.generate(
    save_evolution=True,
    evolution_interval=25,
    diagnostics_folder="comprehensive_analysis"
)

# Create evolution animation with generation numbers
generator.renderer.create_evolution_animation(
    result.evolution_frames,
    "evolution.gif",
    generation_numbers=result.evolution_generation_numbers,
    fps=3
)

# Create comparison image
generator.renderer.create_comparison_image(
    generator.target_image,
    result.collage_image,
    "comparison.jpg"
)
```

### Batch Processing

```python
import os
from pathlib import Path

def batch_process_targets(target_directory: str,
                         source_directory: str,
                         output_directory: str,
                         config: CollageConfig):
    """Process multiple target images with same source collection."""

    # Pre-load sources once
    generator = CollageGenerator(config)
    source_count = generator.load_sources(source_directory)
    print(f"Loaded {source_count} source images")

    # Process each target
    target_files = list(Path(target_directory).glob("*.jpg"))

    for target_file in target_files:
        print(f"Processing {target_file.name}...")

        # Load target
        generator.load_target(str(target_file))

        # Generate collage
        result = generator.generate()

        # Export result
        output_path = Path(output_directory) / f"{target_file.stem}_collage.png"
        generator.export(result, str(output_path))

        print(f"Completed {target_file.name} in {result.processing_time:.1f}s")

# Usage
config = PresetConfigs.balanced()
batch_process_targets("targets/", "sources/", "results/", config)
```

### Error Handling

```python
from image_collage.exceptions import (
    ImageLoadError, ConfigurationError,
    GPUError, InsufficientMemoryError
)

try:
    generator = CollageGenerator(config)
    generator.load_target("target.jpg")
    generator.load_sources("sources/")
    result = generator.generate()

except ImageLoadError as e:
    print(f"Image loading failed: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except GPUError as e:
    print(f"GPU error: {e}")
    # Fall back to CPU
    config.gpu_config.enable_gpu = False
    generator = CollageGenerator(config)
except InsufficientMemoryError as e:
    print(f"Memory error: {e}")
    # Reduce memory usage
    config.grid_size = (50, 50)
    config.genetic_params.population_size = 50
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Performance Optimization

```python
from image_collage.monitoring import PerformanceMonitor

# Monitor performance
monitor = PerformanceMonitor()

def optimized_generation():
    monitor.start_timing()

    # Use appropriate configuration for hardware
    config = PresetConfigs.balanced()

    # Enable GPU if available
    if config.gpu_config.is_available():
        config.gpu_config.enable_gpu = True
        # Optimize batch size for GPU memory
        config.gpu_config.gpu_batch_size = 1024

    # Optimize CPU usage
    import multiprocessing
    config.performance_config.num_processes = multiprocessing.cpu_count()

    generator = CollageGenerator(config)
    generator.load_target("target.jpg")
    generator.load_sources("sources/")

    result = generator.generate()

    # Get performance statistics
    stats = monitor.get_performance_stats()
    print(f"Processing time: {stats['total_time']:.1f}s")
    print(f"Average generation time: {stats['avg_generation_time']:.3f}s")

    return result
```

## 🔍 Debugging and Troubleshooting

### Debug Configuration

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use minimal configuration for testing
config = PresetConfigs.demo()
config.max_generations = 10  # Very short run

# Enable all diagnostics
generator = CollageGenerator(config)
result = generator.generate(
    diagnostics_folder="debug_analysis",
    save_evolution=True
)
```

### Memory Profiling

```python
import psutil
import os

def profile_memory_usage():
    """Profile memory usage during generation."""

    process = psutil.Process(os.getpid())

    def get_memory_mb():
        return process.memory_info().rss / 1024 / 1024

    print(f"Initial memory: {get_memory_mb():.1f} MB")

    config = PresetConfigs.quick()
    generator = CollageGenerator(config)
    print(f"After generator creation: {get_memory_mb():.1f} MB")

    generator.load_target("target.jpg")
    print(f"After target load: {get_memory_mb():.1f} MB")

    generator.load_sources("sources/")
    print(f"After sources load: {get_memory_mb():.1f} MB")

    result = generator.generate()
    print(f"After generation: {get_memory_mb():.1f} MB")
```

---

## 📚 Additional Resources

- **[GETTING_STARTED.md](GETTING_STARTED.md)**: Basic usage guide
- **[EXAMPLES.md](EXAMPLES.md)**: Real-world usage examples
- **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)**: Advanced configuration
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Common issues and solutions

This API reference provides comprehensive documentation for integrating the Image Collage Generator into your Python applications. Start with the basic usage patterns and gradually explore the advanced features as needed.