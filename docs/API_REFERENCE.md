# Python API Reference - Reorganized

This guide provides comprehensive documentation for using the Image Collage Generator as a Python library, organized by user level and typical workflows.

## 📋 Table of Contents

### 🟢 ESSENTIAL APIs - Getting Started
1. [Installation and Quick Setup](#1-installation-and-quick-setup) ⭐
2. [Core Generation Workflow](#2-core-generation-workflow) ⭐
3. [Essential Classes Overview](#3-essential-classes-overview) ⭐

### 🟡 INTERMEDIATE APIs - Core Functionality
4. [Configuration Management](#4-configuration-management) ⭐
5. [Preset Configurations](#5-preset-configurations) ⭐
6. [Result Analysis and Export](#6-result-analysis-and-export) ⭐
7. [Rendering and Visualization](#7-rendering-and-visualization)

### 🔴 ADVANCED APIs - Power User Features
8. [Advanced Configuration Control](#8-advanced-configuration-control) 🔧
9. [GPU Acceleration](#9-gpu-acceleration) 🔧
10. [Diagnostics and Analysis](#10-diagnostics-and-analysis) 🔧
11. [Lineage Tracking and Genealogy](#11-lineage-tracking-and-genealogy) 🔧
12. [Performance Monitoring](#12-performance-monitoring) 🔧

### 🛠️ SPECIALIZED UTILITIES
13. [Image Processing Utilities](#13-image-processing-utilities) 🔧
14. [Error Handling and Exceptions](#14-error-handling-and-exceptions) 🔧

### 📖 USAGE PATTERNS AND WORKFLOWS
15. [Basic Usage Patterns](#15-basic-usage-patterns) ⭐
16. [Custom Configuration Workflows](#16-custom-configuration-workflows)
17. [Advanced Feature Integration](#17-advanced-feature-integration) 🔧
18. [Batch Processing](#18-batch-processing)
19. [Performance Optimization](#19-performance-optimization) 🔧
20. [Debugging and Troubleshooting](#20-debugging-and-troubleshooting) 🔧

---

## 🟢 ESSENTIAL APIs - Getting Started

These APIs are essential for every user. Start here for basic collage generation.

### 1. Installation and Quick Setup

⭐ **COMMONLY USED** - Required for all users

```python
# Standard installation
pip install -e .

# With all features
pip install -e ".[gpu,visualization,dev]"

# Essential imports for basic usage
from image_collage import CollageGenerator
from image_collage.config import PresetConfigs
```

**Quick Start (30 seconds):**
```python
# One-liner for instant results
from image_collage import CollageGenerator
from image_collage.config import PresetConfigs

generator = CollageGenerator(PresetConfigs.demo())
generator.load_target("target.jpg")
generator.load_sources("source_images/")
result = generator.generate()
generator.export(result, "output.png")
```

### 2. Core Generation Workflow

⭐ **COMMONLY USED** - The standard 4-step workflow

**Step 1: Create Generator**
```python
generator = CollageGenerator(config)
```

**Step 2: Load Images**
```python
generator.load_target("target.jpg")      # Load target image
source_count = generator.load_sources("sources/")  # Load source collection
```

**Step 3: Generate Collage**
```python
result = generator.generate()
```

**Step 4: Export Result**
```python
generator.export(result, "output.png")
```

### 3. Essential Classes Overview

⭐ **COMMONLY USED** - Core classes every user needs

#### CollageGenerator
**The main interface - your starting point for all collage generation.**

```python
class CollageGenerator:
    """Main interface for generating image collages using genetic algorithms."""

    def __init__(self, config: CollageConfig):
        """Initialize generator with configuration."""

    def load_target(self, target_path: str) -> None:
        """Load target image to recreate as mosaic."""

    def load_sources(self, source_directory: str) -> int:
        """Load source images from directory (recursive).

        Returns:
            Number of source images loaded
        """

    def generate(self, callback: Optional[Callable] = None) -> CollageResult:
        """Generate collage using genetic algorithm.

        Args:
            callback: Progress callback function(generation, fitness, preview)
        """

    def export(self, result: CollageResult, output_path: str) -> None:
        """Export generated collage to file."""

    # Essential properties
    @property
    def target_image(self) -> np.ndarray:
        """Get loaded target image."""

    @property
    def source_count(self) -> int:
        """Get number of loaded source images."""
```

#### CollageResult
**Contains everything from your generation process.**

```python
class CollageResult:
    """Result of collage generation process."""

    # Essential results
    collage_image: np.ndarray           # Final collage as numpy array
    fitness_score: float                # Final fitness value
    generations_used: int               # Generations completed
    processing_time: float              # Total processing time (seconds)

    # Advanced results
    tile_assignments: np.ndarray        # Grid of tile assignments
    evolution_frames: List[np.ndarray]  # Animation frames (if saved)
    fitness_history: List[float]        # Fitness progression
    convergence_achieved: bool          # Whether convergence was reached
```

---

## 🟡 INTERMEDIATE APIs - Core Functionality

Essential for customizing your collages and understanding the generation process.

### 4. Configuration Management

⭐ **COMMONLY USED** - Essential for any customization

#### CollageConfig
**The central configuration class controlling all aspects of generation.**

```python
class CollageConfig:
    """Complete configuration for collage generation."""

    def __init__(self):
        """Initialize with default settings."""
        # Essential settings
        self.grid_size: Tuple[int, int] = (50, 50)        # Collage grid dimensions
        self.tile_size: Tuple[int, int] = (32, 32)        # Individual tile size
        self.allow_duplicate_tiles: bool = True            # Allow tile reuse

        # Component configurations
        self.fitness_weights: FitnessWeights = FitnessWeights()
        self.genetic_params: GeneticParams = GeneticParams()
        self.gpu_config: GPUConfig = GPUConfig()

    @classmethod
    def load_from_file(cls, filepath: str) -> 'CollageConfig':
        """Load configuration from YAML file."""

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to YAML file."""

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
```

**Most Important Settings:**
- `grid_size`: Controls collage resolution (larger = more detail, slower)
- `fitness_weights`: Controls what the algorithm optimizes for
- `genetic_params.max_generations`: How long to evolve (more = better quality)

### 5. Preset Configurations

⭐ **COMMONLY USED** - Ready-to-use configurations for different needs

#### PresetConfigs
**Factory class providing optimized configurations for different use cases.**

```python
class PresetConfigs:
    """Factory for preset configurations."""

    @staticmethod
    def demo() -> CollageConfig:
        """🟢 Ultra-fast demo (30-60 seconds)
        - 15x20 grid, 30 generations
        - Perfect for testing and learning
        """

    @staticmethod
    def quick() -> CollageConfig:
        """🟢 Quick preview (2-5 minutes)
        - 20x20 grid, 100 generations
        - Good for previews and iteration
        """

    @staticmethod
    def balanced() -> CollageConfig:
        """🟡 Balanced quality/speed (10-20 minutes)
        - 50x50 grid, 1000 generations
        - Recommended for most users
        """

    @staticmethod
    def high() -> CollageConfig:
        """🟡 High quality (30-60 minutes)
        - 100x100 grid, 1500+ generations
        - For final production results
        """

    @staticmethod
    def gpu() -> CollageConfig:
        """🔴 GPU-accelerated (variable timing)
        - 150x150 grid, 3000 generations
        - Requires CUDA-compatible GPU
        """

    @staticmethod
    def extreme() -> CollageConfig:
        """🔴 Maximum quality (hours)
        - 300x300 grid, 5000 generations
        - For ultimate detail and quality
        """
```

**Choosing the Right Preset:**
- **New users**: Start with `demo()` or `quick()`
- **Most projects**: Use `balanced()` for good quality/speed trade-off
- **Final production**: Use `high()` for best results
- **GPU users**: Use `gpu()` for maximum performance
- **Ultimate quality**: Use `extreme()` for poster-sized prints

### 6. Result Analysis and Export

⭐ **COMMONLY USED** - Understanding and using your results

#### Analyzing Results
```python
# Check generation success
result = generator.generate()
print(f"Generated in {result.generations_used} generations")
print(f"Final fitness: {result.fitness_score:.6f}")
print(f"Processing time: {result.processing_time:.1f}s")
print(f"Converged: {result.convergence_achieved}")

# Access the collage
collage_array = result.collage_image  # numpy array
tile_grid = result.tile_assignments   # which source image for each tile
```

#### Export Options
```python
# Basic export
generator.export(result, "output.png")

# Different formats
generator.export(result, "output.jpg", format="JPEG")
generator.export(result, "output.tiff", format="TIFF")

# Access raw data
import cv2
cv2.imwrite("custom_output.png", result.collage_image)
```

### 7. Rendering and Visualization

**Creating animations, comparisons, and custom visualizations.**

#### CollageRenderer
```python
class CollageRenderer:
    """Renders collages and creates visualizations."""

    def render_collage(self, tile_assignments: np.ndarray) -> np.ndarray:
        """Render collage from tile assignments."""

    def create_evolution_animation(self,
                                  frames: List[np.ndarray],
                                  output_path: str,
                                  generation_numbers: Optional[List[int]] = None,
                                  fps: int = 2) -> None:
        """🟡 Create evolution animation GIF with generation numbers."""

    def create_comparison_image(self,
                               target: np.ndarray,
                               result: np.ndarray,
                               output_path: str) -> None:
        """🟡 Create side-by-side target vs result comparison."""

    def create_preview_grid(self,
                           individuals: List[np.ndarray],
                           output_path: str,
                           grid_size: Tuple[int, int] = (3, 3)) -> None:
        """🔧 Create grid preview of multiple individuals."""
```

**Common Usage:**
```python
# Access renderer through generator
renderer = generator.renderer

# Create evolution animation
result = generator.generate(save_evolution=True, evolution_interval=25)
renderer.create_evolution_animation(
    result.evolution_frames,
    "evolution.gif",
    generation_numbers=result.evolution_generation_numbers
)

# Create comparison
renderer.create_comparison_image(
    generator.target_image,
    result.collage_image,
    "comparison.jpg"
)
```

---

## 🔴 ADVANCED APIs - Power User Features

For users who need fine control over the generation process and want to leverage advanced features.

### 8. Advanced Configuration Control

🔧 **SPECIALIZED** - Fine-tuning generation parameters

#### FitnessWeights
**Controls what the algorithm optimizes for.**

```python
class FitnessWeights:
    """Weights for fitness evaluation components."""

    def __init__(self,
                 color: float = 0.4,        # Color similarity (most important)
                 luminance: float = 0.25,   # Brightness matching
                 texture: float = 0.2,      # Texture correlation
                 edges: float = 0.15):      # Edge preservation
        """Initialize fitness weights. Must sum to 1.0."""

    def validate(self) -> bool:
        """Check if weights sum to 1.0."""
```

**Common Adjustments:**
```python
# For portraits - emphasize color and luminance
portrait_weights = FitnessWeights(
    color=0.5, luminance=0.35, texture=0.05, edges=0.1
)

# For landscapes - emphasize edges and texture
landscape_weights = FitnessWeights(
    color=0.3, luminance=0.2, texture=0.3, edges=0.2
)

# For geometric art - emphasize edges
geometric_weights = FitnessWeights(
    color=0.25, luminance=0.25, texture=0.1, edges=0.4
)
```

#### GeneticParams
**Fine control over the evolutionary algorithm.**

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

        # Advanced features
        self.enable_adaptive_parameters: bool = True      # Adaptive rates
        self.enable_advanced_crossover: bool = True       # Multiple crossover strategies
        self.enable_advanced_mutation: bool = True        # Multiple mutation strategies
        self.enable_comprehensive_diversity: bool = True  # Advanced diversity metrics
        self.enable_island_model: bool = False            # Multi-population evolution
```

**Parameter Tuning Guidelines:**
- **Increase `population_size`** for better exploration (slower but better results)
- **Increase `max_generations`** for more evolution time
- **Decrease `mutation_rate`** for fine-tuning (0.02-0.03 for portraits)
- **Increase `crossover_rate`** for more genetic mixing (0.85-0.95)
- **Enable `island_model`** for very challenging images

### 9. GPU Acceleration

🔧 **SPECIALIZED** - Massive performance improvements with CUDA

#### GPUConfig
```python
class GPUConfig:
    """GPU acceleration configuration."""

    def __init__(self):
        self.enable_gpu: bool = False            # Enable GPU acceleration
        self.gpu_devices: List[int] = [0]        # GPU device IDs
        self.gpu_batch_size: int = 256           # Batch size for GPU operations
        self.gpu_memory_limit_gb: float = 20.0   # Memory limit per GPU
        self.auto_mixed_precision: bool = True   # Enable FP16 for speed

    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
```

**GPU Setup:**
```python
config = PresetConfigs.balanced()

# Check GPU availability
if config.gpu_config.is_available():
    print("GPU acceleration available!")
    config.gpu_config.enable_gpu = True

    # Multi-GPU setup (dual RTX 4090s)
    config.gpu_config.gpu_devices = [0, 1]
    config.gpu_config.gpu_batch_size = 1024
    config.gpu_config.gpu_memory_limit_gb = 20.0
else:
    print("GPU not available, using CPU")
```

**Performance Expectations:**
- **CPU**: 50x50 grid in ~10-20 minutes
- **Single GPU**: 150x150 grid in ~10-20 minutes (1000x speedup)
- **Dual GPU**: 300x300 grid in ~20-30 minutes

### 10. Diagnostics and Analysis

🔧 **SPECIALIZED** - Deep analysis of the evolution process

#### DiagnosticsCollector
**Comprehensive analysis of algorithm performance.**

```python
class DiagnosticsCollector:
    """Collects diagnostic data during evolution."""

    def record_generation(self,
                         generation: int,
                         population: List,
                         fitness_scores: List[float],
                         diversity_metrics: dict) -> None:
        """Record data for one generation."""

    def generate_reports(self) -> None:
        """Generate all diagnostic reports and visualizations:

        Creates:
        - dashboard.png: Comprehensive overview
        - fitness_evolution.png: Fitness progression analysis
        - genetic_operations.png: Operation effectiveness
        - performance_metrics.png: Timing and efficiency
        - population_analysis.png: Selection pressure and diversity
        - evolution_grid.png: Individual progression visualization
        - generation_data.csv: Raw numerical data
        - diagnostics_data.json: Structured analysis
        - summary.txt: Human-readable report
        """

    def export_data(self, format: str = "json") -> str:
        """Export collected data for custom analysis."""
```

**Using Diagnostics:**
```python
# Enable diagnostics during generation
result = generator.generate(diagnostics_folder="analysis/")

# Files created in analysis/:
# - dashboard.png (main overview)
# - fitness_evolution.png (progress charts)
# - genetic_operations.png (algorithm analysis)
# - performance_metrics.png (timing analysis)
# - population_analysis.png (selection pressure)
# - evolution_grid.png (visual progression)
# - generation_data.csv (raw data)
# - summary.txt (readable report)
```

### 11. Lineage Tracking and Genealogy

🔧 **SPECIALIZED** - Advanced genealogical analysis

#### LineageTracker
**Tracks family trees and evolutionary relationships.**

```python
class LineageTracker:
    """Tracks genealogical relationships between individuals."""

    def record_birth(self,
                    individual_id: str,
                    parent_ids: List[str],
                    birth_method: str,
                    generation: int) -> None:
        """Record birth of new individual."""

    def generate_lineage_analysis(self) -> None:
        """Generate complete lineage analysis:

        Creates 12+ visualization files:
        - lineage_dashboard.png: Complete overview
        - lineage_trees.png: Family tree structures
        - population_dynamics.png: Birth/death patterns
        - diversity_evolution.png: Multiple diversity metrics
        - fitness_lineages.png: Fitness inheritance
        - survival_curves.png: Individual survival analysis
        - migration_patterns.png: Population migrations
        - evolutionary_timeline.png: Complete timeline
        """

    def get_family_tree(self, individual_id: str) -> dict:
        """Get complete ancestry for individual."""
```

**Enabling Lineage Tracking:**
```python
config = PresetConfigs.high()
config.enable_lineage_tracking = True
config.lineage_output_dir = "genealogy/"

generator = CollageGenerator(config)
result = generator.generate()

# Creates genealogy/ directory with:
# - 12+ analysis visualizations
# - individuals.json (complete genealogical data)
# - lineage_trees.json (family structures)
# - migration_events.json (population events)
# - lineage_summary.json (statistical analysis)
```

### 12. Performance Monitoring

🔧 **SPECIALIZED** - System resource monitoring and optimization

#### PerformanceMonitor
```python
class PerformanceMonitor:
    """Monitors algorithm performance."""

    def start_timing(self) -> None:
        """Start performance timing."""

    def record_generation(self, generation_time: float) -> None:
        """Record timing for one generation."""

    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics."""
```

#### ResourceMonitor
```python
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

---

## 🛠️ SPECIALIZED UTILITIES

Supporting classes and functions for advanced use cases.

### 13. Image Processing Utilities

🔧 **SPECIALIZED** - Low-level image processing

#### ImageLoader
```python
class ImageLoader:
    """Loads and preprocesses images."""

    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """Load image from file (returns RGB numpy array)."""

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
```

#### FeatureExtractor
```python
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

### 14. Error Handling and Exceptions

🔧 **SPECIALIZED** - Comprehensive error handling

```python
from image_collage.exceptions import (
    ImageLoadError,           # Image file problems
    ConfigurationError,       # Invalid configuration
    GPUError,                # GPU acceleration issues
    InsufficientMemoryError  # Memory problems
)

try:
    generator = CollageGenerator(config)
    result = generator.generate()

except ImageLoadError as e:
    print(f"Image loading failed: {e}")

except ConfigurationError as e:
    print(f"Configuration error: {e}")

except GPUError as e:
    print(f"GPU error: {e}")
    # Fall back to CPU
    config.gpu_config.enable_gpu = False

except InsufficientMemoryError as e:
    print(f"Memory error: {e}")
    # Reduce memory usage
    config.grid_size = (50, 50)
    config.genetic_params.population_size = 50
```

---

## 📖 USAGE PATTERNS AND WORKFLOWS

Real-world examples organized by complexity and use case.

### 15. Basic Usage Patterns

⭐ **COMMONLY USED** - Essential workflows for all users

#### Complete Beginner Workflow
```python
from image_collage import CollageGenerator
from image_collage.config import PresetConfigs

# 🟢 Step 1: Choose a preset (start with demo)
config = PresetConfigs.demo()  # 30-60 seconds

# 🟢 Step 2: Create generator
generator = CollageGenerator(config)

# 🟢 Step 3: Load your images
generator.load_target("my_photo.jpg")
source_count = generator.load_sources("my_photos_folder/")
print(f"Loaded {source_count} source images")

# 🟢 Step 4: Generate with progress tracking
def show_progress(generation, fitness, preview):
    print(f"Generation {generation}: Fitness = {fitness:.6f}")

result = generator.generate(callback=show_progress)

# 🟢 Step 5: Save your result
generator.export(result, "my_collage.png")

# 🟢 Step 6: Check results
print(f"Completed in {result.processing_time:.1f} seconds")
print(f"Used {result.generations_used} generations")
print(f"Final fitness: {result.fitness_score:.6f}")
```

#### Standard Production Workflow
```python
# 🟡 For production-quality results
config = PresetConfigs.balanced()  # 10-20 minutes

generator = CollageGenerator(config)
generator.load_target("target.jpg")
generator.load_sources("source_collection/")

# Generate with evolution animation
result = generator.generate(
    save_evolution=True,
    evolution_interval=50
)

# Export main result
generator.export(result, "final_collage.png")

# Create animation and comparison
renderer = generator.renderer
renderer.create_evolution_animation(
    result.evolution_frames,
    "evolution.gif",
    generation_numbers=result.evolution_generation_numbers
)

renderer.create_comparison_image(
    generator.target_image,
    result.collage_image,
    "before_after.jpg"
)
```

### 16. Custom Configuration Workflows

**Tailoring the algorithm for specific image types and requirements.**

#### Portrait Optimization
```python
from image_collage.config import CollageConfig, FitnessWeights

# Custom configuration for portraits
config = CollageConfig()
config.grid_size = (80, 80)  # Good detail for faces

# Portrait-optimized fitness weights
config.fitness_weights = FitnessWeights(
    color=0.6,       # Emphasize color accuracy for skin tones
    luminance=0.25,  # Important for facial lighting
    texture=0.05,    # Minimize texture artifacts on skin
    edges=0.1        # Preserve important facial features
)

# Gentler evolution for better convergence
config.genetic_params.mutation_rate = 0.03
config.genetic_params.population_size = 120
config.genetic_params.max_generations = 1500

generator = CollageGenerator(config)
```

#### Landscape/Nature Optimization
```python
# Landscape-optimized configuration
config = CollageConfig()
config.grid_size = (100, 100)  # High detail for landscapes

# Landscape fitness weights
config.fitness_weights = FitnessWeights(
    color=0.35,      # Natural colors
    luminance=0.2,   # Sky and lighting gradients
    texture=0.25,    # Important for natural textures
    edges=0.2        # Mountain lines, tree edges, etc.
)

# More aggressive evolution for complex scenes
config.genetic_params.mutation_rate = 0.06
config.genetic_params.crossover_rate = 0.85
config.genetic_params.max_generations = 2000

generator = CollageGenerator(config)
```

#### Geometric/Art Optimization
```python
# Geometric art optimization
config = CollageConfig()
config.grid_size = (60, 60)
config.allow_duplicate_tiles = False  # Force unique tiles

# Edge-focused fitness for geometric shapes
config.fitness_weights = FitnessWeights(
    color=0.25,
    luminance=0.25,
    texture=0.1,
    edges=0.4       # Emphasize clean edges and shapes
)

generator = CollageGenerator(config)
```

### 17. Advanced Feature Integration

🔧 **SPECIALIZED** - Combining advanced features

#### Comprehensive Analysis Workflow
```python
# Advanced configuration with all analysis features
config = PresetConfigs.high()

# Enable comprehensive analysis
config.enable_diagnostics = True
config.diagnostics_output_dir = "detailed_analysis"
config.enable_lineage_tracking = True
config.lineage_output_dir = "genealogy_study"

# GPU acceleration if available
if config.gpu_config.is_available():
    config.gpu_config.enable_gpu = True
    config.gpu_config.gpu_batch_size = 1024

generator = CollageGenerator(config)
generator.load_target("complex_image.jpg")
generator.load_sources("large_collection/")

# Generate with all features
result = generator.generate(
    save_evolution=True,
    evolution_interval=25,
    diagnostics_folder="detailed_analysis"
)

# Creates comprehensive output:
# - final collage
# - evolution animation
# - detailed_analysis/ (10 diagnostic plots + data)
# - genealogy_study/ (12+ lineage analysis plots)
```

#### Multi-GPU High Performance Workflow
```python
# Maximum performance configuration
config = PresetConfigs.extreme()

# Multi-GPU setup (dual RTX 4090s)
config.gpu_config.enable_gpu = True
config.gpu_config.gpu_devices = [0, 1]
config.gpu_config.gpu_batch_size = 2048
config.gpu_config.gpu_memory_limit_gb = 22.0

# Enable all advanced features
config.genetic_params.enable_island_model = True
config.genetic_params.island_model_num_islands = 4
config.genetic_params.enable_adaptive_parameters = True

generator = CollageGenerator(config)

# For 300x300 grids (90,000 tiles)
config.grid_size = (300, 300)
config.genetic_params.max_generations = 5000
```

### 18. Batch Processing

**Processing multiple images efficiently.**

#### Batch Processing with Shared Sources
```python
import os
from pathlib import Path

def batch_process_portraits(target_dir: str, source_dir: str, output_dir: str):
    """Process multiple portraits with optimized settings."""

    # Portrait-optimized configuration
    config = PresetConfigs.balanced()
    config.fitness_weights = FitnessWeights(
        color=0.6, luminance=0.25, texture=0.05, edges=0.1
    )

    # Pre-load sources once for efficiency
    generator = CollageGenerator(config)
    source_count = generator.load_sources(source_dir)
    print(f"Loaded {source_count} source images for batch processing")

    # Process each target
    target_files = list(Path(target_dir).glob("*.jpg"))

    for i, target_file in enumerate(target_files, 1):
        print(f"Processing {i}/{len(target_files)}: {target_file.name}")

        # Load target
        generator.load_target(str(target_file))

        # Generate with progress
        def progress(gen, fitness, preview):
            if gen % 50 == 0:
                print(f"  Generation {gen}: {fitness:.6f}")

        result = generator.generate(callback=progress)

        # Export with descriptive name
        output_path = Path(output_dir) / f"{target_file.stem}_collage.png"
        generator.export(result, str(output_path))

        print(f"  Completed in {result.processing_time:.1f}s, "
              f"fitness: {result.fitness_score:.6f}\n")

# Usage
batch_process_portraits("family_photos/", "all_photos/", "collages/")
```

#### Configuration Comparison Batch
```python
def compare_presets(target_image: str, source_dir: str):
    """Compare different presets on the same image."""

    presets = {
        'demo': PresetConfigs.demo(),
        'quick': PresetConfigs.quick(),
        'balanced': PresetConfigs.balanced(),
        'high': PresetConfigs.high()
    }

    results = {}

    for name, config in presets.items():
        print(f"Testing {name} preset...")

        generator = CollageGenerator(config)
        generator.load_target(target_image)
        generator.load_sources(source_dir)

        result = generator.generate()
        generator.export(result, f"comparison_{name}.png")

        results[name] = {
            'time': result.processing_time,
            'fitness': result.fitness_score,
            'generations': result.generations_used
        }

        print(f"  Time: {result.processing_time:.1f}s")
        print(f"  Fitness: {result.fitness_score:.6f}")
        print(f"  Generations: {result.generations_used}\n")

    # Print comparison
    print("Preset Comparison:")
    for name, stats in results.items():
        print(f"{name:>10}: {stats['time']:>6.1f}s, "
              f"fitness: {stats['fitness']:.6f}, "
              f"gens: {stats['generations']}")

compare_presets("portrait.jpg", "family_photos/")
```

### 19. Performance Optimization

🔧 **SPECIALIZED** - Getting maximum performance

#### Memory-Optimized Configuration
```python
def optimize_for_memory(target_memory_gb: float = 8.0):
    """Create configuration optimized for limited memory."""

    config = PresetConfigs.balanced()

    # Estimate memory usage and adjust
    estimated_memory = (config.grid_size[0] * config.grid_size[1] *
                       config.genetic_params.population_size * 4) / 1e9

    if estimated_memory > target_memory_gb:
        # Reduce grid size first
        scale_factor = (target_memory_gb / estimated_memory) ** 0.5
        new_width = int(config.grid_size[0] * scale_factor)
        new_height = int(config.grid_size[1] * scale_factor)
        config.grid_size = (new_width, new_height)

        # Reduce population if still too high
        if estimated_memory > target_memory_gb * 1.5:
            config.genetic_params.population_size = 75

    print(f"Optimized configuration:")
    print(f"  Grid size: {config.grid_size}")
    print(f"  Population: {config.genetic_params.population_size}")
    print(f"  Estimated memory: {estimated_memory:.1f} GB")

    return config
```

#### CPU-Optimized Configuration
```python
import multiprocessing

def optimize_for_cpu():
    """Optimize configuration for CPU performance."""

    config = PresetConfigs.balanced()

    # Use all CPU cores
    config.performance_config.num_processes = multiprocessing.cpu_count()
    print(f"Using {config.performance_config.num_processes} CPU cores")

    # Optimize genetic parameters for CPU
    config.genetic_params.population_size = 100
    config.genetic_params.tournament_size = 3  # Smaller tournaments

    # Enable adaptive parameters for faster convergence
    config.genetic_params.enable_adaptive_parameters = True

    return config
```

#### Performance Monitoring During Generation
```python
from image_collage.monitoring import PerformanceMonitor
import psutil
import time

def monitored_generation(generator, monitor_interval: int = 10):
    """Generate with real-time performance monitoring."""

    monitor = PerformanceMonitor()
    process = psutil.Process()

    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    def progress_with_monitoring(generation, fitness, preview):
        if generation % monitor_interval == 0:
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = current_memory - start_memory

            print(f"Gen {generation:4d}: "
                  f"fitness={fitness:.6f}, "
                  f"memory={current_memory:.0f}MB (+{memory_delta:+.0f}MB)")

    monitor.start_timing()
    result = generator.generate(callback=progress_with_monitoring)

    # Final statistics
    stats = monitor.get_performance_stats()
    final_memory = process.memory_info().rss / 1024 / 1024

    print(f"\nPerformance Summary:")
    print(f"Total time: {result.processing_time:.1f}s")
    print(f"Avg generation: {stats.get('avg_generation_time', 0):.3f}s")
    print(f"Memory used: {final_memory - start_memory:.0f}MB")
    print(f"Final fitness: {result.fitness_score:.6f}")

    return result
```

### 20. Debugging and Troubleshooting

🔧 **SPECIALIZED** - Diagnosing and solving problems

#### Debug Configuration
```python
import logging

def create_debug_config():
    """Create configuration for debugging issues."""

    # Enable verbose logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Minimal configuration for fast testing
    config = PresetConfigs.demo()
    config.genetic_params.max_generations = 10  # Very short run

    # Enable all diagnostics
    config.enable_diagnostics = True
    config.diagnostics_output_dir = "debug_analysis"

    print("Debug configuration created:")
    print(f"  Grid size: {config.grid_size}")
    print(f"  Max generations: {config.genetic_params.max_generations}")
    print(f"  Diagnostics enabled: {config.enable_diagnostics}")

    return config

# Usage for debugging
debug_config = create_debug_config()
generator = CollageGenerator(debug_config)
```

#### Memory Usage Profiling
```python
import psutil
import os

def profile_memory_usage():
    """Profile memory usage throughout generation process."""

    process = psutil.Process(os.getpid())

    def get_memory_mb():
        return process.memory_info().rss / 1024 / 1024

    print(f"Initial memory: {get_memory_mb():.1f} MB")

    # Test different configurations
    configs = {
        'demo': PresetConfigs.demo(),
        'quick': PresetConfigs.quick(),
        'balanced': PresetConfigs.balanced()
    }

    for name, config in configs.items():
        print(f"\nTesting {name} preset:")

        generator = CollageGenerator(config)
        print(f"  After generator creation: {get_memory_mb():.1f} MB")

        generator.load_target("test_target.jpg")
        print(f"  After target load: {get_memory_mb():.1f} MB")

        generator.load_sources("test_sources/")
        print(f"  After sources load: {get_memory_mb():.1f} MB")

        # Run short generation
        config.genetic_params.max_generations = 5
        result = generator.generate()
        print(f"  After generation: {get_memory_mb():.1f} MB")

        # Cleanup
        del generator
        import gc
        gc.collect()
        print(f"  After cleanup: {get_memory_mb():.1f} MB")
```

#### Common Issues and Solutions
```python
def troubleshoot_common_issues():
    """Troubleshooting guide for common problems."""

    issues_and_solutions = {
        "Generation too slow": [
            "• Use PresetConfigs.demo() or quick() for testing",
            "• Reduce grid_size (50x50 instead of 100x100)",
            "• Reduce population_size (75 instead of 100)",
            "• Reduce max_generations (500 instead of 1000)",
            "• Enable GPU acceleration if available"
        ],

        "Out of memory errors": [
            "• Reduce grid_size significantly (30x30 for testing)",
            "• Reduce population_size (50 or fewer)",
            "• Limit source images (--max-images 500)",
            "• Use smaller tile_size",
            "• Close other applications"
        ],

        "Poor quality results": [
            "• Increase max_generations (1500+ for final results)",
            "• Increase population_size (150-200)",
            "• Adjust fitness_weights for your image type",
            "• Use more/better source images",
            "• Try different presets (balanced, high)"
        ],

        "GPU errors": [
            "• Check CUDA installation (nvidia-smi)",
            "• Update GPU drivers",
            "• Reduce gpu_batch_size (128 instead of 256)",
            "• Set gpu_memory_limit_gb lower",
            "• Fall back to CPU: config.gpu_config.enable_gpu = False"
        ]
    }

    print("Common Issues and Solutions:")
    print("=" * 50)

    for issue, solutions in issues_and_solutions.items():
        print(f"\n{issue}:")
        for solution in solutions:
            print(f"  {solution}")

    print(f"\nFor more help:")
    print(f"  • Check logs with logging.basicConfig(level=logging.DEBUG)")
    print(f"  • Use diagnostics_folder for detailed analysis")
    print(f"  • Profile memory usage with psutil")
    print(f"  • Test with minimal demo configuration first")

# Run troubleshooting guide
troubleshoot_common_issues()
```

---

## 📚 Cross-References and Related APIs

### Essential API Relationships
- **CollageGenerator** ↔ **CollageConfig**: Generator requires configuration
- **CollageConfig** ↔ **PresetConfigs**: Presets create configurations
- **CollageResult** ↔ **CollageRenderer**: Results used for visualization
- **FitnessWeights** → **CollageConfig**: Weights control optimization focus

### Workflow Dependencies
1. **Setup**: PresetConfigs → CollageConfig → CollageGenerator
2. **Generation**: load_target() + load_sources() → generate() → CollageResult
3. **Analysis**: CollageResult → DiagnosticsCollector/LineageTracker → Reports
4. **Output**: CollageResult → CollageRenderer → Animations/Comparisons

### Feature Integration Map
- **Basic Users**: CollageGenerator + PresetConfigs
- **Custom Users**: + CollageConfig + FitnessWeights
- **Advanced Users**: + GPU acceleration + Diagnostics
- **Research Users**: + LineageTracker + Performance monitoring

---

## 📚 Additional Resources

- **[GETTING_STARTED.md](GETTING_STARTED.md)**: Basic usage guide
- **[EXAMPLES.md](EXAMPLES.md)**: Real-world usage examples
- **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)**: Advanced configuration
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Common issues and solutions

---

This reorganized API reference provides a clear path from beginner to advanced usage, with logical groupings and comprehensive cross-references. Start with the 🟢 ESSENTIAL APIs and progress through 🟡 INTERMEDIATE and 🔴 ADVANCED features as your needs grow.