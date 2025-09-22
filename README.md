# Image Collage Generator

A genetic algorithm-based tool that creates photomosaic collages by arranging a collection of source images to visually approximate a target image.

## Features

- **Advanced Genetic Algorithm**: Multi-metric fitness evaluation with color, luminance, texture, and edge matching
- **GPU Acceleration**: CUDA support with CuPy for massive performance gains (1000x+ speedup)
- **Multi-GPU Support**: Utilize multiple GPUs simultaneously for extreme workloads
- **Comprehensive Diversity Management**: Advanced diversity preservation with fitness sharing, spatial awareness, and intelligent restart
- **Real-time Diversity Dashboard**: Live monitoring with alerts and intervention recommendations
- **Complete Lineage Tracking**: Genealogical analysis with 16 visualization plots and fitness component inheritance
- **Checkpoint System**: Crash recovery with resume capability for long-running simulations
- **Island Model Evolution**: Multi-population with migration for enhanced diversity
- **Adaptive Parameters**: Dynamic parameter adjustment based on diversity state
- **Configurable Parameters**: Flexible grid sizes, population settings, and genetic operators
- **Performance Optimized**: Parallel processing, intelligent caching, and memory management
- **Multiple Output Formats**: PNG, JPEG, TIFF with quality control
- **Preset Configurations**: Demo, quick, balanced, high-quality, GPU, and extreme modes
- **Progress Monitoring**: Real-time fitness tracking and generation statistics
- **Enhanced Diagnostics**: 10 visual reports + 3 data files with comprehensive diversity and evolution analysis
- **Animated Evolution**: Generate GIFs with generation numbers showing algorithm progression
- **Organized Output**: Automatic timestamped directories for clean workspace management
- **CLI Interface**: User-friendly command-line tool with comprehensive options

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-collage-generator.git
cd image-collage-generator

# Install the package
pip install -e .

# For GPU acceleration (requires CUDA-compatible GPU)
pip install -e ".[gpu]"

# For visualization and diagnostics (includes matplotlib, seaborn, pandas)
pip install -e ".[visualization]"

# Complete installation with all features
pip install -e ".[gpu,visualization,dev]"
```

## Quick Start

### 🚀 Demo Mode (30 seconds)

```bash
# Ultra-fast demo with all outputs
image-collage demo target.jpg source_images/

# Demo with GPU acceleration and advanced features
image-collage demo target.jpg source_images/ --gpu --verbose \
  --track-lineage --enable-dashboard --save-checkpoints
```

This creates organized output in `demo_output_YYYYMMDD_HHMMSS/`:
- 📸 `collage.jpg` - Final collage result
- 🎬 `evolution.gif` - Evolution animation with generation numbers
- 📊 `comparison.jpg` - Target vs result side-by-side
- ⚙️ `config.yaml` - Configuration parameters used
- 📈 `diagnostics/` - 10 visual reports + 3 data files (with --diagnostics)
- 🧬 `lineage/` - 16 lineage analysis plots (with --track-lineage)
- 💾 `checkpoints/` - Crash recovery checkpoints (with --save-checkpoints)

### Basic Usage

```bash
# Generate a collage using balanced settings
image-collage generate target.jpg source_images/ output.png

# Demo preset for ultra-fast testing
image-collage generate target.jpg source_images/ output.png --preset demo

# Use quick preset for faster generation
image-collage generate target.jpg source_images/ output.png --preset quick

# GPU-accelerated high quality (requires dual RTX 4090s or similar)
image-collage generate target.jpg source_images/ output.png --preset gpu

# Ultimate quality with all advanced features
image-collage generate target.jpg source_images/ output.png \
  --preset extreme \
  --save-animation evolution.gif \
  --save-comparison comparison.jpg \
  --diagnostics analysis_folder/ \
  --track-lineage lineage_analysis/ \
  --save-checkpoints \
  --enable-dashboard \
  --enable-fitness-sharing \
  --enable-restart \
  --track-components \
  --gpu \
  --gpu-devices "0,1"

# High quality with custom parameters
image-collage generate target.jpg source_images/ output.png \
  --preset high \
  --grid-size 100 100 \
  --generations 2000 \
  --no-duplicates
```

**Note**:
- Source images are automatically discovered recursively in all subdirectories
- All outputs are organized in timestamped `output_YYYYMMDD_HHMMSS/` directories
- Evolution animations now include generation number titles

### Python API

```python
from image_collage import CollageGenerator
from image_collage.config import PresetConfigs

# Create generator with balanced preset
config = PresetConfigs.balanced()
generator = CollageGenerator(config)

# Load images
generator.load_target("target.jpg")
source_count = generator.load_sources("source_images/")

# Generate collage with progress callback
def progress(generation, fitness, preview):
    print(f"Generation {generation}: Fitness = {fitness:.6f}")

# Generate with all features enabled
result = generator.generate(
    callback=progress,
    save_evolution=True,  # Enable animation frames
    evolution_interval=50,  # Save every 50th generation
    diagnostics_folder="analysis/"  # Enable comprehensive diagnostics
)

# Save outputs
generator.export(result, "output.png", "PNG")

# Save animation with generation numbers if frames were collected
if result.evolution_frames:
    generator.renderer.create_evolution_animation(
        result.evolution_frames, "evolution.gif",
        generation_numbers=result.evolution_generation_numbers
    )

# Create comparison image
generator.renderer.create_comparison_image(
    generator.target_image, result.collage_image, "comparison.jpg"
)

print(f"Generated in {result.generations_used} generations")
print(f"Final fitness: {result.fitness_score:.6f}")
print(f"Processing time: {result.processing_time:.2f} seconds")
```

### Diagnostics-Only Analysis

```python
# Generate diagnostics without creating a collage
result = generator.generate(diagnostics_folder="detailed_analysis/")

# The analysis folder now contains:
# - 8 visualization charts
# - JSON data for custom analysis
# - CSV file for spreadsheet import
# - Human-readable summary report
```

## Configuration

### Preset Modes

- **Demo**: Ultra-fast (15x20 grid, 30 generations) - Perfect for testing and demonstrations
- **Quick**: Low resolution (20x20 grid, 100 generations) - Fast preview generation
- **Balanced**: Medium resolution (50x50 grid, 1000 generations) - Good quality/speed balance
- **High**: High resolution (100x100 grid, 1500+ generations) - Enhanced quality
- **GPU**: GPU-accelerated (150x150 grid, 3000 generations) - Requires CUDA GPU
- **Extreme**: Maximum quality (300x300 grid, 5000 generations) - Ultimate detail

## 📊 Diagnostics and Analysis

Generate comprehensive analysis of the genetic algorithm evolution process:

```bash
# Basic diagnostics
image-collage generate target.jpg sources/ output.jpg --diagnostics analysis/

# Complete analysis with all outputs
image-collage generate target.jpg sources/ output.jpg \
  --preset gpu \
  --save-animation evolution.gif \
  --save-comparison comparison.jpg \
  --diagnostics comprehensive_analysis/
```

### 📈 Enhanced Diagnostic Reports

The diagnostics folder contains 10 visualization plots + 3 data files:

**Visual Analysis (10 plots):**
- `dashboard.png` - Comprehensive overview with properly aligned statistics
- `fitness_evolution.png` - Dual-shaded fitness progression (best-average AND average-worst regions)
- `fitness_distribution.png` - Population diversity and convergence analysis (4 subplots)
- `genetic_operations.png` - Operation effectiveness with clear explanations (4 subplots)
- `performance_metrics.png` - Accurate success rate calculations and timing analysis (4 subplots)
- `population_analysis.png` - Selection pressure analysis with definition and tooltip (4 subplots)
- `evolution_grid.png` - Color-coded individual progression with explanations and color bar
- `comprehensive_diversity.png` - Advanced diversity metrics (Hamming, entropy, clustering) (6 subplots)
- `spatial_diversity.png` - Spatial pattern analysis for image collages (9 subplots)
- `advanced_metrics.png` - Parameter adaptation and evolution trends (4 subplots)

**Data Files (3 files):**
- `diagnostics_data.json` - Complete structured metrics for custom analysis
- `generation_data.csv` - Per-generation data for spreadsheet import with all diversity metrics
- `summary.txt` - Human-readable performance summary with insights

### 🧬 Lineage Tracking Analysis

With `--track-lineage`, generates 16 genealogical visualization plots:

**Lineage Reports (16 plots total):**
- `lineage_dashboard.png` - Comprehensive lineage overview
- `lineage_trees.png` - Genealogical tree structures
- `population_dynamics.png` - Population size and diversity evolution
- `diversity_evolution.png` - Diversity metrics over time
- `fitness_lineages.png` - Fitness inheritance patterns
- `birth_method_distribution.png` - Genetic operation success analysis
- `age_distribution.png` - Individual survival patterns
- `selection_pressure.png` - Selection dynamics over time
- `lineage_dominance.png` - Dominant lineage identification
- `genealogy_network.png` - Complete ancestry network
- `migration_patterns.png` - Inter-population migration (island model)
- `survival_curves.png` - Individual and lineage survival analysis
- `evolutionary_timeline.png` - Comprehensive evolution timeline
- `fitness_component_evolution.png` - Individual fitness component tracking
- `fitness_component_inheritance.png` - Component heritability analysis
- `component_breeding_success.png` - Top performers by fitness component

### 🔬 Enhanced Metrics Tracking

- **Fitness Evolution**: Best, average, worst with dual-shaded visualization regions
- **Genetic Operations**: Accurate success rates based on estimated total operations attempted
- **Population Dynamics**: Selection pressure (Avg - Best fitness) with clear definition
- **Comprehensive Diversity**: Hamming distance, entropy, clustering, spatial patterns
- **Fitness Components**: Individual tracking of color, luminance, texture, edge components
- **Lineage Analysis**: Complete genealogical tracking with inheritance patterns
- **Spatial Diversity**: Pattern entropy, clustering, edge patterns, quadrant analysis
- **Performance**: Processing time trends, evolution efficiency (fitness/second)
- **Algorithm Behavior**: Stagnation detection, convergence analysis, diversity tracking
- **Success Metrics**: Beneficial mutation/crossover rates, final convergence calculations
- **Adaptive Parameters**: Real-time parameter adjustment based on diversity state
- **Intervention Systems**: Automatic restart, fitness sharing, migration events

### Custom Configuration

Create a YAML configuration file:
```yaml
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [75, 75]
  tile_size: [32, 32]
  allow_duplicate_tiles: true
  enable_diagnostics: false
  diagnostics_output_dir: null
  enable_lineage_tracking: false
  lineage_output_dir: null

# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  # Core GA Parameters
  population_size: 150
  max_generations: 1200
  crossover_rate: 0.85
  mutation_rate: 0.06
  elitism_rate: 0.12
  # Termination Criteria
  convergence_threshold: 0.001
  early_stopping_patience: 50
  target_fitness: 0.0

# === FITNESS EVALUATION ===
fitness_evaluation:
  color_weight: 0.45
  luminance_weight: 0.25
  texture_weight: 0.20
  edges_weight: 0.10

# === PERFORMANCE SETTINGS ===
performance:
  enable_parallel_processing: true
  num_processes: 8

# === GPU ACCELERATION ===
gpu_acceleration:
  enable_gpu: false
  gpu_devices: [0, 1]
```

### Example Configuration Files

The [`example_configs/`](example_configs/) directory includes practical configuration examples:

- **[`my_config.yaml`](example_configs/my_config.yaml)** - Custom configuration example from the GETTING_STARTED guide
- **[`island_config.yaml`](example_configs/island_config.yaml)** - Multi-population island model with migration
- **[`expert_config.yaml`](example_configs/expert_config.yaml)** - Expert configuration with ALL advanced features enabled

Use with CLI:
```bash
# Using custom YAML config from GETTING_STARTED guide
image-collage generate target.jpg sources/ output.png --config example_configs/my_config.yaml

# Island model evolution with migration
image-collage generate target.jpg sources/ output.png --config example_configs/island_config.yaml

# Expert mode with all advanced features
image-collage generate target.jpg sources/ expert_result.png --config example_configs/expert_config.yaml \
  --save-animation evolution.gif --diagnostics analysis/ --track-lineage lineage/ --save-checkpoints

# Export preset configurations
image-collage export-config balanced_config.yaml --preset balanced
image-collage export-config gpu_config.yaml --preset gpu
```

## CLI Commands

### Generate Collage
```bash
image-collage generate [OPTIONS] TARGET_IMAGE SOURCE_DIRECTORY OUTPUT_PATH
```

**Options:**
- `--preset`: Choose quality preset (quick/balanced/high/gpu/extreme)
- `--config`: Use YAML configuration file
- `--grid-size`: Grid dimensions (width height)
- `--generations`: Maximum generations
- `--population-size`: GA population size
- `--mutation-rate`: Mutation rate (0.0-1.0)
- `--no-duplicates`: Prevent duplicate tiles
- `--edge-blending`: Enable smooth transitions
- `--parallel/--no-parallel`: Toggle parallel processing
- `--gpu/--no-gpu`: Enable GPU acceleration
- `--gpu-devices`: Specify GPU devices (e.g., "0,1")
- `--gpu-batch-size`: GPU batch size for processing
- `--format`: Output format (PNG/JPEG/TIFF)
- `--save-animation`: Save evolution GIF with generation number titles
- `--save-comparison`: Save target vs result side-by-side comparison
- `--diagnostics`: Generate 10 visual reports + 3 data files with comprehensive analysis
- `--track-lineage`: Enable complete lineage tracking with 16 genealogical plots
- `--save-checkpoints`: Enable crash recovery checkpoint system
- `--resume-from`: Resume from checkpoint directory
- `--enable-dashboard`: Enable real-time diversity monitoring dashboard
- `--enable-fitness-sharing`: Enable fitness sharing for diversity preservation
- `--enable-restart`: Enable intelligent population restart system
- `--track-components`: Track individual fitness component evolution
- `--verbose`: Enable detailed logging and output

### Analyze Images
```bash
image-collage analyze TARGET_IMAGE SOURCE_DIRECTORY
```

Analyzes your target image and source collection to provide optimization recommendations:

**Target Image Analysis:**
- Resolution and aspect ratio detection
- Optimal grid size recommendations based on aspect ratio
- Portrait/landscape/square format identification

**Source Collection Analysis:**
- Total source image count
- Grid size recommendations (with duplicate/no-duplicate suggestions)
- Preset recommendations based on collection size:
  - Small collections (< 400 images): `quick` preset
  - Large collections (> 2000 images): `high` preset
  - Medium collections: `balanced` preset

**Example Output:**
```
Target Image Analysis:
  Resolution: 1920x1080 pixels
  Aspect ratio: 1.778 (landscape)

Source Collection Analysis:
  Total images: 1,247

Recommended grid sizes (aspect ratio 1.778):
  36x20 (720 tiles, no duplicates, ✓ good aspect match)
  71x40 (2,840 tiles, with duplicates, ✓ good aspect match)
  107x60 (6,420 tiles, with duplicates, ⚠ aspect mismatch)

Recommendation: Use 'balanced' preset
```

Use this command before generation to optimize your settings for the best results.

### Resume from Checkpoint
```bash
image-collage resume CHECKPOINT_DIRECTORY
```

Resume a previously interrupted evolution from checkpoint files. Automatically detects configuration and continues from the last saved generation.

### Export Configuration
```bash
image-collage export-config output.yaml --preset balanced
```

### Generate Color Tiles
```bash
image-collage generate-color-tiles NUM_TILES OUTPUT_DIRECTORY [OPTIONS]
```

Creates a diverse set of single-color tiles spanning the RGB spectrum. Perfect for:
- Creating geometric/abstract collages with pure colors
- Testing the algorithm with controlled color palettes
- Educational demonstrations of color theory

**Options:**
- `--tile-size`: Size of generated tiles (width height) - default: 32x32
- `--prefix`: Filename prefix for generated tiles - default: "color_tile"
- `--preview`: Save color palette preview image
- `--analyze`: Show color distribution analysis

**Examples:**
```bash
# Generate 100 diverse color tiles
image-collage generate-color-tiles 100 my_colors/

# Generate with custom size and preview
image-collage generate-color-tiles 500 tiles/ --tile-size 64 64 --preview palette.png

# Generate with analysis
image-collage generate-color-tiles 50 colors/ --analyze
```

## Algorithm Details

### Fitness Function
The fitness evaluation combines multiple metrics:

- **Color Similarity (40%)**: CIEDE2000 color difference calculation
- **Luminance Matching (25%)**: Brightness distribution comparison  
- **Texture Correlation (20%)**: Local binary pattern analysis
- **Edge Preservation (15%)**: Sobel operator edge detection

### Genetic Operators
- **Selection**: Tournament selection with configurable tournament size
- **Crossover**: Two-point crossover with position swapping
- **Mutation**: Random tile replacement and position swapping
- **Elitism**: Preserves top individuals across generations

### Color Tile Generation
The diverse color generation algorithm ensures optimal RGB spectrum coverage:

- **Key Colors**: Always includes fundamental colors (black, white, primary, secondary)
- **HSV Distribution**: Uses golden ratio spacing in HSV space for perceptual uniformity
- **Saturation/Value Variation**: Combines different brightness and saturation levels
- **Perceptual Spacing**: Maximizes visual distinctiveness between generated colors

This approach produces more visually diverse results than random color generation.

## Performance Guidelines

### System Requirements
- **CPU**: Multi-core recommended (4+ cores optimal)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: SSD recommended for image I/O

### Optimization Tips
- Use appropriate grid size for your source collection
- Enable parallel processing for faster computation
- Consider using quick preset for initial testing
- Larger source collections generally produce better results

## Examples

### Photo Mosaic from Family Photos
```bash
# Create a family portrait mosaic from organized photo collection
# This will search family_photos/ and all subdirectories for images
image-collage generate family_portrait.jpg family_photos/ mosaic.png \
  --grid-size 80 80 \
  --no-duplicates \
  --edge-blending

# Example directory structure:
# family_photos/
# ├── vacations/
# │   ├── 2023/
# │   └── 2024/
# ├── birthdays/
# └── holidays/
```

### Geometric Collage with Color Tiles
```bash
# Generate a diverse set of 200 color tiles
image-collage generate-color-tiles 200 color_tiles/ --tile-size 48 48

# Create a geometric collage using only pure colors
image-collage generate target_portrait.jpg color_tiles/ geometric_result.png \
  --preset balanced \
  --grid-size 60 60 \
  --no-duplicates

# Generate with preview and analysis
image-collage generate-color-tiles 100 tiles/ \
  --preview color_palette.png \
  --analyze
```

### Artistic Collage with Custom Weights
```python
from image_collage import CollageGenerator
from image_collage.config import CollageConfig, FitnessWeights

config = CollageConfig()
config.fitness_weights = FitnessWeights(
    color=0.6,      # Prioritize color matching
    luminance=0.3,  # Secondary luminance
    texture=0.1,    # Minimal texture
    edges=0.0       # Ignore edges
)

generator = CollageGenerator(config)
# ... generate collage
```

## Development

### Running Tests
```bash
pytest image_collage/tests/
```

### Code Style
```bash
black image_collage/
flake8 image_collage/
mypy image_collage/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.