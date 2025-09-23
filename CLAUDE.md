# Claude Code Project Context

## 🚨 MANDATORY: Development Guidelines
**CRITICAL**: Before making ANY code changes, read and follow `CODING_GUIDELINES.md`

### 🤖 AI Assistant Session Startup:
**Use `cstart` alias to automatically enforce guidelines** - See `SESSION_STARTUP.md` for details

**Recommended alias for ~/.bashrc or ~/.zshrc:**
```bash
alias cstart='claude "Review SESSION_STARTUP.md and follow the instructions."'
```

### 👤 Human Developer Resources:
- **`DEVELOPER_GUIDELINES.md`** - Personal workflow and collaboration patterns
- **`CODING_GUIDELINES.md`** - Universal architecture principles
- **`SESSION_STARTUP.md`** - Session startup protocol template
- **`./scripts/dry_audit.sh`** - Automated compliance checking

### 🔍 Helpful Reminders:
If the human seems to be:
- Making requests without mentioning guidelines → Gently remind about CODING_GUIDELINES.md
- Rushing through features → Suggest reviewing DEVELOPER_GUIDELINES.md session startup ritual
- Accepting quick/messy solutions → Reference the "Quality Control Habits" section
- Asking for duplicated functionality → Run grep search and suggest utils/ consolidation
- Working across multiple sessions → Ask if they need a quick architecture review

## Project Overview
Image Collage Generator - A genetic algorithm-based tool that creates photomosaic collages by arranging source images to visually approximate a target image.

## Project Structure
```
image_collage/
├── utils/              # ⭐ CENTRALIZED UTILITIES (utils-first development)
│   ├── diversity_metrics.py  # All diversity/similarity calculations
│   ├── color_tile_generator.py # Color generation utilities
│   └── __init__.py           # Exports all utilities
├── core/                 # Main CollageGenerator class
├── genetic/             # Genetic Algorithm Engine with advanced diversity
├── preprocessing/       # Image loading and feature extraction
├── fitness/            # Multi-metric fitness evaluation
├── rendering/          # Collage rendering and output
├── cache/              # Performance caching system
├── config/             # Configuration management
├── cli/                # Command-line interface
├── diagnostics/        # Enhanced evolution diagnostics
├── lineage/            # Genealogical tracking and analysis
├── examples/           # Usage examples
└── tests/              # Test files (to be created)
```

### 🏗️ Architecture Notes
- **utils/ is the single source of truth** for all reusable calculations
- **All modules import from utils/** rather than implementing their own calculations
- **No duplicate implementations** - consolidated following DRY principles
- **Performance optimized** - O(n²) algorithms use sampling for large datasets

## Development Commands

### Installation & Setup
```bash
# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[gpu,visualization,dev]"
```

### Testing
```bash
# Run tests (when implemented)
pytest image_collage/tests/

# Run with coverage
pytest --cov=image_collage --cov-report=term-missing
```

### Code Quality
```bash
# Format code
black image_collage/

# Lint code
flake8 image_collage/

# Type checking
mypy image_collage/
```

### CLI Usage Examples
```bash
# DEMO MODE - Quick start (30-60 seconds, timestamped organized output)
image-collage demo target.jpg source_images/
image-collage demo target.jpg source_images/ --gpu --diagnostics --verbose

# Basic collage generation (creates output_YYYYMMDD_HHMMSS/ directory)
image-collage generate target.jpg source_images/ output.png

# Quick preview with all outputs in timestamped directory
image-collage generate target.jpg sources/ preview.png --preset quick \
  --save-animation evolution.gif --save-comparison comparison.jpg

# Advanced preset with anti-convergence evolution (prevents premature convergence)
image-collage generate target.jpg sources/ advanced.png --preset advanced \
  --save-animation evolution.gif --diagnostics advanced_analysis/

# Demo preset with comprehensive diagnostics and enhanced visualizations
image-collage generate target.jpg sources/ demo.png --preset demo \
  --diagnostics diagnostics_folder/

# Ultra-fast testing with evolution animation (includes generation numbers)
image-collage generate target.jpg sources/ demo.png --preset demo \
  --save-animation demo.gif --save-comparison demo_comparison.jpg \
  --diagnostics demo_analysis/

# CHECKPOINT SYSTEM - Crash recovery and resuming long runs
image-collage generate target.jpg sources/ result.png --preset high \
  --save-checkpoints --save-animation evolution.gif

# Resume from crashed/interrupted run (automatic checkpoint discovery)
image-collage resume target.jpg sources/ output_20231201_143022/

# Genealogical analysis with comprehensive lineage tracking
image-collage generate target.jpg sources/ lineage.png --preset advanced \
  --track-lineage lineage_analysis/ --diagnostics diagnostics/ \
  --seed 42

# Full analysis with all features: diagnostics, lineage, reproducibility
image-collage generate target.jpg sources/ complete.png --preset ultra \
  --diagnostics analysis/ --track-lineage genealogy/ \
  --save-animation evolution.gif --seed 123 --verbose

# GPU-accelerated generation (dual RTX 4090s)
image-collage generate target.jpg sources/ gpu_result.png --preset gpu

# Extreme quality with both GPUs, animated evolution, and full diagnostics
image-collage generate target.jpg sources/ ultimate.png \
  --preset extreme \
  --gpu \
  --gpu-devices "0,1" \
  --gpu-batch-size 1024 \
  --save-animation evolution_ultimate.gif \
  --save-comparison ultimate_comparison.jpg \
  --diagnostics ultimate_analysis/ \
  --track-lineage lineage_ultimate/ \
  --save-checkpoints \
  --enable-dashboard \
  --enable-fitness-sharing \
  --enable-restart \
  --track-components

# High quality with custom settings
image-collage generate target.jpg sources/ result.png \
  --preset high \
  --grid-size 100 100 \
  --generations 2000 \
  --no-duplicates

# Analyze images for recommendations
image-collage analyze target.jpg source_images/

# Generate diverse color tiles for geometric collages
image-collage generate-color-tiles 200 color_tiles/ --tile-size 48 48 --preview palette.png

# Export configuration
image-collage export-config config.yaml --preset gpu
image-collage export-config balanced.yaml --preset balanced

# Example with organized source directory structure:
# sources/
# ├── nature/
# │   ├── flowers/
# │   └── landscapes/
# ├── people/
# └── objects/
# All images in all subdirectories will be found automatically

# Output Organization:
# All outputs are now organized in timestamped directories:
# output_20231201_143022/
# ├── collage.jpg                 # Main output
# ├── evolution.gif              # Animation with generation numbers
# ├── comparison.jpg              # Target vs result comparison
# ├── config.yaml                # Configuration used
# ├── diagnostics/                # Enhanced diagnostic reports (10 plots + 3 data files)
# │   ├── dashboard.png           # Comprehensive overview with proper alignment
# │   ├── fitness_evolution.png   # Enhanced with dual shading regions
# │   ├── fitness_distribution.png # Population diversity and convergence analysis
# │   ├── genetic_operations.png  # Improved with operation explanations
# │   ├── performance_metrics.png # Accurate success rates reporting
# │   ├── population_analysis.png # Selection pressure with definition
# │   ├── evolution_grid.png      # Color-coded with explanations
# │   ├── comprehensive_diversity.png # Advanced diversity metrics (Hamming, entropy, clustering)
# │   ├── spatial_diversity.png   # Spatial pattern analysis (9 subplots)
# │   ├── advanced_metrics.png    # Parameter adaptation trends (4 subplots)
# │   ├── generation_data.csv     # Complete metrics including all diversity data
# │   ├── diagnostics_data.json   # Comprehensive structured metrics
# │   └── summary.txt             # Human-readable performance summary
# ├── lineage/                    # Lineage tracking reports (16 plots total)
# │   ├── lineage_dashboard.png   # Comprehensive lineage overview
# │   ├── lineage_trees.png       # Genealogical tree structures
# │   ├── population_dynamics.png # Population evolution patterns
# │   ├── diversity_evolution.png # Diversity metrics over time
# │   ├── fitness_lineages.png    # Fitness inheritance tracking
# │   ├── birth_method_distribution.png # Genetic operation success analysis
# │   ├── age_distribution.png    # Individual survival patterns
# │   ├── selection_pressure.png  # Selection dynamics over time
# │   ├── lineage_dominance.png   # Dominant lineage identification
# │   ├── genealogy_network.png   # Complete ancestry network
# │   ├── migration_patterns.png  # Inter-population migration (island model)
# │   ├── survival_curves.png     # Individual and lineage survival analysis
# │   ├── evolutionary_timeline.png # Comprehensive evolution timeline
# │   ├── fitness_component_evolution.png # Individual component tracking
# │   ├── fitness_component_inheritance.png # Component heritability
# │   └── component_breeding_success.png # Top performers by component
# └── checkpoints/                # Crash recovery checkpoints
#     ├── generation_050.checkpoint # Evolution state snapshots
#     ├── generation_100.checkpoint
#     └── checkpoint_metadata.json
```

### Python API Examples
```python
from image_collage import CollageGenerator, PresetConfigs

# Basic usage
generator = CollageGenerator(PresetConfigs.balanced())
generator.load_target("target.jpg")
generator.load_sources("source_images/")
result = generator.generate()
generator.export(result, "output.png")

# Custom configuration
from image_collage.config import CollageConfig, FitnessWeights

config = CollageConfig()
config.grid_size = (80, 80)
config.fitness_weights = FitnessWeights(color=0.5, luminance=0.3, texture=0.1, edges=0.1)

# Save and load configurations
config.save_to_file("my_config.yaml")

# Load from file
loaded_config = CollageConfig.load_from_file("my_config.yaml")
```

## Key Implementation Details

### Advanced Genetic Algorithm with Anti-Convergence
- **Adaptive Selection**: Tournament selection with diversity-aware parent selection
- **Enhanced Crossover**: Multiple strategies (uniform, block, two-point) with adaptive switching
- **Advanced Mutation**: Multi-strategy mutation with local shuffling and block mutations
- **Dynamic Elitism**: Adjusts elite count based on population diversity
- **Population Restart**: Automatic restart mechanism when stagnation occurs
- **Diversity Preservation**: Ensures minimum genetic diversity and prevents duplicates
- **Adaptive Parameters**: Dynamic adjustment of mutation/crossover rates based on evolution state

#### Anti-Convergence Mechanisms
1. **Stagnation Detection**: Tracks generations without improvement
2. **Diversity Monitoring**: Real-time population diversity assessment
3. **Adaptive Mutation Rates**: Increases mutation when diversity drops
4. **Elite Diversification**: Mutates elite individuals when diversity is low
5. **Population Restarts**: Replaces portion of population during severe stagnation
6. **Advanced Parent Selection**: Promotes diverse parent combinations

### Fitness Evaluation
- **Color Similarity (40%)**: CIEDE2000 color difference
- **Luminance Matching (25%)**: Brightness distribution comparison
- **Texture Correlation (20%)**: Local binary pattern analysis
- **Edge Preservation (15%)**: Sobel operator edge detection

### Performance Features
- **GPU acceleration** with CuPy for massive speedups (1000x+)
- **Multi-GPU support** for dual RTX 4090 configurations
- Parallel fitness evaluation using multiprocessing
- LRU cache for preprocessed images and features
- Incremental fitness calculation for mutations
- Memory monitoring and automatic quality reduction
- Recursive directory search for organized source collections

### Enhanced Diagnostics and Analysis
The diagnostics system provides comprehensive analysis of the genetic algorithm's performance through multiple visualization types:

#### Dashboard Overview (`dashboard.png`)
- **Main fitness evolution plot** showing best and average fitness over generations
- **Genetic operations summary** bar chart of beneficial mutations vs crossovers
- **Population diversity trends** tracking genetic diversity over time
- **Processing time distribution** histogram showing performance consistency
- **Convergence analysis** measuring rate of fitness improvement
- **Evolution summary statistics** with properly aligned columns showing:
  - Configuration parameters (grid size, population, rates)
  - Performance metrics (total time, generations, speed)
  - Success metrics (improvement percentage, final fitness, convergence)
  - Population dynamics (diversity, selection pressure)
  - Operation effectiveness (beneficial operations, success ratios)

#### Fitness Evolution Analysis (`fitness_evolution.png`)
- **Dual-shaded fitness evolution** with both best-average AND average-worst regions
- **Best, average, and worst fitness trends** over all generations
- **Fitness improvement rate** showing generation-to-generation changes
- Color-coded regions help visualize population fitness spread

#### Genetic Operations Analysis (`genetic_operations.png`)
- **Beneficial operations over time** tracking mutations and crossovers per generation
- **Cumulative success tracking** showing total beneficial operations
- **Rolling average success rates** smoothed over multiple generations
- **Operation effectiveness distribution** pie chart with clear explanation:
  - Shows relative contribution of mutations vs crossovers to fitness improvements
  - Includes explanatory text clarifying both operations work together
  - Displays their relative effectiveness, not mutual exclusivity

#### Performance Metrics Analysis (`performance_metrics.png`)
- **Processing time per generation** with trend analysis
- **Evolution efficiency** measuring fitness improvement per second
- **Memory usage patterns** estimated from population diversity
- **Accurate performance summary** with corrected success rate calculations:
  - Operation success rates based on estimated total operations attempted
  - Final convergence metrics using fitness variance over recent generations
  - Processing time statistics (total, average, min, max per generation)

#### Population Analysis (`population_analysis.png`)
- **Population fitness range** showing best, worst, and average with filled regions
- **Selection pressure analysis** with clear definition and tooltip:
  - **Definition**: Selection Pressure = Average Fitness - Best Fitness
  - **Interpretation**: Higher values indicate more room for improvement
  - **Trend**: Decreasing trend shows population convergence
- **Genetic diversity trends** tracking population uniqueness over time
- **Stagnation detection** showing generations since last fitness improvement

#### Evolution Grid Visualization (`evolution_grid.png`)
- **Best individuals heatmaps** showing source image arrangements across generations
- **Color-coded visualization** where colors represent different source images
- **Generation progression** from initial random arrangements to optimized layouts
- **Comprehensive color bar** with "Source Image Index" labeling
- **Explanatory text** describing what heatmaps represent and color meaning

#### Fitness Distribution Analysis (`fitness_distribution.png`)
- **Fitness standard deviation** showing population diversity in fitness values
- **Convergence rate analysis** measuring how quickly the algorithm converges
- **Population diversity over time** tracking genetic uniqueness
- **Performance consistency** showing processing time variations per generation

#### Data Export Formats
- **CSV format** (`generation_data.csv`) with all numerical metrics for external analysis
- **JSON format** (`diagnostics_data.json`) with complete structured data
- **Text summary** (`summary.txt`) with human-readable statistics and insights

#### Key Diagnostic Metrics Explained
- **Beneficial Mutation Rate**: Percentage of mutations that improved fitness
- **Beneficial Crossover Rate**: Percentage of crossovers that improved fitness
- **Selection Pressure**: Difference between average and best fitness (convergence indicator)
- **Population Diversity**: Average percentage of different tiles between individuals
- **Convergence Rate**: Rate of fitness improvement over recent generations
- **Evolution Efficiency**: Fitness improvement achieved per processing second

### Comprehensive Lineage Tracking System

The lineage tracking system provides detailed genealogical analysis of evolutionary dynamics:

#### Core Lineage Features
- **Individual Tracking**: Complete genealogy of every individual across all generations
- **Family Trees**: Visual representation of parent-child relationships
- **Birth Method Analysis**: Tracks how individuals were created (initial, crossover, mutation, immigration)
- **Age Distribution**: Monitors individual survival and generational persistence
- **Population Dynamics**: Birth/death rates, turnover, and demographic analysis

#### Advanced Genealogical Metrics
- **Hamming Diversity**: Distance-based population diversity measurement
- **Entropy Diversity**: Information-theoretic diversity analysis
- **Spatial Diversity**: Problem-specific spatial pattern analysis for image collages
- **Cluster Diversity**: Machine learning-based population clustering analysis
- **Selection Pressure**: Comprehensive fitness variance and selection intensity
- **Dominance Analysis**: Identification of successful lineages and their contributions

#### Lineage Visualization Suite
1. **Lineage Trees** (`lineage_trees.png`): Genealogical family trees for dominant lineages
2. **Population Dynamics** (`population_dynamics.png`): Population size, turnover, and extinction
3. **Diversity Evolution** (`diversity_evolution.png`): Multiple diversity metrics over time
4. **Fitness Lineages** (`fitness_lineages.png`): Fitness evolution of different family lines
5. **Birth Methods** (`birth_methods.png`): Distribution of creation methods over time
6. **Age Distribution** (`age_distribution.png`): Individual and population aging analysis
7. **Selection Pressure** (`selection_pressure.png`): Detailed selection dynamics
8. **Lineage Dominance** (`lineage_dominance.png`): Lineage size and contribution analysis
9. **Migration Patterns** (`migration_patterns.png`): Island model migration analysis
10. **Survival Curves** (`survival_curves.png`): Individual and lineage survival analysis
11. **Evolutionary Timeline** (`evolutionary_timeline.png`): Complete evolution overview
12. **Lineage Dashboard** (`lineage_dashboard.png`): Comprehensive analysis dashboard

#### Island Model Integration
When island model is enabled, lineage tracking includes:
- **Migration Event Tracking**: Records all inter-island migrations
- **Source Island Analysis**: Migration patterns and frequency
- **Immigrant Fitness Impact**: Effect of migrants on population fitness
- **Inter-Island Diversity**: Diversity analysis across island populations

### Checkpoint System and Crash Recovery

The checkpoint system provides robust crash recovery for long-running genetic algorithm evolution:

#### Core Checkpoint Features
- **Automatic State Capture**: Saves complete evolution state including population, fitness history, random states
- **Configurable Save Intervals**: Save checkpoints every N generations (default: 25)
- **Smart Cleanup**: Maintains only the most recent checkpoints (default: 5) to save disk space
- **Complete Reproducibility**: Preserves random states for deterministic resuming
- **Comprehensive Metadata**: Human-readable summaries alongside binary checkpoints

#### Checkpoint Contents
Each checkpoint preserves:
- **Population State**: Complete genetic population with fitness scores
- **Evolution Progress**: Generation number, fitness history, convergence state
- **Random States**: NumPy and Python random states for reproducibility
- **Configuration**: All settings used for the run
- **Optional Components**: Lineage data, diagnostics state, evolution frames
- **Timing Information**: Processing time and checkpoint timestamps

#### Crash Recovery Workflow
```bash
# Start generation with checkpoints
image-collage generate target.jpg sources/ result.png --save-checkpoints --preset high

# If crashed/interrupted, resume automatically
image-collage resume target.jpg sources/ output_20231201_143022/
```

#### Output Directory Structure
```
output_20231201_143022/
├── config.yaml                    # Original configuration
├── checkpoints/                   # Checkpoint directory
│   ├── checkpoint_gen_0025_*.pkl  # Binary checkpoint (generation 25)
│   ├── checkpoint_gen_0025_*_summary.json  # Human-readable summary
│   ├── checkpoint_gen_0050_*.pkl  # Latest checkpoint (generation 50)
│   └── checkpoint_gen_0050_*_summary.json
├── lineage/                       # Lineage tracking (if enabled)
└── diagnostics/                   # Diagnostics analysis (if enabled)
```

#### Crash Recovery Instructions
When checkpoints are enabled, the CLI automatically provides recovery instructions:
- **During setup**: Shows resume command format
- **On interruption/crash**: Displays exact command to continue from last checkpoint
- **In demo mode**: Always shows recovery instructions since checkpoints are enabled

#### YAML Configuration
```yaml
# Checkpoint system settings
enable_checkpoints: true     # Enable checkpoint saving
checkpoint_interval: 25      # Save every 25 generations
max_checkpoints: 5          # Keep 5 most recent checkpoints
```

#### Data Export and Analysis
- **JSON Data Export**: Complete genealogical data for custom analysis
- **Statistical Summaries**: Key metrics and lineage statistics
- **CSV Generation**: Generation-by-generation lineage data
- **Pickle Export**: Raw Python objects for advanced research

#### Usage Examples
```bash
# Enable lineage tracking with CLI
image-collage generate target.jpg sources/ result.png \
  --track-lineage genealogy/ --seed 42

# Configure in YAML
basic_settings:
  enable_lineage_tracking: true
  lineage_output_dir: "lineage_analysis"
  random_seed: 42
```

### Output Organization and Animation Features

#### Timestamped Output Directories
All generation outputs are automatically organized in timestamped directories to keep workspaces clean:
```
output_YYYYMMDD_HHMMSS/
├── collage.jpg                 # Final collage result
├── evolution.gif              # Animated evolution with generation numbers
├── comparison.jpg              # Side-by-side target vs result
├── config.yaml                # Configuration parameters used
├── diagnostics/                # Complete diagnostic analysis
│   ├── dashboard.png           # Comprehensive overview
│   ├── fitness_evolution.png   # Dual-shaded fitness progression
│   ├── genetic_operations.png  # Operation effectiveness analysis
│   ├── performance_metrics.png # Accurate performance statistics
│   ├── population_analysis.png # Selection pressure and diversity
│   ├── evolution_grid.png      # Color-coded individual progression
│   ├── fitness_distribution.png
│   ├── generation_data.csv
│   ├── diagnostics_data.json
│   └── summary.txt
└── lineage/                    # Genealogical analysis (if enabled)
    ├── lineage_dashboard.png   # Complete lineage overview
    ├── lineage_trees.png       # Family tree visualizations
    ├── population_dynamics.png # Birth/death rates and turnover
    ├── diversity_evolution.png # Multiple diversity metrics
    ├── fitness_lineages.png    # Fitness evolution by family
    ├── birth_methods.png       # Creation method distribution
    ├── age_distribution.png    # Individual and population aging
    ├── selection_pressure.png  # Selection dynamics analysis
    ├── lineage_dominance.png   # Lineage contribution analysis
    ├── migration_patterns.png  # Island model migrations
    ├── survival_curves.png     # Survival analysis
    ├── evolutionary_timeline.png # Complete evolution timeline
    ├── individuals.json        # Complete genealogical data
    ├── lineage_trees.json      # Family tree structures
    ├── generation_stats.json   # Generation-by-generation stats
    ├── migration_events.json   # Migration event records
    ├── lineage_summary.json    # Statistical summary
    └── lineage_data.pkl        # Raw data for advanced analysis
```

#### Enhanced Evolution Animation
The evolution animation (`evolution.gif`) now includes:
- **Generation number titles** on each frame showing progression
- **White title bar** with centered "Generation N" text
- **Smooth transitions** between evolutionary stages
- **Configurable frame intervals** for different animation speeds
- **High-quality preview rendering** optimized for animation

#### Comparison Visualization
The comparison image (`comparison.jpg`) provides:
- **Side-by-side layout** of target image and final collage result
- **Matched dimensions** with automatic resizing for fair comparison
- **PNG format** for lossless quality preservation

### Color Tile Generation
- **Diverse color sampling** using golden ratio HSV distribution
- **Perceptual color spacing** for maximum visual distinctiveness
- **RGB spectrum coverage** ensuring full color space representation
- **Customizable tile sizes** and batch generation
- **Preview palette generation** for color visualization
- **Distribution analysis** with RGB coverage metrics

### Configuration Presets
- **Demo**: 15x20 grid, 30 generations, ultra-fast testing (30-60 seconds) - *checkpoints always enabled*
- **Quick**: 20x20 grid, 100 generations, basic fitness - *checkpoints recommended*
- **Balanced**: 50x50 grid, 1000 generations, full fitness - *checkpoints recommended*
- **High**: 100x100 grid, 1500+ generations, enhanced fitness - *checkpoints highly recommended*
- **Advanced**: 60x60 grid, 1500 generations, **advanced evolution with anti-convergence** - *checkpoints highly recommended*
- **Ultra**: 80x80 grid, 2000 generations, **all DIVERSITY.md techniques** - *checkpoints highly recommended*
- **GPU**: 150x150 grid, 3000 generations, dual GPU support - *checkpoints essential*
- **Extreme**: 300x300 grid, 5000 generations, ultimate detail - *checkpoints essential*

#### Checkpoint Recommendations by Preset
- **Short runs** (Demo, Quick): Checkpoints optional but helpful for learning the system
- **Medium runs** (Balanced, High, Advanced, Ultra): Checkpoints recommended (save every 25-50 generations)
- **Long runs** (GPU, Extreme): Checkpoints essential (save every 10-25 generations)

## CLI Parameter to Configuration Mapping

### Core Parameters
| CLI Option | Configuration Path | Description |
|------------|-------------------|-------------|
| `--preset` | Uses PresetConfigs methods | Predefined configuration sets |
| `--config` | Loads custom YAML/JSON file | Custom configuration override |
| `--grid-size` | `CollageConfig.grid_size` | Grid dimensions (width, height) |
| `--seed` | `CollageConfig.random_seed` | Random seed for reproducibility |

### Genetic Algorithm Parameters
| CLI Option | Configuration Path | Default | Description |
|------------|-------------------|---------|-------------|
| `--generations` | `genetic_params.max_generations` | 1000 | Maximum evolution generations |
| `--population-size` | `genetic_params.population_size` | 100 | GA population size |
| `--mutation-rate` | `genetic_params.mutation_rate` | 0.05 | Mutation probability (0.0-1.0) |
| `--crossover-rate` | `genetic_params.crossover_rate` | 0.8 | Crossover probability (0.0-1.0) |
| `--elitism-rate` | `genetic_params.elitism_rate` | 0.1 | Elite preservation rate (0.0-1.0) |
| `--tournament-size` | `genetic_params.tournament_size` | 5 | Tournament selection size |
| `--convergence-threshold` | `genetic_params.convergence_threshold` | 0.001 | Convergence threshold for stopping |
| `--early-stopping-patience` | `genetic_params.early_stopping_patience` | 50 | Patience for early stopping |

### GPU Acceleration
| CLI Option | Configuration Path | Default | Description |
|------------|-------------------|---------|-------------|
| `--gpu/--no-gpu` | `gpu_config.enable_gpu` | False | Enable/disable GPU acceleration |
| `--gpu-devices` | `gpu_config.gpu_devices` | [0] | GPU device IDs (comma-separated) |
| `--gpu-batch-size` | `gpu_config.gpu_batch_size` | 256 | GPU batch size |

### Performance Settings
| CLI Option | Configuration Path | Default | Description |
|------------|-------------------|---------|-------------|
| `--parallel/--no-parallel` | `enable_parallel_processing` | True | Enable/disable parallel processing |
| `--processes` | `num_processes` | 4 | Number of parallel processes |

### Output Settings
| CLI Option | Configuration Path | Default | Description |
|------------|-------------------|---------|-------------|
| `--no-duplicates` | `allow_duplicate_tiles` (inverted) | True | Prevent duplicate tiles |
| `--edge-blending` | `enable_edge_blending` | False | Enable edge blending between tiles |
| `--quality` | `output_quality` | 95 | JPEG quality (1-100) |

### Advanced Features
| CLI Option | Configuration Path | Default | Description |
|------------|-------------------|---------|-------------|
| `--save-checkpoints` | `enable_checkpoints` | False | Save evolution checkpoints for resuming |
| `--enable-fitness-sharing` | `enable_fitness_sharing` | False | Enable fitness sharing for diversity |
| `--enable-intelligent-restart` | `enable_intelligent_restart` | False | Enable intelligent population restart |
| `--enable-diversity-dashboard` | `enable_diversity_dashboard` | False | Enable real-time diversity dashboard |
| `--track-components` | `enable_component_tracking` | False | Track individual fitness components |

### Output-Only Options (Not Stored in Config)
| CLI Option | Description |
|------------|-------------|
| `--format` | Output image format (PNG, JPEG, TIFF) |
| `--save-animation` | Save evolution animation as GIF |
| `--save-comparison` | Save target vs result comparison |
| `--diagnostics` | Save comprehensive diagnostics |
| `--track-lineage` | Track genealogy and save lineage analysis |
| `--verbose` | Enable verbose output |

### Configuration Parameters Not Available via CLI
**Genetic Parameters:**
- `elitism_rate`, `tournament_size`, `convergence_threshold`, `early_stopping_patience`, `target_fitness`
- `enable_adaptive_parameters`, `stagnation_threshold`, `diversity_threshold`, `restart_threshold`, `restart_ratio`
- `enable_advanced_crossover`, `enable_advanced_mutation`, `enable_comprehensive_diversity`, `enable_spatial_diversity`
- `enable_island_model`, `island_model_*` parameters

**Fitness Weights:**
- `color`, `luminance`, `texture`, `edges` weights (must sum to 1.0)

**Collage Config:**
- `tile_size`, `max_source_images`, `cache_size_mb`, `preview_frequency`
- `checkpoint_interval`, `max_checkpoints`, `checkpoint_dir`
- Various advanced diversity parameters
- GPU memory/batch parameters
- Lineage tracking parameters

**To use these parameters, create a custom YAML configuration file:**
```bash
image-collage generate target.jpg sources/ result.png --config custom_config.yaml
```

## Common Development Tasks

### Adding New Fitness Metrics
1. Extend `FitnessEvaluator` class in `fitness/evaluator.py`
2. Add weight parameter to `FitnessWeights` in `config/settings.py`
3. Update fitness calculation in `_evaluate_tile_fitness()`

### Adding New Genetic Operators
1. Extend `GeneticAlgorithmEngine` in `genetic/ga_engine.py`
2. Add configuration parameters to `GeneticParams`
3. Update `evolve_population()` method

### Adding CLI Commands
1. Add new command function in `cli/main.py`
2. Use `@cli.command()` decorator
3. Add appropriate click options and arguments

### Performance Optimization Areas
- GPU acceleration with CuPy/CUDA
- JIT compilation with Numba
- Advanced caching strategies
- Memory-mapped file I/O for large collections

## Dependencies
- **Core**: opencv-python, numpy, scikit-image, Pillow
- **CLI**: click, tqdm
- **Config**: pydantic
- **GPU**: cupy (CUDA acceleration), CUDA 12.6+ compatible drivers
- **Diagnostics**: matplotlib, seaborn, pandas (visualization and analysis)
- **Optional**: numba (JIT compilation for performance)

## Testing Strategy (To Implement)
- Unit tests for each component
- Integration tests for full pipeline
- Performance benchmarks
- Visual quality regression tests
- Memory usage tests

## Known Limitations
- No GPU acceleration implemented yet
- Limited to RGB images
- Memory usage scales with source collection size
- No resumable generation for long runs

## Future Enhancements
- GPU-accelerated fitness evaluation
- Advanced texture analysis methods
- Multi-objective optimization
- Interactive parameter tuning
- Web interface
- Video collage generation