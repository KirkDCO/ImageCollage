# Example Configuration Files

This directory contains example configuration files for all available presets in the Image Collage Generator. Each configuration file demonstrates the optimal parameters for different use cases and quality requirements.

## Available Presets

### `demo.yaml` - Ultra-Fast Testing (30-60 seconds)
- **Grid Size**: 20×15 (300 tiles)
- **Population**: 30, Generations: 30
- **Purpose**: Quick testing and demonstrations
- **Fitness**: Color (70%) + Luminance (30%) for speed
- **Use Case**: Testing, demonstrations, rapid prototyping

### `quick.yaml` - Fast Results (2-5 minutes)
- **Grid Size**: 20×20 (400 tiles)
- **Population**: 50, Generations: 100
- **Purpose**: Quick results with decent quality
- **Fitness**: Balanced across all metrics
- **Use Case**: Preview generation, iterative design

### `balanced.yaml` - Default Quality (5-15 minutes)
- **Grid Size**: 50×50 (2,500 tiles)
- **Population**: 100, Generations: 1000
- **Purpose**: Good balance of quality and speed
- **Fitness**: Full multi-metric evaluation
- **Use Case**: Most general-purpose collages

### `high.yaml` - High Quality (15-45 minutes)
- **Grid Size**: 100×100 (10,000 tiles)
- **Population**: 150, Generations: 1500
- **Purpose**: High-quality results with fine detail
- **Fitness**: Enhanced evaluation with all metrics
- **Use Case**: Professional artwork, detailed collages

### `advanced.yaml` - Advanced Evolution (30-90 minutes)
- **Grid Size**: 100×100 (10,000 tiles)
- **Population**: 200, Generations: 2000
- **Purpose**: Advanced genetic algorithm features
- **Features**: All diversity preservation techniques enabled
- **Use Case**: Research, preventing convergence issues

### `ultra.yaml` - Ultra Quality (45-120 minutes)
- **Grid Size**: 80×80 (6,400 tiles)
- **Population**: 200, Generations: 2000
- **Purpose**: Maximum quality with all advanced features
- **Features**: Comprehensive diversity, island model, spatial diversity
- **Use Case**: Publication-quality artwork, research

### `gpu.yaml` - GPU Accelerated (Variable timing)
- **Grid Size**: 150×150 (22,500 tiles)
- **Population**: 200, Generations: 3000
- **Purpose**: GPU-accelerated high-performance generation
- **Requirements**: CUDA-compatible GPU
- **Use Case**: Large-scale collages, batch processing

### `extreme.yaml` - Maximum Quality (Hours)
- **Grid Size**: 300×300 (90,000 tiles)
- **Population**: 300, Generations: 5000
- **Purpose**: Ultimate quality for large-format printing
- **Requirements**: High-end system, lots of RAM
- **Use Case**: Large prints, exhibition pieces

## Special Example Configurations

These configurations demonstrate specific features and techniques referenced in the GETTING_STARTED guide:

### `my_config.yaml` - Custom Configuration Example (15-30 minutes)
- **Grid Size**: 60×60 (3,600 tiles)
- **Population**: 120, Generations: 800
- **Purpose**: Example from GETTING_STARTED guide demonstrating custom configuration
- **Features**: Custom fitness weights emphasizing color matching
- **Use Case**: Learning configuration customization, custom fitness priorities

### `island_config.yaml` - Multi-Population Evolution (20-40 minutes)
- **Grid Size**: 75×75 (5,625 tiles)
- **Population**: 150 (across 3 islands), Generations: 1200
- **Purpose**: Demonstrates island model evolution with migration
- **Features**: 3 islands with migration every 15 generations
- **Use Case**: Advanced evolution techniques, preventing convergence issues

### `expert_config.yaml` - ALL FEATURES ENABLED (2-6 hours)
- **Grid Size**: 200×200 (40,000 tiles)
- **Population**: 300, Generations: 2000
- **Purpose**: Comprehensive demonstration of every advanced feature
- **Features**: Island model, lineage tracking, checkpoints, diversity management, component tracking
- **Requirements**: High-end system, preferably dual GPU
- **Use Case**: Research, comprehensive testing, ultimate quality results

## Usage Examples

### Using a preset configuration:
```bash
# Use a preset directly
image-collage generate target.jpg sources/ output.png --preset balanced

# Use a specific config file
image-collage generate target.jpg sources/ output.png --config example_configs/high.yaml

# Modify a config file and use it
cp example_configs/balanced.yaml my_config.yaml
# Edit my_config.yaml as needed
image-collage generate target.jpg sources/ output.png --config my_config.yaml
```

### Using special example configurations:
```bash
# Custom configuration from GETTING_STARTED guide
image-collage generate target.jpg sources/ custom_result.png --config example_configs/my_config.yaml

# Island model evolution with migration
image-collage generate target.jpg sources/ island_result.png \
  --config example_configs/island_config.yaml \
  --track-lineage island_lineage/

# Expert mode with comprehensive analysis
image-collage generate target.jpg sources/ expert_result.png \
  --config example_configs/expert_config.yaml \
  --save-animation complete_evolution.gif \
  --save-comparison detailed_comparison.jpg \
  --diagnostics full_diagnostics/ \
  --track-lineage complete_genealogy/ \
  --save-checkpoints \
  --enable-dashboard \
  --verbose
```

### Customizing configurations:
```bash
# Start with a preset and override specific parameters
image-collage generate target.jpg sources/ output.png \\
  --preset high \\
  --grid-size 120 120 \\
  --generations 2000 \\
  --seed 42
```

## Configuration Sections

Each configuration file is organized into logical sections:

- **basic_settings**: Grid size, tile settings, diagnostics configuration, lineage tracking
- **genetic_algorithm**: Population, generations, termination criteria, crossover/mutation rates, advanced features
- **fitness_evaluation**: Weights for color, luminance, texture, edges
- **gpu_acceleration**: GPU devices, batch sizes, memory limits
- **performance**: Parallel processing, cache settings
- **output**: Quality settings, edge blending

## Reproducibility and Lineage Tracking

All configuration files support random seeds for reproducible results and lineage tracking for genealogical analysis:

```yaml
basic_settings:
  random_seed: 42  # Set to any integer for reproducible results
  # random_seed: null  # Use for non-deterministic behavior

  enable_diagnostics: true  # Enable comprehensive diagnostic analysis
  diagnostics_output_dir: "diagnostics_analysis"  # Directory for diagnostic plots
  # enable_diagnostics: false  # Disable for faster execution

  enable_lineage_tracking: true  # Enable genealogical analysis
  lineage_output_dir: "lineage_analysis"  # Directory for lineage plots
  # enable_lineage_tracking: false  # Disable for faster execution
```

### Diagnostics Features

When enabled, diagnostics provides comprehensive evolution analysis:

- **Fitness Evolution**: Best, average, and worst fitness progression with dual-shaded regions
- **Population Analysis**: Selection pressure metrics with clear definitions
- **Genetic Operations**: Success rates and operation effectiveness analysis
- **Performance Metrics**: Processing time trends and evolution efficiency
- **Diversity Management**: Population diversity tracking and convergence analysis
- **Evolution Grid**: Color-coded visualization of individual progression
- **Advanced Metrics**: Parameter adaptation trends and comprehensive diversity measures

**Output Files**:
- `dashboard.png` - Comprehensive overview with aligned statistics
- `fitness_evolution.png` - Dual-shaded fitness progression
- `genetic_operations.png` - Operation effectiveness analysis
- `performance_metrics.png` - Timing and efficiency analysis
- `population_analysis.png` - Selection pressure and diversity
- `evolution_grid.png` - Individual progression visualization
- `generation_data.csv` - Complete numerical data
- `diagnostics_data.json` - Structured metrics
- `summary.txt` - Human-readable analysis

### Lineage Tracking Features

When enabled, lineage tracking provides comprehensive genealogical analysis:

- **Family Trees**: Visual genealogy of individual lineages
- **Population Dynamics**: Birth/death rates, age distributions
- **Diversity Evolution**: Multiple diversity metrics over time
- **Migration Patterns**: Island model migration analysis (if enabled)
- **Survival Analysis**: Individual and lineage survival curves
- **Selection Pressure**: Detailed selection pressure metrics
- **Dominance Analysis**: Identification of dominant lineages

**Output Files**:
- `lineage_trees.png` - Genealogical family trees
- `population_dynamics.png` - Population size and turnover
- `diversity_evolution.png` - Comprehensive diversity metrics
- `selection_pressure.png` - Selection pressure analysis
- `lineage_dashboard.png` - Complete overview dashboard
- `individuals.json` - Raw genealogical data
- `lineage_summary.json` - Statistical summary

## Advanced Features

Higher-end presets (advanced, ultra, gpu, extreme) include:

- **Adaptive Parameters**: Dynamic adjustment based on population state
- **Advanced Crossover/Mutation**: Enhanced genetic operators
- **Comprehensive Diversity**: Prevention of premature convergence
- **Spatial Diversity**: Problem-specific diversity for image collages
- **Island Model**: Multi-population evolution with migration

## Performance Recommendations

- **Demo/Quick**: Any modern computer
- **Balanced/High**: 8+ GB RAM, multi-core CPU
- **Advanced/Ultra**: 16+ GB RAM, 8+ core CPU
- **GPU/Extreme**: High-end system, 32+ GB RAM, GPU acceleration