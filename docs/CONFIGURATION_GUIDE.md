# Advanced Configuration Guide

This guide provides in-depth coverage of all configuration options, advanced techniques, and optimization strategies for the Image Collage Generator.

## 📋 Configuration File Structure

### Complete Configuration Template

```yaml
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [100, 100]              # Grid dimensions [width, height]
  tile_size: [32, 32]                # Individual tile size in pixels
  allow_duplicate_tiles: true        # Allow same source image multiple times
  random_seed: 42                    # Seed for reproducible results (null = random)
  enable_lineage_tracking: false     # Enable genealogical analysis
  lineage_output_dir: "lineage"      # Directory for lineage analysis files
  enable_diagnostics: false          # Enable comprehensive diagnostics analysis
  diagnostics_output_dir: "diagnostics"  # Directory for diagnostic reports

# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  # Core Parameters
  population_size: 150               # Number of individuals in population
  max_generations: 1500              # Maximum evolution generations
  crossover_rate: 0.8                # Probability of crossover (0.0-1.0)
  mutation_rate: 0.05                # Probability of mutation (0.0-1.0)
  elitism_rate: 0.1                  # Fraction of elite individuals preserved (0.0-0.5)
  tournament_size: 5                 # Selection tournament size (2-20)
  # Termination Criteria
  convergence_threshold: 0.001       # Fitness improvement threshold for stopping
  early_stopping_patience: 50       # Generations without improvement before stopping
  target_fitness: 0.0                # Target fitness to reach (0.0 = evolve to completion)

  # Advanced Evolution Features
  enable_adaptive_parameters: true    # Dynamic parameter adjustment
  enable_advanced_crossover: true     # Enhanced crossover methods
  enable_advanced_mutation: true      # Advanced mutation strategies
  stagnation_threshold: 50           # Generations before considering stagnation
  diversity_threshold: 0.4           # Minimum diversity to maintain
  restart_threshold: 50              # Generations of stagnation before restart
  restart_ratio: 0.3                 # Fraction of population to replace on restart

  # Comprehensive Diversity Management
  enable_comprehensive_diversity: true # Enable advanced diversity preservation
  enable_spatial_diversity: true      # Enable spatial diversity for image problems

  # Multi-Population Island Model
  enable_island_model: false         # Enable island model evolution
  island_model_num_islands: 4        # Number of separate populations
  island_model_migration_interval: 20 # Generations between migrations
  island_model_migration_rate: 0.1   # Fraction of individuals migrating

# === FITNESS EVALUATION ===
fitness_evaluation:
  # Fitness component weights (must sum to 1.0)
  color_weight: 0.4                  # Color similarity importance
  luminance_weight: 0.25             # Brightness matching importance
  texture_weight: 0.2                # Texture correlation importance
  edges_weight: 0.15                 # Edge preservation importance

# === GPU ACCELERATION ===
gpu_acceleration:
  enable_gpu: false                  # Enable CUDA GPU acceleration
  gpu_devices: [0]                   # GPU device IDs to use
  gpu_batch_size: 256                # Batch size for GPU processing
  gpu_memory_limit_gb: 20.0          # GPU memory limit (GB)
  auto_mixed_precision: true         # Enable FP16 for memory efficiency

# === PERFORMANCE SETTINGS ===
performance:
  enable_parallel_processing: true   # Enable CPU multiprocessing
  num_processes: 4                   # Number of worker processes
  cache_size_mb: 1024                # Image cache size (MB)
  max_source_images: 10000           # Maximum source images to load
  preview_frequency: 10              # Generations between preview updates

# === OUTPUT SETTINGS ===
output:
  output_quality: 95                 # JPEG quality (1-100)
  enable_edge_blending: false        # Enable smooth tile transitions

# === CHECKPOINT SYSTEM ===
checkpoint_system:
  enable_checkpoints: false          # Enable crash recovery
  checkpoint_interval: 25            # Generations between checkpoints
  max_checkpoints: 5                 # Maximum checkpoint files to keep
  checkpoint_dir: "checkpoints"      # Checkpoint directory name

# === ADVANCED DIVERSITY MANAGEMENT ===
diversity_management:
  enable_fitness_sharing: false      # Enable fitness sharing
  fitness_sharing_radius: 5.0        # Sharing radius in fitness space
  fitness_sharing_alpha: 1.0         # Sharing function exponent
  enable_crowding_replacement: false # Enable crowding replacement
  crowding_factor: 2.0               # Crowding selection factor

# === INTELLIGENT RESTART SYSTEM ===
# ⚠️  WARNING: Intelligent restart system is not currently integrated with the GA engine
# ⚠️  Enabling this will silently fallback to basic restart system instead
# ⚠️  See TECH_DEBT.md Section 15 for integration status and debugging info
intelligent_restart:
  enable_intelligent_restart: false  # Enable automatic restart (NOT FUNCTIONAL - see warning above)
  restart_diversity_threshold: 0.1   # Diversity threshold for restart
  restart_stagnation_threshold: 30   # Stagnation threshold for restart
  restart_elite_preservation: 0.1    # Elite preservation fraction

# === DIVERSITY DASHBOARD ===
diversity_dashboard:
  enable_diversity_dashboard: false  # Enable real-time monitoring
  dashboard_update_interval: 10      # Update interval (generations)
  dashboard_alert_critical_diversity: 0.05  # Critical diversity alert
  dashboard_alert_low_diversity: 0.1        # Low diversity warning

# === COMPONENT TRACKING ===
component_tracking:
  enable_component_tracking: false   # Track fitness components
  track_component_inheritance: false # Track component inheritance
```

## 🎯 Configuration Strategies by Use Case

### Portrait and Face Mosaics

```yaml
# portrait_optimized.yaml
basic_settings:
  grid_size: [80, 100]              # Slightly taller for portraits

fitness_evaluation:
  color_weight: 0.6                 # Emphasize skin tones
  luminance_weight: 0.25            # Important for facial features
  texture_weight: 0.05              # Minimize texture artifacts
  edges_weight: 0.1                 # Preserve facial edges

genetic_algorithm:
  mutation_rate: 0.04               # Gentler mutations for portraits
  population_size: 120              # Larger population for precision
  tournament_size: 7                # More selective pressure

output:
  enable_edge_blending: true        # Smooth transitions for faces
```

### Landscape and Nature Scenes

```yaml
# landscape_optimized.yaml
basic_settings:
  grid_size: [120, 80]              # Wide landscape format

fitness_evaluation:
  color_weight: 0.35                # Balanced color importance
  luminance_weight: 0.25            # Sky/land contrast
  texture_weight: 0.3               # Important for landscapes
  edges_weight: 0.1                 # Natural edge preservation

genetic_algorithm:
  mutation_rate: 0.06               # More exploration for landscapes
  enable_spatial_diversity: true   # Important for terrain variety
```

### Abstract and Artistic Projects

```yaml
# artistic_abstract.yaml
basic_settings:
  grid_size: [90, 90]               # Square format
  allow_duplicate_tiles: false     # Maximum variety

fitness_evaluation:
  color_weight: 0.5                 # Primary focus on color
  luminance_weight: 0.3             # Secondary luminance
  texture_weight: 0.15              # Some texture interest
  edges_weight: 0.05                # Minimal edge constraint

genetic_algorithm:
  mutation_rate: 0.08               # Higher exploration
  enable_advanced_mutation: true   # Creative mutations
  diversity_threshold: 0.5          # Maintain high diversity
```

### High-Resolution Print Production

```yaml
# print_production.yaml
basic_settings:
  grid_size: [200, 200]             # High resolution
  max_generations: 3000             # Extended evolution
  convergence_threshold: 0.0005     # Tight convergence

genetic_algorithm:
  population_size: 200              # Large population
  elitism_rate: 0.15                # Preserve more elite
  enable_advanced_crossover: true  # Best crossover methods

gpu_acceleration:
  enable_gpu: true                  # Essential for large grids
  gpu_batch_size: 2048              # Large batches

checkpoint_system:
  enable_checkpoints: true          # Critical for long runs
  checkpoint_interval: 50           # Regular saves
```

### Fast Preview and Testing

```yaml
# fast_preview.yaml
basic_settings:
  grid_size: [30, 30]               # Small grid
  max_generations: 100              # Quick evolution
  early_stopping_patience: 20      # Stop early

genetic_algorithm:
  population_size: 50               # Small population

performance:
  preview_frequency: 5              # Frequent updates
  cache_size_mb: 512                # Smaller cache
```

## ⚙️ Parameter Optimization Guide

### Population Size Guidelines

| Grid Size | Recommended Population | Reasoning |
|-----------|----------------------|-----------|
| 20×20 (400 tiles) | 50-100 | Simple search space |
| 50×50 (2,500 tiles) | 100-150 | Balanced exploration |
| 100×100 (10,000 tiles) | 150-250 | Complex search space |
| 200×200 (40,000 tiles) | 200-400 | Very complex space |

### Mutation Rate Tuning

```yaml
# Conservative (portraits, detailed work)
genetic_algorithm:
  mutation_rate: 0.03               # Gentle changes

# Balanced (general use)
genetic_algorithm:
  mutation_rate: 0.05               # Standard rate

# Aggressive (abstract, creative)
genetic_algorithm:
  mutation_rate: 0.08               # More exploration

# Adaptive (let algorithm decide)
genetic_algorithm:
  enable_adaptive_parameters: true  # Dynamic adjustment
```

### Fitness Weight Optimization

#### Color-Dominant Approach
```yaml
fitness_evaluation:
  color_weight: 0.7                 # Strong color matching
  luminance_weight: 0.2             # Secondary brightness
  texture_weight: 0.05              # Minimal texture
  edges_weight: 0.05                # Minimal edges
```

#### Balanced Approach
```yaml
fitness_evaluation:
  color_weight: 0.4                 # Primary color
  luminance_weight: 0.25            # Secondary luminance
  texture_weight: 0.2               # Tertiary texture
  edges_weight: 0.15                # Quaternary edges
```

#### Structure-Preserving Approach
```yaml
fitness_evaluation:
  color_weight: 0.35                # Moderate color
  luminance_weight: 0.3             # Important structure
  texture_weight: 0.15              # Some texture
  edges_weight: 0.2                 # Strong edge preservation
```

## 🧬 Advanced Genetic Algorithm Features

### Island Model Configuration

```yaml
# Multi-population evolution with migration
genetic_algorithm:
  enable_island_model: true
  island_model_num_islands: 4       # 4 separate populations
  island_model_migration_interval: 20  # Migrate every 20 generations
  island_model_migration_rate: 0.15    # 15% migration rate

# Benefits:
# - Prevents premature convergence
# - Explores multiple solutions simultaneously
# - Better final results for complex problems
```

### Diversity Preservation

```yaml
# Comprehensive diversity management
genetic_algorithm:
  enable_comprehensive_diversity: true
  diversity_threshold: 0.4          # Maintain 40% diversity

diversity_management:
  enable_fitness_sharing: true      # Penalize similar solutions
  fitness_sharing_radius: 8.0       # Sharing neighborhood size
  enable_crowding_replacement: true # Replace similar individuals

# When to use:
# - Large search spaces (100×100+ grids)
# - Complex target images
# - When standard GA converges prematurely
```

### Intelligent Restart System

**⚠️ INTEGRATION ISSUE**: The intelligent restart system is not currently connected to the GA engine. While the configuration is supported and the IntelligentRestartManager system is fully implemented, enabling intelligent restart will silently use the basic restart system instead.

**Current Status**: See `TECH_DEBT.md` Section 15 for detailed analysis and `DEBUGGING.md` for integration procedures.

```yaml
# Automatic population restart (NOT CURRENTLY FUNCTIONAL - uses basic restart instead)
intelligent_restart:
  enable_intelligent_restart: true   # ⚠️ This flag is ignored by GA engine
  restart_diversity_threshold: 0.1  # ⚠️ These parameters are not used
  restart_stagnation_threshold: 50  # ⚠️ Basic restart parameters used instead
  restart_elite_preservation: 0.2   # ⚠️ See genetic_algorithm.restart_* parameters

# Benefits (when properly integrated):
# - Multi-criteria restart decisions (diversity + stagnation + convergence)
# - Adaptive threshold adjustment based on restart success
# - Elite-avoiding diverse individual generation
# - Comprehensive restart performance analysis
```

**Workaround**: Use the basic restart system which is functional:
```yaml
genetic_algorithm:
  restart_threshold: 40      # Restart after 40 stagnant generations
  restart_ratio: 0.4         # Replace 40% of population with random individuals
```

### Adaptive Parameters

```yaml
# Dynamic parameter adjustment
genetic_algorithm:
  enable_adaptive_parameters: true

# Automatically adjusts:
# - Mutation rate based on diversity
# - Selection pressure based on convergence
# - Population dynamics based on performance

# Algorithm behavior:
# High diversity → Lower mutation rate
# Low diversity → Higher mutation rate
# Fast convergence → Increase exploration
# Slow progress → Increase exploitation
```

## 🎮 GPU Optimization Configurations

### Single GPU Setup

```yaml
# Optimized for RTX 3080/4080 (12-16GB VRAM)
gpu_acceleration:
  enable_gpu: true
  gpu_devices: [0]
  gpu_batch_size: 1024              # Start here, adjust up
  gpu_memory_limit_gb: 14.0         # Leave 2GB for system
  auto_mixed_precision: true        # Enable FP16

basic_settings:
  grid_size: [100, 100]             # Good balance for single GPU

genetic_algorithm:
  population_size: 150              # Reasonable for GPU memory
```

### Dual GPU Setup

```yaml
# Optimized for dual RTX 4090 (48GB total VRAM)
gpu_acceleration:
  enable_gpu: true
  gpu_devices: [0, 1]               # Use both GPUs
  gpu_batch_size: 2048              # Large batches
  gpu_memory_limit_gb: 22.0         # Per GPU limit
  auto_mixed_precision: true

basic_settings:
  grid_size: [200, 200]             # Large grids possible

genetic_algorithm:
  population_size: 300              # Large populations
  max_generations: 3000             # Extended evolution
```

### Memory-Constrained GPU

```yaml
# For older/smaller GPUs (6-8GB VRAM)
gpu_acceleration:
  enable_gpu: true
  gpu_batch_size: 512               # Smaller batches
  gpu_memory_limit_gb: 6.0          # Conservative limit
  auto_mixed_precision: true        # Essential for memory

basic_settings:
  grid_size: [75, 75]               # Moderate grid size

performance:
  cache_size_mb: 512                # Smaller cache
```

## 📊 Performance Optimization Configurations

### CPU-Optimized (No GPU)

```yaml
# Maximum CPU performance
performance:
  enable_parallel_processing: true
  num_processes: 8                  # Match CPU cores
  cache_size_mb: 2048               # Large cache

genetic_algorithm:
  population_size: 100              # Reasonable for CPU

basic_settings:
  grid_size: [75, 75]               # CPU-appropriate size
  max_generations: 1200             # Reasonable evolution time
```

### Memory-Efficient Configuration

```yaml
# For systems with limited RAM
performance:
  cache_size_mb: 512                # Small cache
  max_source_images: 1000           # Limit sources
  num_processes: 2                  # Fewer processes

basic_settings:
  grid_size: [50, 50]               # Smaller grid

genetic_algorithm:
  population_size: 75               # Smaller population
```

### Speed-Optimized Configuration

```yaml
# Fastest possible generation
basic_settings:
  grid_size: [40, 40]               # Small grid
  max_generations: 200              # Quick evolution
  early_stopping_patience: 20      # Stop early

genetic_algorithm:
  population_size: 50               # Small population
  tournament_size: 3                # Fast selection

fitness_evaluation:
  color_weight: 0.8                 # Simplified fitness
  luminance_weight: 0.2
  texture_weight: 0.0               # Skip expensive texture
  edges_weight: 0.0                 # Skip expensive edges
```

## 🔬 Research and Analysis Configurations

### Comprehensive Analysis

```yaml
# Maximum analysis and tracking
basic_settings:
  enable_lineage_tracking: true     # Full genealogy
  lineage_output_dir: "complete_lineage"
  enable_diagnostics: true          # Comprehensive diagnostics analysis
  diagnostics_output_dir: "detailed_analysis"  # Diagnostic reports

diversity_dashboard:
  enable_diversity_dashboard: true  # Real-time monitoring
  dashboard_update_interval: 5      # Frequent updates

component_tracking:
  enable_component_tracking: true   # Track fitness components
  track_component_inheritance: true # Component genealogy

checkpoint_system:
  enable_checkpoints: true          # Crash recovery
  checkpoint_interval: 10           # Frequent saves

# Optional CLI overrides:
# --diagnostics custom_path/         # Override diagnostics_output_dir
# --track-lineage custom_lineage/    # Override lineage_output_dir
# --save-animation evolution.gif     # Add animation output
# --verbose                          # Show detailed progress
```

### Reproducible Research

```yaml
# Ensure reproducible results
basic_settings:
  random_seed: 42                   # Fixed seed

genetic_algorithm:
  # Fixed parameters (no adaptation)
  enable_adaptive_parameters: false
  population_size: 100              # Fixed population
  mutation_rate: 0.05               # Fixed rates
  crossover_rate: 0.8

# Benefits:
# - Identical results across runs
# - Valid for comparative studies
# - Reproducible publications
```

### Parameter Sensitivity Study

```yaml
# Template for parameter studies
basic_settings:
  random_seed: 42                   # Keep fixed for comparison
  max_generations: 500              # Shorter for multiple runs

genetic_algorithm:
  # Vary these parameters across experiments:
  population_size: 100              # Test: 50, 100, 150, 200
  mutation_rate: 0.05               # Test: 0.03, 0.05, 0.08, 0.1
  crossover_rate: 0.8               # Test: 0.6, 0.7, 0.8, 0.9

# Run multiple experiments:
# for rate in 0.03 0.05 0.08 0.1; do
#   sed "s/mutation_rate: 0.05/mutation_rate: $rate/" template.yaml > test_$rate.yaml
#   image-collage generate target.jpg sources/ result_$rate.png --config test_$rate.yaml
# done
```

## 🔧 Configuration Management

### Configuration Inheritance

```yaml
# Base configuration (base_config.yaml)
basic_settings:
  grid_size: [100, 100]
  max_generations: 1500

genetic_algorithm:
  population_size: 150
  mutation_rate: 0.05

# Specialized configurations can override specific sections
# portrait_config.yaml (inherits from base_config.yaml)
fitness_evaluation:
  color_weight: 0.6                 # Override for portraits
  luminance_weight: 0.25
  texture_weight: 0.05
  edges_weight: 0.1
```

### Environment-Specific Configurations

```yaml
# development.yaml (fast iteration)
basic_settings:
  grid_size: [30, 30]
  max_generations: 100
performance:
  preview_frequency: 5

# staging.yaml (quality validation)
basic_settings:
  grid_size: [75, 75]
  max_generations: 800
checkpoint_system:
  enable_checkpoints: true

# production.yaml (final output)
basic_settings:
  grid_size: [150, 150]
  max_generations: 2500
gpu_acceleration:
  enable_gpu: true
checkpoint_system:
  enable_checkpoints: true
```

### Configuration Validation

```bash
# Validate configuration syntax
python -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    print('Configuration is valid YAML')
"

# Test configuration with demo
image-collage generate target.jpg sources/ test.png \
  --config config.yaml \
  --preset demo \
  --verbose

# Export current configuration for reference
image-collage export-config current_settings.yaml --preset balanced
```

## 📈 Performance Benchmarking

### Configuration Comparison Template

```bash
# benchmark_configs.sh
#!/bin/bash

configs=("quick" "balanced" "high")
target="test_target.jpg"
sources="test_sources/"

for config in "${configs[@]}"; do
    echo "Benchmarking $config configuration..."

    time image-collage generate $target $sources "benchmark_${config}.png" \
        --preset $config \
        --diagnostics "benchmark_${config}_analysis/" \
        --verbose > "benchmark_${config}.log" 2>&1

    echo "Completed $config benchmark"
done

# Extract performance metrics
python benchmark_analysis.py
```

### Hardware-Specific Tuning

```yaml
# For 8-core CPU, 32GB RAM, RTX 4090
# high_end_system.yaml
basic_settings:
  grid_size: [150, 150]

genetic_algorithm:
  population_size: 250

gpu_acceleration:
  enable_gpu: true
  gpu_batch_size: 4096
  gpu_memory_limit_gb: 22.0

performance:
  num_processes: 8
  cache_size_mb: 4096

# For 4-core CPU, 16GB RAM, no GPU
# standard_system.yaml
basic_settings:
  grid_size: [75, 75]

genetic_algorithm:
  population_size: 100

performance:
  num_processes: 4
  cache_size_mb: 1024
  max_source_images: 2000
```

---

## 📚 Related Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)**: Basic usage and concepts
- **[GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md)**: Detailed GPU tuning
- **[EXAMPLES.md](EXAMPLES.md)**: Real-world configuration examples
- **[example_configs/](example_configs/)**: Ready-to-use configurations

This guide provides the foundation for creating optimal configurations for your specific use cases. Start with the provided examples and gradually customize parameters based on your hardware capabilities and quality requirements.