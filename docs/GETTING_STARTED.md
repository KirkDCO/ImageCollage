# Getting Started with Image Collage Generator

Welcome to the Image Collage Generator! This guide will take you from a complete beginner to an advanced user, progressively introducing the sophisticated features of this genetic algorithm-powered photomosaic creation tool.

## 📚 Table of Contents

1. [Quick Start (5 minutes)](#quick-start-5-minutes)
2. [Basic Usage (15 minutes)](#basic-usage-15-minutes)
3. [Understanding Presets (30 minutes)](#understanding-presets-30-minutes)
4. [Configuration & Customization (1 hour)](#configuration--customization-1-hour)
5. [Advanced Features (2 hours)](#advanced-features-2-hours)
6. [Expert Usage & Optimization (4+ hours)](#expert-usage--optimization-4-hours)

---

## Quick Start (5 minutes)

### Installation

```bash
# Basic installation
pip install -e .

# With GPU support (recommended if you have NVIDIA GPU)
pip install -e ".[gpu]"

# Complete installation with all features
pip install -e ".[gpu,visualization,dev]"
```

### Your First Collage (30 seconds)

```bash
# Ultra-fast demo - creates a collage in 30 seconds
image-collage demo target.jpg source_images/

# What this creates:
# demo_output_YYYYMMDD_HHMMSS/
# ├── collage.jpg        # Your photomosaic result
# ├── evolution.gif      # Animation showing how it evolved
# ├── comparison.jpg     # Target vs result side-by-side
# └── config.yaml        # Settings used for this run
```

**That's it!** You now have your first AI-generated photomosaic. Let's explore what just happened and how to do much more.

---

## Basic Usage (15 minutes)

### Understanding What Happened

The Image Collage Generator uses a **genetic algorithm** (inspired by biological evolution) to arrange your source images into a mosaic that visually resembles your target image.

**Key Concepts:**
- **Target Image**: The image you want to recreate as a mosaic
- **Source Images**: Your collection of photos that become the "tiles"
- **Grid**: The target is divided into a grid (e.g., 20x15 = 300 tiles)
- **Evolution**: The AI tries different arrangements over many "generations"
- **Fitness**: How well each arrangement matches the target image

### Basic Commands

```bash
# 1. Quick preview (2-5 minutes)
image-collage generate target.jpg sources/ output.png --preset quick

# 2. Balanced quality (5-15 minutes)
image-collage generate target.jpg sources/ output.png --preset balanced

# 3. High quality (15-45 minutes)
image-collage generate target.jpg sources/ output.png --preset high

# 4. With animation and comparison
image-collage generate target.jpg sources/ output.png \
  --preset balanced \
  --save-animation evolution.gif \
  --save-comparison comparison.jpg
```

### Understanding Output Structure

Every run creates a timestamped directory:
```
output_20250918_143022/
├── collage.jpg         # Final result
├── evolution.gif       # Evolution animation (if requested)
├── comparison.jpg      # Side-by-side comparison (if requested)
└── config.yaml         # Exact settings used
```

### Your Source Image Collection

**Organization Tips:**
```
my_photos/
├── family/
│   ├── vacations/
│   ├── birthdays/
│   └── holidays/
├── nature/
│   ├── landscapes/
│   └── animals/
└── misc/
```

The tool automatically finds ALL images in ALL subdirectories. More diverse photos = better results!

---

## Understanding Presets (30 minutes)

### Preset Overview

| Preset | Grid Size | Time | Best For |
|--------|-----------|------|----------|
| **demo** | 15×20 (300 tiles) | 30-60s | Testing, demonstrations |
| **quick** | 20×20 (400 tiles) | 2-5 min | Fast previews |
| **balanced** | 50×50 (2,500 tiles) | 5-15 min | Most use cases |
| **high** | 100×100 (10,000 tiles) | 15-45 min | High quality results |
| **advanced** | 60×60 (3,600 tiles) | 10-30 min | Anti-convergence evolution |
| **ultra** | 80×80 (6,400 tiles) | 30-60 min | All diversity techniques |
| **gpu** | 150×150 (22,500 tiles) | 10-30 min | GPU acceleration |
| **extreme** | 300×300 (90,000 tiles) | 1-4 hours | Ultimate quality |

### Trying Different Presets

```bash
# Compare different quality levels
image-collage generate portrait.jpg family_photos/ quick_result.png --preset quick
image-collage generate portrait.jpg family_photos/ balanced_result.png --preset balanced
image-collage generate portrait.jpg family_photos/ high_result.png --preset high

# Each preset optimizes different parameters:
# - Grid size (number of tiles)
# - Population size (how many arrangements the AI considers)
# - Generations (how long the AI evolves)
# - Fitness weights (what aspects it prioritizes)
```

### When to Use Each Preset

**Demo**: First time trying the tool, showing others how it works
**Quick**: Iterating on ideas, testing different source collections
**Balanced**: Most general-purpose collages, good quality/time balance
**High**: When you want impressive results and can wait
**GPU**: When you have a gaming/workstation GPU (RTX 3080+)
**Extreme**: For professional use, exhibitions, large prints

### Adding Diagnostics and Analysis

```bash
# See how the evolution works
image-collage generate target.jpg sources/ result.png \
  --preset balanced \
  --diagnostics analysis/

# Creates analysis/ folder with:
# - 10 visual reports showing evolution progress
# - 3 data files for deeper analysis
# - Graphs of fitness improvement over time
```

---

## Configuration & Customization (1 hour)

### Creating Custom Configurations

Instead of using presets, you can create custom YAML configuration files:

```yaml
# my_config.yaml
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [60, 60]           # 3,600 tiles
  tile_size: [32, 32]
  allow_duplicate_tiles: true
  enable_lineage_tracking: false
  lineage_output_dir: null
  enable_diagnostics: false     # Disable diagnostics for faster runs
  diagnostics_output_dir: null

# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  # Core GA Parameters
  population_size: 120          # AI considers 120 arrangements
  max_generations: 800          # Evolution time
  crossover_rate: 0.8           # How much to mix successful attempts
  mutation_rate: 0.08           # How much variation to try
  elitism_rate: 0.1
  tournament_size: 5
  # Termination Criteria
  convergence_threshold: 0.001
  early_stopping_patience: 40
  target_fitness: 0.0

# === FITNESS EVALUATION ===
fitness_evaluation:
  color_weight: 0.5             # Prioritize color matching
  luminance_weight: 0.3         # Secondary: brightness
  texture_weight: 0.15          # Tertiary: patterns
  edges_weight: 0.05            # Minimal: edge preservation

# === PERFORMANCE SETTINGS ===
performance:
  enable_parallel_processing: true
  num_processes: 8              # Use 8 CPU cores
```

```bash
# Use your custom configuration
image-collage generate target.jpg sources/ output.png --config example_configs/my_config.yaml
```

### Understanding Fitness Weights

The AI evaluates how good each arrangement is using four criteria:

**Color (usually 40-50%)**: Do the tile colors match the target?
**Luminance (usually 25-30%)**: Do the brightness levels match?
**Texture (usually 15-20%)**: Do the patterns and textures match?
**Edges (usually 10-15%)**: Are important edges preserved?

```yaml
# For portraits (prioritize skin tones and edges)
fitness_evaluation:
  color_weight: 0.6
  luminance_weight: 0.25
  texture_weight: 0.05
  edges_weight: 0.1

# For landscapes (prioritize textures and colors)
fitness_evaluation:
  color_weight: 0.45
  luminance_weight: 0.25
  texture_weight: 0.25
  edges_weight: 0.05
```

### Grid Size Considerations

**Small Grids (20×20 to 50×50)**:
- Faster processing
- More abstract/artistic results
- Good for testing and previews
- Requires fewer source images (100-500)

**Medium Grids (75×75 to 100×100)**:
- Good detail/performance balance
- Recognizable subjects
- Works with moderate source collections (500-1000)

**Large Grids (150×150+)**:
- High detail and accuracy
- Requires powerful hardware
- Needs large source collections (1000+ images)
- Professional quality results

### Export and Share Configurations

```bash
# Export any preset as a starting point
image-collage export-config my_balanced.yaml --preset balanced

# Edit the exported file, then use it
image-collage generate target.jpg sources/ output.png --config my_balanced.yaml
```

---

## Advanced Features (2 hours)

### Comprehensive Evolution Analysis

Enable detailed tracking of how the genetic algorithm works:

```bash
image-collage generate target.jpg sources/ advanced_result.png \
  --preset high \
  --diagnostics comprehensive_analysis/ \
  --track-lineage genealogy_analysis/
```

**This creates:**
- **10 diagnostic visualizations**: Fitness evolution, population dynamics, success rates
- **16 lineage analysis plots**: Family trees, inheritance patterns, breeding success
- **Complete data exports**: JSON and CSV files for custom analysis

### Real-time Monitoring

```bash
# Watch the evolution happen in real-time
image-collage generate target.jpg sources/ monitored_result.png \
  --preset gpu \
  --enable-dashboard \
  --verbose
```

The dashboard shows:
- Live diversity metrics
- Population health alerts
- Performance recommendations
- Intervention suggestions

### Advanced Diversity Management

```bash
# Enable sophisticated diversity preservation
image-collage generate target.jpg sources/ diverse_result.png \
  --preset high \
  --enable-fitness-sharing \
  --enable-restart \
  --track-components
```

**Features:**
- **Fitness Sharing**: Prevents the AI from getting stuck in local optima
- **Intelligent Restart**: ⚠️ **NOT FUNCTIONAL** - Configuration supported but system uses basic restart instead
- **Component Tracking**: ⚠️ **NOT FUNCTIONAL** - Feature exists but no visualizations are generated

### Crash Recovery System

```bash
# Enable automatic checkpoint saving
image-collage generate target.jpg sources/ safe_result.png \
  --preset extreme \
  --save-checkpoints

# If something crashes, resume from where it left off
image-collage resume output_20250918_143022/
```

### Multi-Population Evolution (Island Model) ⚠️ **NOT FUNCTIONAL**

**Status**: Island model configuration is supported but migration system is broken - 0 migration events occur despite being enabled. See TECH_DEBT.md for analysis.

```yaml
# example_configs/island_config.yaml (PRODUCES NO MIGRATIONS)
genetic_algorithm:
  enable_island_model: true    # ⚠️ Results in 0 migrations despite config
  island_model_num_islands: 3
  island_model_migration_interval: 15
  island_model_migration_rate: 0.2
```

**Expected**: Should run 3 separate populations that exchange best solutions, but migration system is currently non-functional.

### Color Tile Generation

Create geometric/abstract collages with pure colors. The color tile generator creates scientifically distributed colors across the RGB spectrum for maximum visual diversity.

#### Understanding Color Tile Generation

The system uses advanced color distribution algorithms:
- **Golden Ratio HSV Spacing**: Colors distributed using mathematical ratios for optimal perceptual spacing
- **RGB Spectrum Coverage**: Ensures complete coverage of red, green, and blue color space
- **Key Color Inclusion**: Always includes fundamental colors (black, white, primary, secondary)
- **Perceptual Uniformity**: Colors chosen for maximum visual distinctiveness

#### Basic Color Tile Commands

```bash
# Generate 100 diverse color tiles (quick start)
image-collage generate-color-tiles 100 my_colors/

# Generate with larger tile size and preview
image-collage generate-color-tiles 500 color_tiles/ \
  --tile-size 64 64 \
  --preview color_palette.png

# Generate with detailed analysis
image-collage generate-color-tiles 200 analyzed_colors/ \
  --tile-size 48 48 \
  --preview palette.png \
  --analyze

# This shows output like:
# 🎨 Generating 200 diverse color tiles...
# 📁 Output directory: analyzed_colors/
# 📐 Tile size: 48x48 pixels
# ✅ Successfully generated 200 color tiles in 0.45 seconds
# 🖼 Creating color palette preview...
# 📸 Preview saved to: palette.png
# 📊 Color Distribution Analysis:
#    Total colors: 200
#    RGB coverage: R=0.95, G=0.97, B=0.93
#    Average coverage: 0.95
#    Brightness range: 15 - 240
```

#### Creative Applications

**1. Pure Geometric Collages**
```bash
# Create pure geometric art
image-collage generate-color-tiles 300 geometric_colors/ --tile-size 32 32
image-collage generate portrait.jpg geometric_colors/ pure_geometric.png \
  --preset balanced \
  --grid-size 60 60 \
  --no-duplicates
```

**2. Mixed Photo-Color Collages**
```bash
# Combine photos with color tiles for artistic effects
mkdir mixed_collection
cp family_photos/*.jpg mixed_collection/
image-collage generate-color-tiles 150 temp_colors/
cp temp_colors/*.jpg mixed_collection/

image-collage generate target.jpg mixed_collection/ artistic_blend.png \
  --preset balanced \
  --grid-size 80 80
```

**3. Color Theory Education**
```bash
# Generate educational color palettes
image-collage generate-color-tiles 50 primary_colors/ \
  --preview primary_palette.png \
  --analyze

# Use for teaching color relationships
image-collage generate color_wheel.jpg primary_colors/ color_theory_demo.png \
  --preset quick \
  --grid-size 30 30
```

#### Advanced Color Tile Techniques

**Large-Scale Color Collections**
```bash
# Generate extensive color libraries (warning: takes time for 1000+)
image-collage generate-color-tiles 1000 extensive_colors/ \
  --tile-size 32 32 \
  --prefix "color_" \
  --preview extensive_palette.png \
  --analyze

# The system will warn and confirm for large collections:
# Warning: Generating a large number of tiles may take time
# Continue with 1000 tiles? [y/N]:
```

**Custom Tile Sizes for Different Uses**
```bash
# Small tiles for detailed mosaics
image-collage generate-color-tiles 400 small_tiles/ --tile-size 16 16

# Large tiles for abstract art
image-collage generate-color-tiles 100 large_tiles/ --tile-size 128 128

# Standard tiles for general use
image-collage generate-color-tiles 300 standard_tiles/ --tile-size 32 32
```

#### Color Analysis Features

When using `--analyze`, you get detailed statistics:
- **RGB Coverage**: How well each color channel is represented (0.0-1.0)
- **Average Coverage**: Overall color space coverage quality
- **Brightness Range**: Minimum to maximum brightness values
- **Color Count**: Total unique colors generated

**Example Analysis Output:**
```
📊 Color Distribution Analysis:
   Total colors: 500
   RGB coverage: R=0.98, G=0.96, B=0.97
   Average coverage: 0.97
   Brightness range: 12 - 243

💡 Usage example:
   image-collage generate target.jpg color_tiles/ output.png
```

#### Performance Notes

- **Generation Speed**: ~200-500 tiles per second on modern systems
- **Memory Usage**: Minimal - tiles generated on-demand
- **File Size**: 32x32 tiles ≈ 1-2KB each
- **Large Collections**: 1000+ tiles may prompt confirmation

---

## Expert Usage & Optimization (4+ hours)

### GPU Acceleration & Optimization

**Hardware Requirements:**
- NVIDIA GTX 1060+ (6GB VRAM minimum)
- RTX 3080+ recommended (12GB+ VRAM)
- Dual RTX 4090s optimal (48GB total VRAM)

```bash
# Basic GPU usage
image-collage generate target.jpg sources/ gpu_result.png --preset gpu

# Optimized for dual RTX 4090s
image-collage generate target.jpg sources/ ultimate.png \
  --preset extreme \
  --gpu \
  --gpu-devices "0,1" \
  --gpu-batch-size 4096
```

**GPU Optimization Strategy:**
1. Start with `--gpu-batch-size 1024`
2. Monitor GPU utilization with `nvidia-smi`
3. Increase batch size until utilization > 80%
4. Reduce if you get out-of-memory errors

### Advanced YAML Configuration

```yaml
# example_configs/expert_config.yaml - All features enabled
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [200, 200]
  tile_size: [32, 32]
  allow_duplicate_tiles: true
  enable_diagnostics: true
  diagnostics_output_dir: "complete_diagnostics"
  enable_lineage_tracking: true
  lineage_output_dir: "complete_lineage"

# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  # Core GA Parameters
  population_size: 300
  max_generations: 2000
  crossover_rate: 0.85
  mutation_rate: 0.08
  elitism_rate: 0.12
  tournament_size: 5
  # Termination Criteria
  convergence_threshold: 0.0005
  early_stopping_patience: 80
  target_fitness: 0.0
  # Advanced Evolution Features
  enable_adaptive_parameters: true
  enable_advanced_crossover: true
  enable_advanced_mutation: true
  enable_comprehensive_diversity: true
  enable_spatial_diversity: true
  # Multi-Population Island Model
  enable_island_model: true
  island_model_num_islands: 4
  island_model_migration_interval: 20
  island_model_migration_rate: 0.1

# === FITNESS EVALUATION ===
fitness_evaluation:
  color_weight: 0.4
  luminance_weight: 0.25
  texture_weight: 0.2
  edges_weight: 0.15

# === GPU ACCELERATION ===
gpu_acceleration:
  enable_gpu: true
  gpu_devices: [0, 1]
  gpu_batch_size: 4096
  auto_mixed_precision: true

# === PERFORMANCE SETTINGS ===
performance:
  enable_parallel_processing: true
  num_processes: 8
```

### Performance Analysis & Tuning

**Comprehensive Analysis Run:**
```bash
image-collage generate target.jpg sources/ analyzed_result.png \
  --config example_configs/expert_config.yaml \
  --save-animation complete_evolution.gif \
  --save-comparison detailed_comparison.jpg \
  --diagnostics full_diagnostics/ \
  --track-lineage complete_genealogy/ \
  --save-checkpoints \
  --enable-dashboard \
  --verbose
```

**Expected Outputs:**
- Final collage (40,000 tiles)
- Complete evolution animation
- 10 diagnostic visualization reports
- 16 lineage analysis plots
- Real-time dashboard logs
- Checkpoint files for recovery
- Complete data exports (JSON, CSV)

### Research & Experimentation

**Comparative Studies:**
```bash
# Test different evolutionary strategies
for preset in quick balanced high extreme; do
  image-collage generate target.jpg sources/ result_$preset.png \
    --preset $preset \
    --diagnostics analysis_$preset/ \
    --save-animation evolution_$preset.gif
done
```

**Parameter Sensitivity Analysis:**
```yaml
# Create multiple configs with varying parameters
mutation_rates: [0.05, 0.1, 0.15, 0.2]
population_sizes: [50, 100, 150, 200]
fitness_weights:
  - {color: 0.6, luminance: 0.3, texture: 0.1, edges: 0.0}
  - {color: 0.4, luminance: 0.4, texture: 0.1, edges: 0.1}
  - {color: 0.4, luminance: 0.2, texture: 0.3, edges: 0.1}
```

### Advanced Source Management

**Source Collection Strategies:**
```bash
# Analyze your source collection and target image
image-collage analyze target.jpg sources/

# Example output and recommendations:
# Target Image Analysis:
#   Resolution: 1920x1080 pixels
#   Aspect ratio: 1.778 (landscape)
# Source Collection Analysis:
#   Total images: 1,247
# Recommended grid sizes (aspect ratio 1.778):
#   36x20 (720 tiles, no duplicates, ✓ good aspect match)
#   71x40 (2,840 tiles, with duplicates, ✓ good aspect match)
# Recommendation: Use 'balanced' preset

# The analyze command provides:
# - Optimal grid size recommendations based on target aspect ratio
# - Preset recommendations based on source collection size
# - Duplicate handling suggestions
# - Aspect ratio matching analysis
```

**Optimal Collection Size:**
- **Small grids (≤50×50)**: 200-500 source images
- **Medium grids (75×100)**: 500-1000 source images
- **Large grids (150×200)**: 1000-2000 source images
- **Extreme grids (300×300)**: 2000+ source images

### Production Workflows

**Batch Processing:**
```bash
#!/bin/bash
# Process multiple targets with the same settings
for target in targets/*.jpg; do
  basename=$(basename "$target" .jpg)
  image-collage generate "$target" sources/ "results/${basename}_collage.png" \
    --preset high \
    --save-animation "results/${basename}_evolution.gif" \
    --diagnostics "analysis/${basename}/"
done
```

**Quality Assurance:**
```bash
# Comprehensive quality run with all analysis
image-collage generate target.jpg sources/ production_result.png \
  --config example_configs/expert_config.yaml \
  --save-animation production_evolution.gif \
  --save-comparison production_comparison.jpg \
  --diagnostics production_diagnostics/ \
  --track-lineage production_lineage/ \
  --save-checkpoints \
  --enable-dashboard \
  --enable-fitness-sharing \
  --enable-restart \
  --track-components \
  --verbose
```

---

## 🎯 Learning Path Summary

### Beginner (First Hour)
1. ✅ Install and run demo
2. ✅ Try different presets
3. ✅ Understand basic concepts
4. ✅ Organize source images

### Intermediate (Next 2-3 Hours)
5. ✅ Create custom configurations
6. ✅ Experiment with fitness weights
7. ✅ Enable diagnostics and analysis
8. ✅ Learn grid size trade-offs

### Advanced (Next 4-6 Hours)
9. ✅ Enable all tracking features
10. ✅ Set up GPU acceleration
11. ✅ Use crash recovery system
12. ✅ Analyze evolution patterns

### Expert (Ongoing)
13. ✅ Optimize performance for your hardware
14. ✅ Conduct parameter studies
15. ✅ Develop production workflows
16. ✅ Contribute to research and development

---

## 📚 Additional Resources

- **[README.md](README.md)**: Complete feature overview
- **[GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md)**: Detailed GPU tuning guide
- **[DIVERSITY.md](DIVERSITY.md)**: Advanced diversity management
- **[LINEAGE_TRACKING.md](LINEAGE_TRACKING.md)**: Genealogical analysis details
- **[example_configs/](example_configs/)**: All preset configurations
- **[TODO.md](TODO.md)**: Future enhancements and research directions

## 🆘 Getting Help

1. **Check the documentation** - Most questions are answered in the guides above
2. **Run with `--verbose`** - See detailed progress and debugging information
3. **Enable diagnostics** - Understand what the algorithm is doing
4. **Start small** - Use demo preset first, then scale up
5. **Monitor resources** - Check CPU, GPU, and memory usage

---

**Welcome to the world of AI-powered photomosaic generation!** 🎨🤖

Start with the 5-minute quick start, then progressively work through each section as you become more comfortable with the tool. Each level builds on the previous one, taking you from a complete beginner to an expert user capable of conducting sophisticated evolutionary algorithm research.