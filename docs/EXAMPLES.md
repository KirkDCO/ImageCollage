# Real-World Examples and Workflows

This guide provides practical examples and workflows for creating different types of collages with the Image Collage Generator.

## 📸 Photo Mosaic Workflows

### Family Portrait from Vacation Photos

**Scenario**: Create a family portrait mosaic using photos from your vacation collection.

**Directory Structure**:
```
family_vacation/
├── beach_photos/
│   ├── IMG_001.jpg
│   ├── IMG_002.jpg
│   └── ...
├── city_tours/
│   ├── landmarks/
│   └── restaurants/
├── hotel_photos/
└── group_photos/
```

**Workflow**:
```bash
# Step 1: Analyze your collection for optimal settings
image-collage analyze family_portrait.jpg family_vacation/

# This provides recommendations like:
# Target Image Analysis:
#   Resolution: 2048x1536 pixels
#   Aspect ratio: 1.333 (landscape)
# Source Collection Analysis:
#   Total images: 347
# Recommended grid sizes (aspect ratio 1.333):
#   27x20 (540 tiles, no duplicates, ✓ good aspect match)
#   53x40 (2,120 tiles, with duplicates, ✓ good aspect match)
# Recommendation: Use 'quick' preset

# Step 2: Quick preview
image-collage generate family_portrait.jpg family_vacation/ preview.png \
  --preset quick \
  --save-comparison preview_comparison.jpg

# Step 3: High-quality final version
image-collage generate family_portrait.jpg family_vacation/ family_mosaic.png \
  --preset high \
  --grid-size 80 80 \
  --no-duplicates \
  --save-animation family_evolution.gif \
  --save-comparison family_before_after.jpg \
  --diagnostics family_analysis/
```

**Expected Results**:
- 6,400-tile mosaic (80×80)
- Processing time: 15-30 minutes
- Uses each vacation photo only once
- Evolution animation showing the creation process

### Professional Headshot from Corporate Photos

**Scenario**: Create an executive headshot using company event photos and team pictures.

```bash
# Professional workflow
image-collage generate executive_photo.jpg corporate_photos/ executive_mosaic.png \
  --preset balanced \
  --grid-size 60 60 \
  --generations 1500 \
  --edge-blending \
  --format PNG \
  --save-comparison executive_comparison.jpg

# Fitness weights optimized for portraits
cat > portrait_config.yaml << EOF
# === FITNESS EVALUATION ===
fitness_evaluation:
  color_weight: 0.6      # Prioritize skin tones
  luminance_weight: 0.25 # Important for facial features
  texture_weight: 0.05   # Minimize texture artifacts
  edges_weight: 0.1      # Preserve facial edges
EOF

image-collage generate executive_photo.jpg corporate_photos/ executive_final.png \
  --config portrait_config.yaml \
  --grid-size 100 100
```

### Wedding Collage from Ceremony Photos

**Scenario**: Create a romantic wedding portrait using ceremony and reception photos.

```bash
# Wedding-specific settings
image-collage generate wedding_couple.jpg wedding_photos/ wedding_mosaic.png \
  --preset high \
  --grid-size 90 90 \
  --generations 2000 \
  --edge-blending \
  --no-duplicates \
  --save-animation wedding_creation.gif \
  --save-comparison wedding_before_after.jpg \
  --diagnostics wedding_analysis/ \
  --verbose

# Alternative: Soft, romantic style
cat > romantic_config.yaml << EOF
# === FITNESS EVALUATION ===
fitness_evaluation:
  color_weight: 0.5
  luminance_weight: 0.35
  texture_weight: 0.1
  edges_weight: 0.05

# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  mutation_rate: 0.03    # Gentler mutations
  population_size: 120
EOF

image-collage generate wedding_couple.jpg wedding_photos/ romantic_mosaic.png \
  --config romantic_config.yaml
```

## 🎨 Artistic and Creative Projects

### Color Tile Masterclass

**Scenario**: Explore the full potential of the color tile generation system for artistic projects.

#### Pure Color Spectrum Art

```bash
# Generate scientific color distribution
image-collage generate-color-tiles 500 spectrum_colors/ \
  --tile-size 64 64 \
  --preview spectrum_palette.png \
  --analyze

# Expected comprehensive analysis:
# 🎨 Generating 500 diverse color tiles...
# 📁 Output directory: spectrum_colors/
# 📐 Tile size: 64x64 pixels
# ✅ Successfully generated 500 color tiles in 1.12 seconds
# 🖼 Creating color palette preview...
# 📸 Preview saved to: spectrum_palette.png
# 📊 Color Distribution Analysis:
#    Total colors: 500
#    RGB coverage: R=0.98, G=0.97, B=0.99
#    Average coverage: 0.98
#    Brightness range: 12 - 248
# 💡 Usage example:
#    image-collage generate target.jpg spectrum_colors/ output.png

# Create rainbow spectrum portrait
image-collage generate face_portrait.jpg spectrum_colors/ rainbow_portrait.png \
  --preset balanced \
  --grid-size 80 80 \
  --no-duplicates \
  --save-animation rainbow_evolution.gif \
  --save-comparison rainbow_comparison.jpg
```

#### Educational Color Theory Demonstration

```bash
# Generate educational color sets
mkdir color_education

# Primary and secondary colors (small set)
image-collage generate-color-tiles 12 color_education/primaries/ \
  --tile-size 96 96 \
  --preview primary_palette.png \
  --analyze

# Full spectrum for education (medium set)
image-collage generate-color-tiles 100 color_education/full_spectrum/ \
  --tile-size 48 48 \
  --preview education_palette.png \
  --analyze

# Create educational demonstrations
image-collage generate color_wheel.jpg color_education/primaries/ primary_demo.png \
  --preset quick \
  --grid-size 20 20 \
  --no-duplicates

image-collage generate gradient_target.jpg color_education/full_spectrum/ spectrum_demo.png \
  --preset balanced \
  --grid-size 50 50 \
  --no-duplicates
```

#### Mixed Media Artistic Projects

```bash
# Combine photos and colors for artistic effect
mkdir artistic_blend

# Generate complementary color palette
image-collage generate-color-tiles 200 artistic_blend/colors/ \
  --tile-size 32 32 \
  --preview artistic_palette.png

# Copy selected photos
cp selected_art_photos/*.jpg artistic_blend/
cp artistic_blend/colors/*.jpg artistic_blend/

# Create artistic blend with 70% photos, 30% colors
image-collage generate artistic_target.jpg artistic_blend/ blended_art.png \
  --preset high \
  --grid-size 100 100 \
  --save-animation artistic_evolution.gif \
  --diagnostics artistic_analysis/

# Pure color version for comparison
image-collage generate artistic_target.jpg artistic_blend/colors/ pure_color_art.png \
  --preset balanced \
  --grid-size 80 80 \
  --no-duplicates
```

#### Large-Scale Color Installation

```bash
# Generate extensive color library for large installations
image-collage generate-color-tiles 2000 installation_colors/ \
  --tile-size 128 128 \
  --prefix "install_color_" \
  --preview installation_palette.png \
  --analyze

# The system will prompt:
# Warning: Generating a large number of tiles may take time
# Continue with 2000 tiles? [y/N]: y

# Expected output for large generation:
# 🎨 Generating 2000 diverse color tiles...
# 📁 Output directory: installation_colors/
# 📐 Tile size: 128x128 pixels
# ✅ Successfully generated 2000 color tiles in 4.23 seconds
# 🖼 Creating color palette preview...
# 📸 Preview saved to: installation_palette.png
# 📊 Color Distribution Analysis:
#    Total colors: 2000
#    RGB coverage: R=0.99, G=0.99, B=0.99
#    Average coverage: 0.99
#    Brightness range: 8 - 247

# Create ultra-high resolution artwork
image-collage generate masterpiece_target.jpg installation_colors/ installation_art.png \
  --preset extreme \
  --grid-size 200 200 \
  --no-duplicates \
  --save-checkpoints \
  --save-animation installation_evolution.gif \
  --diagnostics installation_analysis/ \
  --verbose
```

#### Custom Tile Sizes for Different Applications

```bash
# Micro-tiles for extreme detail
image-collage generate-color-tiles 800 micro_tiles/ \
  --tile-size 8 8 \
  --preview micro_palette.png

# Standard tiles for general use
image-collage generate-color-tiles 400 standard_tiles/ \
  --tile-size 32 32 \
  --preview standard_palette.png

# Macro-tiles for bold abstract art
image-collage generate-color-tiles 64 macro_tiles/ \
  --tile-size 256 256 \
  --preview macro_palette.png

# Create multi-scale comparison
image-collage generate comparison_target.jpg micro_tiles/ micro_result.png \
  --preset balanced --grid-size 100 100

image-collage generate comparison_target.jpg standard_tiles/ standard_result.png \
  --preset balanced --grid-size 50 50

image-collage generate comparison_target.jpg macro_tiles/ macro_result.png \
  --preset balanced --grid-size 25 25
```

## 🎨 Traditional Artistic and Creative Projects

### Abstract Art from Nature Photos

**Scenario**: Create abstract geometric art using nature photography.

```bash
# Generate diverse color tiles for abstract base
image-collage generate-color-tiles 300 abstract_colors/ \
  --tile-size 48 48 \
  --preview color_palette.png \
  --analyze

# Expected output:
# 🎨 Generating 300 diverse color tiles...
# ✅ Successfully generated 300 color tiles in 0.68 seconds
# 🖼 Creating color palette preview...
# 📸 Preview saved to: color_palette.png
# 📊 Color Distribution Analysis:
#    Total colors: 300
#    RGB coverage: R=0.96, G=0.98, B=0.94
#    Average coverage: 0.96
#    Brightness range: 18 - 237

# Combine with nature photos
mkdir mixed_sources
cp -r nature_photos/* mixed_sources/
cp abstract_colors/* mixed_sources/

# Create abstract collage
image-collage generate portrait.jpg mixed_sources/ abstract_art.png \
  --preset balanced \
  --grid-size 70 70 \
  --allow-duplicates \
  --save-animation abstract_evolution.gif

# Pure geometric version
image-collage generate portrait.jpg abstract_colors/ geometric_art.png \
  --preset balanced \
  --grid-size 60 60 \
  --no-duplicates
```

### Album Cover from Band Photos

**Scenario**: Create an album cover using live performance and studio photos.

```bash
# Music-focused workflow
image-collage generate album_concept.jpg band_photos/ album_cover.png \
  --preset high \
  --grid-size 100 100 \
  --edge-blending \
  --format PNG \
  --save-animation creation_process.gif

# High-contrast artistic style
cat > music_config.yaml << EOF
# === FITNESS EVALUATION ===
fitness_evaluation:
  color_weight: 0.45
  luminance_weight: 0.35
  texture_weight: 0.15
  edges_weight: 0.05

# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  population_size: 150
  mutation_rate: 0.08

# === OUTPUT SETTINGS ===
output:
  enable_edge_blending: true
EOF

image-collage generate album_concept.jpg band_photos/ artistic_cover.png \
  --config music_config.yaml \
  --grid-size 120 120
```

### Landscape Mosaic from Travel Photography

**Scenario**: Create a landscape masterpiece from travel photos across different locations.

```bash
# Landscape-optimized settings
cat > landscape_config.yaml << EOF
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [120, 80]    # Wide landscape format
  tile_size: [32, 32]
  allow_duplicate_tiles: true

# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  population_size: 150
  max_generations: 1800

# === FITNESS EVALUATION ===
fitness_evaluation:
  color_weight: 0.4
  luminance_weight: 0.25
  texture_weight: 0.25    # Important for landscapes
  edges_weight: 0.1
EOF

image-collage generate mountain_vista.jpg travel_photos/ landscape_mosaic.png \
  --config landscape_config.yaml \
  --save-animation landscape_evolution.gif \
  --save-comparison landscape_comparison.jpg \
  --diagnostics landscape_analysis/
```

## 🏢 Business and Commercial Applications

### Company Anniversary Poster

**Scenario**: Create a commemorative poster using 10 years of company photos.

```bash
# Organize company photos by year
company_photos/
├── 2014_founding/
├── 2015_first_office/
├── 2016_team_growth/
├── ...
└── 2024_anniversary/

# Professional poster creation
image-collage generate company_logo.jpg company_photos/ anniversary_poster.png \
  --preset high \
  --grid-size 150 100 \
  --generations 2000 \
  --no-duplicates \
  --edge-blending \
  --format PNG \
  --save-comparison anniversary_comparison.jpg

# Generate analysis report for presentation
image-collage generate company_logo.jpg company_photos/ anniversary_final.png \
  --preset high \
  --diagnostics anniversary_report/ \
  --save-animation company_evolution.gif \
  --verbose
```

### Product Showcase Collage

**Scenario**: Create a product hero image using customer photos and product shots.

```bash
# Product marketing workflow
mkdir product_showcase
cp customer_photos/*.jpg product_showcase/
cp product_shots/*.jpg product_showcase/
cp lifestyle_photos/*.jpg product_showcase/

image-collage generate hero_product.jpg product_showcase/ marketing_collage.png \
  --preset balanced \
  --grid-size 80 60 \
  --edge-blending \
  --format PNG \
  --save-comparison product_before_after.jpg

# High-resolution version for print
image-collage generate hero_product.jpg product_showcase/ print_version.png \
  --preset extreme \
  --grid-size 200 150 \
  --format TIFF \
  --save-checkpoints
```

### Event Poster from Conference Photos

**Scenario**: Create a conference poster using speaker and attendee photos.

```bash
# Conference collage workflow
image-collage generate keynote_speaker.jpg conference_photos/ conference_poster.png \
  --preset high \
  --grid-size 100 75 \
  --generations 1500 \
  --edge-blending \
  --save-animation conference_creation.gif \
  --save-comparison conference_comparison.jpg \
  --diagnostics conference_analysis/ \
  --verbose

# Generate multiple versions for different uses
# Social media version (square)
image-collage generate keynote_speaker.jpg conference_photos/ social_version.png \
  --preset balanced \
  --grid-size 60 60

# Website banner (wide)
image-collage generate keynote_speaker.jpg conference_photos/ banner_version.png \
  --preset balanced \
  --grid-size 120 40
```

## 🔬 Research and Educational Projects

### Scientific Visualization

**Scenario**: Create educational visualizations using microscopy or scientific images.

```bash
# Scientific image collection
science_images/
├── microscopy/
├── astronomy/
├── geological/
└── botanical/

# Educational visualization
image-collage generate diagram.jpg science_images/ scientific_mosaic.png \
  --preset high \
  --grid-size 100 100 \
  --no-duplicates \
  --diagnostics science_analysis/ \
  --save-animation educational_process.gif

# Generate detailed analysis for publication
image-collage generate diagram.jpg science_images/ publication_figure.png \
  --preset extreme \
  --grid-size 200 200 \
  --format TIFF \
  --save-checkpoints \
  --track-lineage scientific_genealogy/ \
  --diagnostics detailed_analysis/
```

### Historical Documentation

**Scenario**: Create a historical timeline mosaic using archival photographs.

```bash
# Historical photo organization
historical_photos/
├── 1900s/
├── 1910s/
├── 1920s/
├── ...
└── 2020s/

# Timeline visualization
image-collage generate historical_figure.jpg historical_photos/ timeline_mosaic.png \
  --preset high \
  --grid-size 150 100 \
  --generations 2500 \
  --no-duplicates \
  --save-animation historical_evolution.gif \
  --save-comparison timeline_comparison.jpg \
  --diagnostics historical_analysis/

# Create era-specific versions
for decade in 1920s 1950s 1980s; do
  image-collage generate historical_figure.jpg historical_photos/$decade/ era_${decade}.png \
    --preset balanced \
    --grid-size 60 60
done
```

## 🎮 Advanced Techniques and Experiments

### Multi-GPU Extreme Quality

**Scenario**: Create the highest possible quality using dual RTX 4090s.

```bash
# Ultimate quality workflow
image-collage generate masterpiece.jpg extensive_collection/ ultimate_result.png \
  --preset extreme \
  --gpu \
  --gpu-devices "0,1" \
  --gpu-batch-size 4096 \
  --grid-size 300 300 \
  --generations 5000 \
  --save-animation ultimate_evolution.gif \
  --save-comparison ultimate_comparison.jpg \
  --diagnostics ultimate_analysis/ \
  --track-lineage complete_genealogy/ \
  --save-checkpoints \
  --enable-dashboard \
  --enable-fitness-sharing \
  --enable-restart \
  --track-components \
  --verbose

# Expected: 90,000 tiles, 4-8 hours processing time
```

### Evolutionary Algorithm Research

**Scenario**: Conduct comparative studies of different genetic algorithm parameters.

```bash
# Research methodology
mkdir research_results

# Test different mutation rates
for rate in 0.03 0.06 0.09 0.12; do
  image-collage generate test_image.jpg sources/ mutation_${rate}.png \
    --preset balanced \
    --mutation-rate $rate \
    --diagnostics research_results/mutation_${rate}/ \
    --save-animation evolution_${rate}.gif \
    --seed 42  # Reproducible results
done

# Test population sizes
for pop in 50 100 150 200; do
  image-collage generate test_image.jpg sources/ population_${pop}.png \
    --preset balanced \
    --population-size $pop \
    --diagnostics research_results/population_${pop}/ \
    --seed 42
done

# Comparative analysis
python3 << EOF
import json
import pandas as pd
import matplotlib.pyplot as plt

# Collect results from all experiments
results = []
for experiment in ['mutation_0.03', 'mutation_0.06', 'population_50', 'population_100']:
    with open(f'research_results/{experiment}/diagnostics_data.json') as f:
        data = json.load(f)
        results.append({
            'experiment': experiment,
            'final_fitness': data['final_fitness'],
            'generations': data['generations_used'],
            'processing_time': data['processing_time']
        })

df = pd.DataFrame(results)
df.to_csv('research_results/comparative_analysis.csv')
print(df)
EOF
```

### Custom Fitness Function Experiment

**Scenario**: Test different fitness weight combinations for portrait optimization.

```bash
# Portrait optimization study
mkdir portrait_study

# Test different color weights
cat > high_color.yaml << EOF
# === FITNESS EVALUATION ===
fitness_evaluation:
  color_weight: 0.7
  luminance_weight: 0.2
  texture_weight: 0.05
  edges_weight: 0.05
EOF

cat > balanced_weights.yaml << EOF
# === FITNESS EVALUATION ===
fitness_evaluation:
  color_weight: 0.4
  luminance_weight: 0.3
  texture_weight: 0.2
  edges_weight: 0.1
EOF

cat > edge_focused.yaml << EOF
# === FITNESS EVALUATION ===
fitness_evaluation:
  color_weight: 0.4
  luminance_weight: 0.25
  texture_weight: 0.15
  edges_weight: 0.2
EOF

# Generate with different strategies
for config in high_color balanced_weights edge_focused; do
  image-collage generate portrait.jpg photos/ portrait_${config}.png \
    --config ${config}.yaml \
    --preset balanced \
    --seed 42 \
    --diagnostics portrait_study/${config}/ \
    --save-comparison ${config}_comparison.jpg
done
```

## 📊 Performance Optimization Examples

### Batch Processing Workflow

**Scenario**: Process multiple targets efficiently using the same source collection.

```bash
# Efficient batch processing
#!/bin/bash
SOURCE_DIR="company_photos"
CONFIG="example_configs/balanced.yaml"

# Pre-warm the cache by loading sources once
echo "Pre-warming source collection cache..."
image-collage analyze dummy_target.jpg $SOURCE_DIR

# Process multiple targets
for target in targets/*.jpg; do
  basename=$(basename "$target" .jpg)
  echo "Processing $basename..."

  image-collage generate "$target" $SOURCE_DIR "batch_results/${basename}_collage.png" \
    --config $CONFIG \
    --save-animation "batch_results/${basename}_evolution.gif" \
    --save-comparison "batch_results/${basename}_comparison.jpg" \
    --diagnostics "batch_analysis/${basename}/" \
    --parallel \
    --verbose
done

echo "Batch processing complete. Results in batch_results/"
```

### Memory-Efficient Large Collection Processing

**Scenario**: Handle very large source collections (10,000+ images) efficiently.

```bash
# Large collection optimization
image-collage generate large_target.jpg huge_collection/ efficient_result.png \
  --preset balanced \
  --max-source-images 2000 \  # Limit to prevent memory issues
  --cache-size-mb 2048 \       # Increase cache
  --parallel \
  --num-processes 6 \          # Match CPU cores
  --save-checkpoints \         # Enable recovery
  --verbose

# Alternative: Process in stages
# Stage 1: Quick preview with subset
image-collage generate large_target.jpg huge_collection/ preview.png \
  --preset quick \
  --max-source-images 500

# Stage 2: Full quality with optimized settings
image-collage generate large_target.jpg huge_collection/ final.png \
  --preset high \
  --max-source-images 2000 \
  --save-checkpoints
```

## 🎯 Quality Optimization Tips

### Source Collection Best Practices

```bash
# Analyze your collection before generation
image-collage analyze target.jpg sources/

# Optimal collection ratios:
# Grid 50x50 (2,500 tiles) → 1,000-2,500 source images
# Grid 100x100 (10,000 tiles) → 2,000-5,000 source images
# Grid 200x200 (40,000 tiles) → 5,000+ source images

# Quality check your sources
find sources/ -name "*.jpg" -exec identify {} \; | grep -E "1x1|corrupt"
```

### Progressive Quality Workflow

```bash
# Step 1: Demo for concept validation (30 seconds)
image-collage generate target.jpg sources/ concept.png --preset demo

# Step 2: Quick for composition testing (2-5 minutes)
image-collage generate target.jpg sources/ composition.png --preset quick

# Step 3: Balanced for final review (10-20 minutes)
image-collage generate target.jpg sources/ review.png --preset balanced

# Step 4: High quality for final output (30-60 minutes)
image-collage generate target.jpg sources/ final.png --preset high \
  --save-animation final_evolution.gif \
  --save-comparison final_comparison.jpg \
  --diagnostics final_analysis/
```

---

## 📚 Additional Resources

- **[GETTING_STARTED.md](GETTING_STARTED.md)**: Comprehensive beginner to expert guide
- **[GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md)**: GPU acceleration and optimization
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Common issues and solutions
- **[example_configs/](example_configs/)**: Ready-to-use configuration files

Each example includes estimated processing times, expected results, and optimization tips. Start with the demo preset to validate your approach, then scale up to higher quality settings once you're satisfied with the composition.