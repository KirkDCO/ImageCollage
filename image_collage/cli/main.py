import click
import json
import time
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any

from ..core.collage_generator import CollageGenerator, CollageResult
from ..config.settings import CollageConfig, PresetConfigs


@click.group()
@click.version_option()
def cli():
    """Image Collage Generator - Create photomosaic collages using genetic algorithms."""
    pass


@cli.command()
@click.argument('target_image', type=click.Path(exists=True, path_type=Path))
@click.argument('source_directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
# === BASIC SETTINGS ===
@click.option('--preset', type=click.Choice(['demo', 'quick', 'balanced', 'high', 'advanced', 'ultra', 'gpu', 'extreme']), default='balanced',
              help='Quality/speed preset configuration')
@click.option('--config', type=click.Path(exists=True), help='Custom configuration file (YAML/JSON)')
@click.option('--grid-size', type=(int, int), help='Grid dimensions (width height)')
@click.option('--generations', type=int, help='Maximum generations')
@click.option('--no-duplicates', is_flag=True, help='Prevent duplicate tiles')
@click.option('--seed', type=int, help='Random seed for reproducibility')
# === GENETIC ALGORITHM PARAMETERS ===
@click.option('--population-size', type=int, help='GA population size')
@click.option('--mutation-rate', type=float, help='Mutation rate (0.0-1.0)')
@click.option('--crossover-rate', type=float, help='Crossover rate (0.0-1.0)')
@click.option('--elitism-rate', type=float, help='Elite preservation rate (0.0-1.0)')
@click.option('--tournament-size', type=int, help='Tournament selection size')
@click.option('--convergence-threshold', type=float, help='Convergence threshold for stopping')
@click.option('--early-stopping-patience', type=int, help='Patience for early stopping')
# === GPU ACCELERATION ===
@click.option('--gpu/--no-gpu', default=False, help='Enable/disable GPU acceleration')
@click.option('--gpu-devices', type=str, help='GPU device IDs (comma-separated)')
@click.option('--gpu-batch-size', type=int, help='GPU batch size')
# === PERFORMANCE SETTINGS ===
@click.option('--parallel/--no-parallel', default=True, help='Enable/disable parallel processing')
@click.option('--processes', type=int, help='Number of parallel processes')
# === OUTPUT SETTINGS ===
@click.option('--format', type=click.Choice(['PNG', 'JPEG', 'TIFF']), default='PNG',
              help='Output image format')
@click.option('--quality', type=int, default=95, help='JPEG quality (1-100)')
@click.option('--edge-blending', is_flag=True, help='Enable edge blending between tiles')
@click.option('--save-animation', type=click.Path(), help='Save evolution animation as GIF')
@click.option('--save-comparison', type=click.Path(), help='Save target vs result comparison')
@click.option('--diagnostics', type=click.Path(), help='Save comprehensive diagnostics')
@click.option('--track-lineage', type=click.Path(), help='Track genealogy and save 16 lineage analysis plots')
@click.option('--save-checkpoints', is_flag=True, help='Save evolution checkpoints for resuming')
@click.option('--enable-fitness-sharing', is_flag=True, help='Enable fitness sharing for diversity')
@click.option('--enable-intelligent-restart', is_flag=True, help='Enable intelligent population restart')
@click.option('--enable-diversity-dashboard', is_flag=True, help='Enable real-time diversity dashboard')
@click.option('--track-components', is_flag=True, help='Track individual fitness components')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def generate(target_image: Path, source_directory: Path, output_path: Path,
             preset: str, config: Optional[str], grid_size: Optional[tuple],
             generations: Optional[int], population_size: Optional[int],
             mutation_rate: Optional[float], crossover_rate: Optional[float],
             elitism_rate: Optional[float], tournament_size: Optional[int],
             convergence_threshold: Optional[float], early_stopping_patience: Optional[int],
             no_duplicates: bool, edge_blending: bool, parallel: bool,
             processes: Optional[int], gpu: bool, gpu_devices: Optional[str],
             gpu_batch_size: Optional[int], format: str, quality: int,
             save_animation: Optional[Path], save_comparison: Optional[Path],
             diagnostics: Optional[Path], track_lineage: Optional[Path],
             save_checkpoints: bool, enable_fitness_sharing: bool,
             enable_intelligent_restart: bool, enable_diversity_dashboard: bool,
             track_components: bool, seed: Optional[int], verbose: bool):
    """Generate a photomosaic collage from a target image and source images.

    All outputs are automatically organized in timestamped directories (output_YYYYMMDD_HHMMSS/).
    Source images are recursively discovered in all subdirectories of SOURCE_DIRECTORY.

    \b
    Output Organization:
      output_20231201_143022/
      ├── collage.jpg                 # Main result
      ├── evolution.gif              # Animation with generation titles
      ├── comparison.jpg              # Target vs result comparison
      ├── config.yaml                # Configuration used
      └── diagnostics/                # 10 visual reports + 3 data files
          ├── dashboard.png           # Overview with proper alignment
          ├── fitness_evolution.png   # Dual-shaded fitness progression
          ├── genetic_operations.png  # Operation effectiveness analysis
          ├── performance_metrics.png # Accurate success rate metrics
          └── ... (additional reports)

    \b
    Examples:
      image-collage generate target.jpg sources/ demo.jpg --preset demo
      image-collage generate target.jpg sources/ result.jpg --preset balanced
      image-collage generate target.jpg sources/ result.jpg --preset ultra
    """

    if verbose:
        click.echo(f"Loading configuration preset: {preset}")
    
    collage_config = PresetConfigs.get_preset(preset)
    
    if config:
        if verbose:
            click.echo(f"Loading custom configuration from: {config}")
        try:
            from ..config.settings import CollageConfig
            custom_config_obj = CollageConfig.load_from_file(str(config))
            collage_config = custom_config_obj
        except Exception as e:
            click.echo(f"Error loading config file: {e}", err=True)
            return
    
    if grid_size:
        collage_config.grid_size = grid_size
    if generations:
        collage_config.genetic_params.max_generations = generations
    if population_size:
        collage_config.genetic_params.population_size = population_size
    if mutation_rate is not None:
        collage_config.genetic_params.mutation_rate = mutation_rate
    if crossover_rate is not None:
        collage_config.genetic_params.crossover_rate = crossover_rate
    if elitism_rate is not None:
        collage_config.genetic_params.elitism_rate = elitism_rate
    if tournament_size is not None:
        collage_config.genetic_params.tournament_size = tournament_size
    if convergence_threshold is not None:
        collage_config.genetic_params.convergence_threshold = convergence_threshold
    if early_stopping_patience is not None:
        collage_config.genetic_params.early_stopping_patience = early_stopping_patience
    if no_duplicates:
        collage_config.allow_duplicate_tiles = False
    if edge_blending:
        collage_config.enable_edge_blending = True
    if not parallel:
        collage_config.enable_parallel_processing = False
    if processes:
        collage_config.num_processes = processes
    if gpu:
        collage_config.gpu_config.enable_gpu = True
    if gpu_devices:
        device_list = [int(d.strip()) for d in gpu_devices.split(',')]
        collage_config.gpu_config.gpu_devices = device_list
    if gpu_batch_size:
        collage_config.gpu_config.gpu_batch_size = gpu_batch_size
    if quality:
        collage_config.output_quality = quality
    if seed is not None:
        collage_config.random_seed = seed
    if save_checkpoints:
        collage_config.enable_checkpoints = True
        collage_config.checkpoint_dir = "checkpoints"

        # Display crash recovery instructions
        click.echo("📦 Checkpoint saving enabled!")
        click.echo("   If the process crashes or is interrupted, you can resume with:")
        if verbose:
            click.echo(f"   image-collage resume {target_image} {source_directory} OUTPUT_DIRECTORY/")
        else:
            click.echo("   image-collage resume <target> <sources> <output_dir>/")
        click.echo("")

    # Configure advanced diversity features
    if enable_fitness_sharing:
        collage_config.enable_fitness_sharing = True
        if verbose:
            click.echo("🔄 Fitness sharing enabled for diversity preservation")

    if enable_intelligent_restart:
        collage_config.enable_intelligent_restart = True
        if verbose:
            click.echo("🔁 Intelligent restart system enabled")

    if enable_diversity_dashboard:
        collage_config.enable_diversity_dashboard = True
        if verbose:
            click.echo("📊 Real-time diversity dashboard enabled")

    if track_components:
        collage_config.enable_component_tracking = True
        if verbose:
            click.echo("🧬 Fitness component tracking enabled")

    # Create timestamped output directory for new run
    from datetime import datetime
    import os

    # Create timestamped output directory for new run
    datetime_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(f"output_{datetime_stamp}")

    # Create organized output paths
    organized_output_path = base_output_dir / output_path.name

    if save_animation:
        save_animation = base_output_dir / Path(save_animation).name
    if save_comparison:
        save_comparison = base_output_dir / Path(save_comparison).name
    # Handle diagnostics from both CLI argument and config file
    diagnostics_path = None
    if diagnostics:
        diagnostics_path = base_output_dir / Path(diagnostics).name
    elif collage_config.enable_diagnostics:
        if collage_config.diagnostics_output_dir:
            diagnostics_path = base_output_dir / collage_config.diagnostics_output_dir
        else:
            diagnostics_path = base_output_dir / "diagnostics"
    if track_lineage:
        track_lineage = base_output_dir / Path(track_lineage).name

    # Configure lineage tracking with organized path
    if track_lineage:
        collage_config.enable_lineage_tracking = True
        collage_config.lineage_output_dir = str(track_lineage)

    # Create the output directory
    base_output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        click.echo(f"Organized output directory: {base_output_dir}")

    # Create generator and load images using helper functions
    from .helpers import create_generator, load_images
    generator = create_generator(collage_config)

    click.echo("Loading target image...")
    click.echo("Loading source images...")
    if not load_images(generator, target_image, source_directory):
        return

    source_count = len(generator.source_images)
    
    click.echo(f"Loaded {source_count} source images")
    
    if verbose:
        click.echo(f"Grid size: {collage_config.grid_size}")
        click.echo(f"Population size: {collage_config.genetic_params.population_size}")
        click.echo(f"Max generations: {collage_config.genetic_params.max_generations}")
        if collage_config.gpu_config.enable_gpu:
            click.echo(f"GPU acceleration: Enabled on devices {collage_config.gpu_config.gpu_devices}")
        else:
            click.echo(f"GPU acceleration: Disabled")
        click.echo(f"Parallel processing: {collage_config.enable_parallel_processing}")
        if collage_config.enable_parallel_processing and not collage_config.gpu_config.enable_gpu:
            click.echo(f"CPU processes: {collage_config.num_processes}")
    
    click.echo("Starting collage generation...")
    start_time = time.time()
    
    try:
        # Create progress callback using helper function
        from .helpers import create_progress_callback
        callback = create_progress_callback(verbose, interval=collage_config.dashboard_update_interval)

        result = generator.generate(
            callback=callback,
            save_evolution=bool(save_animation),
            evolution_interval=max(1, collage_config.genetic_params.max_generations // 50),
            diagnostics_folder=str(diagnostics_path) if diagnostics_path else None,
            output_folder=str(base_output_dir),
            save_checkpoints=save_checkpoints,
            checkpoint_interval=collage_config.checkpoint_interval if save_checkpoints else 10
        )
        
        generation_time = time.time() - start_time
        
        click.echo(f"\nGeneration completed!")
        click.echo(f"Generations used: {result.generations_used}")
        click.echo(f"Final fitness: {result.fitness_score:.6f}")
        click.echo(f"Processing time: {generation_time:.2f} seconds")
        
        click.echo(f"Saving collage to: {organized_output_path}")
        if generator.export(result, str(organized_output_path), format):
            click.echo("Collage saved successfully!")
        else:
            click.echo("Error: Failed to save collage", err=True)
            return
        
        if save_animation and result.evolution_frames:
            click.echo(f"Creating evolution animation: {save_animation}")
            if generator.renderer.create_evolution_animation(
                result.evolution_frames,
                str(save_animation),
                generation_numbers=result.evolution_generation_numbers
            ):
                click.echo("Evolution animation saved successfully!")
            else:
                click.echo("Warning: Failed to create evolution animation", err=True)
            
        if save_comparison:
            click.echo(f"Creating comparison image: {save_comparison}")
            generator.renderer.create_comparison_image(
                generator.target_image, result.collage_image, str(save_comparison)
            )
        
    except KeyboardInterrupt:
        click.echo("\n⚠️  Generation interrupted by user")
        if save_checkpoints and base_output_dir:
            click.echo(f"💾 To resume from the last checkpoint, run:")
            click.echo(f"   image-collage resume {target_image} {source_directory} {base_output_dir}/")
    except Exception as e:
        from .helpers import handle_generation_error
        handle_generation_error(e, str(base_output_dir), save_checkpoints)
        import traceback
        if verbose:
            click.echo(f"Detailed error: {traceback.format_exc()}")


@cli.command()
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option('--preset', type=click.Choice(['demo', 'quick', 'balanced', 'high', 'advanced', 'ultra', 'gpu', 'extreme']), default='balanced')
def export_config(output_path: Path, preset: str):
    """Export configuration to a YAML file."""
    config = PresetConfigs.get_preset(preset)

    try:
        config.save_to_file(str(output_path))
        click.echo(f"Configuration exported to: {output_path}")
    except Exception as e:
        click.echo(f"Error exporting configuration: {e}", err=True)


@cli.command()
@click.argument('target_image', type=click.Path(exists=True, path_type=Path))
@click.argument('source_directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
def analyze(target_image: Path, source_directory: Path):
    """Analyze target image and source collection for optimization recommendations.
    
    Source images are automatically discovered recursively in all subdirectories."""
    
    # Create generator and load images using helper functions
    from .helpers import create_generator, load_images
    generator = create_generator()

    click.echo("Analyzing target image...")
    click.echo("Analyzing source images...")
    if not load_images(generator, target_image, source_directory):
        return

    source_count = len(generator.source_images)
    
    target_shape = generator.target_image.shape
    height, width = target_shape[:2]
    aspect_ratio = width / height

    click.echo(f"\nTarget Image Analysis:")
    click.echo(f"  Resolution: {width}x{height} pixels")
    click.echo(f"  Aspect ratio: {aspect_ratio:.3f} ({'landscape' if aspect_ratio > 1 else 'portrait' if aspect_ratio < 1 else 'square'})")

    click.echo(f"\nSource Collection Analysis:")
    click.echo(f"  Total images: {source_count}")

    # Generate aspect-ratio appropriate grid recommendations
    base_sizes = [20, 40, 60, 80, 100]
    grid_recommendations = []

    for base_size in base_sizes:
        if aspect_ratio > 1:  # Landscape
            grid_width = base_size
            grid_height = int(base_size / aspect_ratio)
        else:  # Portrait or square
            grid_height = base_size
            grid_width = int(base_size * aspect_ratio)

        # Ensure minimum size
        grid_width = max(grid_width, 10)
        grid_height = max(grid_height, 10)

        total_tiles = grid_width * grid_height
        aspect_match = abs((grid_width / grid_height) - aspect_ratio) < 0.1

        if source_count >= total_tiles:
            duplicates_note = "no duplicates"
        else:
            duplicates_note = "with duplicates"

        aspect_note = "✓ good aspect match" if aspect_match else "⚠ aspect mismatch"
        grid_recommendations.append(f"{grid_width}x{grid_height} ({total_tiles} tiles, {duplicates_note}, {aspect_note})")

    click.echo(f"\nRecommended grid sizes (aspect ratio {aspect_ratio:.3f}):")
    for rec in grid_recommendations:
        click.echo(f"  {rec}")
    
    if source_count < 400:
        click.echo("\nRecommendation: Use 'quick' preset for faster generation")
    elif source_count > 2000:
        click.echo("\nRecommendation: Use 'high' preset for best quality")
    else:
        click.echo("\nRecommendation: Use 'balanced' preset")


@cli.command()
@click.argument('target_image', type=click.Path(exists=True, path_type=Path))
@click.argument('source_directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--gpu/--no-gpu', default=False, help='Enable GPU acceleration for faster demo')
@click.option('--diagnostics/--no-diagnostics', default=False, help='Generate 10 visual reports + 3 data files')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed progress')
def demo(target_image: Path, source_directory: Path, gpu: bool, diagnostics: bool, verbose: bool):
    """Run a quick demonstration of the collage generator.

    Creates a fast, low-resolution collage with animation and comparison outputs
    to showcase all enhanced features. Completes in 30-60 seconds.

    Output files created in demo_output_YYYYMMDD_HHMMSS/:
      - collage.jpg (final result)
      - evolution.gif (animated evolution with generation numbers)
      - comparison.jpg (target vs result side-by-side)
      - config.yaml (settings used for this run)
      - diagnostics/ (10 visual reports + 3 data files with --diagnostics flag)
        ├── dashboard.png (comprehensive overview with proper alignment)
        ├── fitness_evolution.png (dual-shaded fitness progression)
        ├── genetic_operations.png (operation effectiveness with explanations)
        ├── performance_metrics.png (accurate success rate calculations)
        ├── population_analysis.png (selection pressure with definition)
        ├── evolution_grid.png (color-coded with explanations)
        └── ... (additional diagnostic reports)
    """
    from datetime import datetime
    import os

    # Create datetime-stamped output folder
    datetime_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"demo_output_{datetime_stamp}"
    os.makedirs(output_folder, exist_ok=True)

    click.echo("🎨 Running Image Collage Generator Demo")
    click.echo("=====================================")
    click.echo("📦 Checkpoints enabled - you can resume if interrupted:")
    click.echo(f"   image-collage resume {target_image} {source_directory} {output_folder}/")
    click.echo("")

    collage_output = os.path.join(output_folder, "collage.jpg")
    animation_output = os.path.join(output_folder, "evolution.gif")
    comparison_output = os.path.join(output_folder, "comparison.jpg")

    click.echo(f"Target: {target_image}")
    click.echo(f"Sources: {source_directory}")
    click.echo(f"GPU: {'Enabled' if gpu else 'Disabled'}")
    click.echo(f"Diagnostics: {'Enabled' if diagnostics else 'Disabled'}")
    click.echo(f"Output folder: {output_folder}/")
    click.echo("")

    # Call generate function directly instead of using CliRunner
    click.echo("Starting demo generation...")

    try:
        # Use the actual generation logic instead of calling the CLI command
        from ..core.collage_generator import CollageGenerator
        from ..config.settings import PresetConfigs

        # Load demo configuration
        collage_config = PresetConfigs.get_preset('demo')

        if gpu:
            collage_config.gpu_config.enable_gpu = True

        # Configure diagnostics
        if diagnostics:
            collage_config.enable_diagnostics = True
            collage_config.diagnostics_output_dir = "diagnostics"

        # Determine diagnostics output path
        diagnostics_output = None
        if diagnostics or collage_config.enable_diagnostics:
            if collage_config.diagnostics_output_dir:
                diagnostics_output = os.path.join(output_folder, collage_config.diagnostics_output_dir)
            else:
                diagnostics_output = os.path.join(output_folder, "diagnostics")

        # Create generator and load images using helper functions
        from .helpers import create_generator, load_images, create_progress_callback
        generator = create_generator(collage_config)

        if verbose:
            click.echo("Loading target image...")
            click.echo("Loading source images...")
        if not load_images(generator, target_image, source_directory):
            return

        source_count = len(generator.source_images)
        if verbose:
            click.echo(f"Loaded {source_count} source images")
            click.echo(f"Grid size: {collage_config.grid_size}")
            click.echo(f"Generations: {collage_config.genetic_params.max_generations}")

        # Create progress callback using helper function (more frequent for demo)
        demo_progress = create_progress_callback(verbose, interval=5)

        # Generate collage
        result = generator.generate(
            callback=demo_progress,
            save_evolution=True,
            evolution_interval=max(1, collage_config.genetic_params.max_generations // 10),
            diagnostics_folder=diagnostics_output,
            output_folder=output_folder,
            save_checkpoints=True,  # Always save checkpoints in demo
            checkpoint_interval=10
        )

        # Save outputs
        if not generator.export(result, collage_output, 'JPEG'):
            click.echo("Error: Failed to save collage", err=True)
            return

        # Save animation
        if result.evolution_frames:
            if generator.renderer.create_evolution_animation(
                result.evolution_frames,
                animation_output,
                generation_numbers=result.evolution_generation_numbers
            ):
                if verbose:
                    click.echo("Evolution animation saved!")
            else:
                click.echo("Warning: Failed to create animation", err=True)

        # Save comparison
        generator.renderer.create_comparison_image(
            generator.target_image, result.collage_image, comparison_output
        )

        click.echo("")
        click.echo("✅ Demo completed successfully!")
        click.echo(f"📁 Output folder: {output_folder}/")
        click.echo(f"📸 Collage: {os.path.basename(collage_output)}")
        click.echo(f"🎬 Animation: {os.path.basename(animation_output)}")
        click.echo(f"📊 Comparison: {os.path.basename(comparison_output)}")
        click.echo(f"⚙️  Config: config.yaml")
        if diagnostics:
            click.echo(f"📈 Diagnostics: {os.path.basename(diagnostics_output)}/")
            click.echo("   └─ Contains 10 visual reports + 3 data files")

    except KeyboardInterrupt:
        click.echo("\n⚠️  Demo interrupted by user")
        click.echo(f"💾 To resume from the last checkpoint, run:")
        click.echo(f"   image-collage resume {target_image} {source_directory} {output_folder}/")
    except Exception as e:
        click.echo("❌ Demo failed!")
        click.echo(f"Error: {e}")
        click.echo(f"💾 If checkpoints were saved, you can resume with:")
        click.echo(f"   image-collage resume {target_image} {source_directory} {output_folder}/")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())


@cli.command()
@click.argument('num_tiles', type=int)
@click.argument('output_directory', type=click.Path(path_type=Path))
@click.option('--tile-size', type=(int, int), default=(32, 32),
              help='Size of generated tiles (width height)')
@click.option('--prefix', default='color_tile',
              help='Filename prefix for generated tiles')
@click.option('--preview', type=click.Path(path_type=Path),
              help='Save color palette preview to this path')
@click.option('--analyze', is_flag=True,
              help='Show color distribution analysis')
def generate_color_tiles(num_tiles: int, output_directory: Path,
                        tile_size: tuple, prefix: str,
                        preview: Optional[Path], analyze: bool):
    """Generate a diverse set of single-color tiles spanning the RGB spectrum.

    Creates a collection of solid-color tiles with diverse colors distributed
    across the entire RGB color space. Useful for creating geometric collages
    or testing the algorithm with pure colors.

    NUM_TILES: Number of color tiles to generate (minimum 1)
    OUTPUT_DIRECTORY: Directory where tiles will be saved

    Examples:
      image-collage generate-color-tiles 100 my_colors/
      image-collage generate-color-tiles 500 tiles/ --tile-size 64 64 --preview palette.png
      image-collage generate-color-tiles 50 colors/ --analyze
    """
    from ..utils.color_tile_generator import ColorTileGenerator

    if num_tiles <= 0:
        click.echo("Error: Number of tiles must be greater than 0", err=True)
        return

    if num_tiles > 10000:
        click.echo("Warning: Generating a large number of tiles may take time")
        if not click.confirm(f"Continue with {num_tiles} tiles?"):
            return

    try:
        # Create generator
        generator = ColorTileGenerator(tile_size=tile_size)

        click.echo(f"🎨 Generating {num_tiles} diverse color tiles...")
        click.echo(f"📁 Output directory: {output_directory}")
        click.echo(f"📐 Tile size: {tile_size[0]}x{tile_size[1]} pixels")

        # Generate tiles
        start_time = time.time()
        generated_count = generator.generate_tiles(
            num_tiles=num_tiles,
            output_directory=str(output_directory),
            prefix=prefix
        )
        end_time = time.time()

        if generated_count == num_tiles:
            click.echo(f"✅ Successfully generated {generated_count} color tiles in {end_time - start_time:.2f} seconds")
        else:
            click.echo(f"⚠ Generated {generated_count} out of {num_tiles} requested tiles")

        # Generate preview if requested
        if preview:
            click.echo(f"🖼 Creating color palette preview...")
            generator.preview_color_palette(num_tiles, str(preview))
            click.echo(f"📸 Preview saved to: {preview}")

        # Show analysis if requested
        if analyze:
            colors = generator.generate_diverse_colors(num_tiles)
            stats = generator.analyze_color_distribution(colors)

            click.echo("\n📊 Color Distribution Analysis:")
            click.echo(f"   Total colors: {stats['count']}")
            click.echo(f"   RGB coverage: R={stats['rgb_coverage']['red']:.2f}, G={stats['rgb_coverage']['green']:.2f}, B={stats['rgb_coverage']['blue']:.2f}")
            click.echo(f"   Average coverage: {stats['average_coverage']:.2f}")
            click.echo(f"   Brightness range: {stats['brightness_range']['min']:.0f} - {stats['brightness_range']['max']:.0f}")

        # Show usage example
        click.echo(f"\n💡 Usage example:")
        click.echo(f"   image-collage generate target.jpg {output_directory}/ output.png")

    except Exception as e:
        click.echo(f"Error generating color tiles: {e}", err=True)
        import traceback
        click.echo(traceback.format_exc())


@cli.command()
def info():
    """Show information about the image collage generator."""
    click.echo("Image Collage Generator")
    click.echo("======================")
    click.echo("A genetic algorithm-based tool for creating photomosaic collages.")
    click.echo("")
    click.echo("Features:")
    click.echo("  • Multiple fitness metrics (color, luminance, texture, edges)")
    click.echo("  • Configurable genetic algorithm parameters")
    click.echo("  • GPU acceleration with CUDA support")
    click.echo("  • Parallel processing support")
    click.echo("  • Multiple output formats")
    click.echo("  • Preset configurations for different quality levels")
    click.echo("  • Progress monitoring and statistics")
    click.echo("  • Evolution animation with generation numbers")
    click.echo("  • Enhanced diagnostics with 8+ visual reports")
    click.echo("  • Diverse color tile generation for geometric collages")
    click.echo("")
    click.echo("Quick Start:")
    click.echo("  image-collage demo target.jpg source_folder/")
    click.echo("  image-collage demo target.jpg source_folder/ --diagnostics --gpu")
    click.echo("  image-collage generate --preset demo --diagnostics analysis/ target.jpg sources/ output.jpg")
    click.echo("  image-collage generate-color-tiles 100 color_tiles/ --preview palette.png")
    click.echo("")
    click.echo("For detailed usage instructions, use --help with any command.")


@cli.command()
@click.argument('target_image', type=click.Path(exists=True, path_type=Path))
@click.argument('source_directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output_directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--checkpoint', '-c', type=str, help='Specific checkpoint generation to resume from (default: latest)')
@click.option('--list-checkpoints', is_flag=True, help='List available checkpoints and exit')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def resume(target_image: Path, source_directory: Path, output_directory: Path,
           checkpoint: Optional[str], list_checkpoints: bool, verbose: bool):
    """Resume evolution from a crashed or interrupted run.

    Automatically finds the latest checkpoint in the output directory and
    resumes evolution from that point using the original configuration.

    \b
    Examples:
      # Resume from crashed run
      image-collage resume target.jpg source_images/ output_20231201_143022/

      # Resume with verbose output
      image-collage resume target.jpg source_images/ output_20231201_143022/ --verbose
    """
    from ..checkpoints import CheckpointManager
    from ..core.collage_generator import CollageGenerator
    from ..config.settings import CollageConfig

    if verbose:
        click.echo(f"Looking for checkpoints in: {output_directory}")

    # Find checkpoints directory
    checkpoint_dir = output_directory / "checkpoints"
    if not checkpoint_dir.exists():
        click.echo(f"Error: No checkpoints directory found in {output_directory}", err=True)
        click.echo("Make sure this is a valid output directory from a run with --save-checkpoints", err=True)
        return

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(str(checkpoint_dir))

    # List checkpoints and exit if requested
    if list_checkpoints:
        available_checkpoints = checkpoint_manager.list_checkpoints()
        if not available_checkpoints:
            click.echo(f"No checkpoints found in {checkpoint_dir}")
            return

        click.echo(f"Available checkpoints in {output_directory}:")
        click.echo("Generation | Timestamp           | Fitness   | File")
        click.echo("-----------|---------------------|-----------|" + "-" * 50)

        for cp in available_checkpoints:
            generation = cp.get('generation', 'N/A')
            timestamp = cp.get('timestamp', 'N/A')
            fitness = cp.get('best_fitness', 'N/A')
            filename = Path(cp.get('checkpoint_path', '')).name

            if fitness != 'N/A':
                fitness = f"{fitness:.6f}"

            click.echo(f"{generation:>10} | {timestamp:<19} | {fitness:<9} | {filename}")

        click.echo(f"\nTo resume from a specific checkpoint:")
        click.echo(f"  image-collage resume {target_image} {source_directory} {output_directory} --checkpoint <generation>")
        return

    # Find specific checkpoint or latest
    if checkpoint:
        # Find checkpoint by generation number
        available_checkpoints = checkpoint_manager.list_checkpoints()
        target_checkpoint = None

        try:
            target_generation = int(checkpoint)
            for cp in available_checkpoints:
                if cp.get('generation') == target_generation:
                    target_checkpoint = cp.get('checkpoint_path')
                    break

            if not target_checkpoint:
                click.echo(f"Error: No checkpoint found for generation {target_generation}", err=True)
                click.echo("Available generations:", err=True)
                for cp in available_checkpoints:
                    click.echo(f"  {cp.get('generation')}", err=True)
                return

        except ValueError:
            click.echo(f"Error: Invalid generation number '{checkpoint}'. Must be an integer.", err=True)
            return

        selected_checkpoint = target_checkpoint
        if verbose:
            click.echo(f"Using specified checkpoint: generation {target_generation}")
    else:
        # Find latest checkpoint
        selected_checkpoint = checkpoint_manager.find_latest_checkpoint()
        if not selected_checkpoint:
            click.echo(f"Error: No checkpoints found in {checkpoint_dir}", err=True)
            return

        if verbose:
            click.echo(f"Found latest checkpoint: {selected_checkpoint}")

    # Load checkpoint
    try:
        evolution_state, metadata = checkpoint_manager.load_checkpoint(selected_checkpoint)

        if verbose:
            click.echo(f"Loaded checkpoint from generation {evolution_state.generation}")
            click.echo(f"Best fitness: {evolution_state.best_fitness:.6f}")
            click.echo(f"Population size: {len(evolution_state.population)}")

    except Exception as e:
        click.echo(f"Error loading checkpoint: {e}", err=True)
        return

    # Look for original config.yaml in output directory
    config_file = output_directory / "config.yaml"
    if not config_file.exists():
        click.echo(f"Error: No config.yaml found in {output_directory}", err=True)
        click.echo("The original configuration file is required to resume", err=True)
        return

    if verbose:
        click.echo(f"Loading configuration from: {config_file}")

    # Load configuration
    try:
        config = CollageConfig.load_from_file(str(config_file))
        config.enable_checkpoints = True  # Ensure checkpoints remain enabled
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        return

    # Determine diagnostics folder based on config
    diagnostics_folder = None
    if config.enable_diagnostics:
        if config.diagnostics_output_dir:
            diagnostics_folder = output_directory / config.diagnostics_output_dir
        else:
            diagnostics_folder = output_directory / "diagnostics"

    # Create generator and load images using helper functions
    from .helpers import create_generator, load_images, create_progress_callback
    generator = create_generator(config)

    if verbose:
        click.echo(f"Loading target image: {target_image}")
        click.echo(f"Loading source images from: {source_directory}")

    if not load_images(generator, target_image, source_directory):
        return

    source_count = len(generator.source_images)

    if verbose:
        click.echo(f"Loaded {source_count} source images")
        click.echo(f"Resuming evolution from generation {evolution_state.generation}")
        click.echo(f"Estimated remaining generations: {config.genetic_params.max_generations - evolution_state.generation}")

    try:
        # Resume the evolution using the checkpoint file path
        result = generator.resume_from_checkpoint(
            checkpoint_path=selected_checkpoint,
            output_folder=str(output_directory),
            diagnostics_folder=str(diagnostics_folder) if diagnostics_folder else None,
            save_checkpoints=True,  # Continue saving checkpoints
            save_evolution=True,    # Continue saving evolution frames
            callback=create_progress_callback(verbose)
        )

        click.echo(f"Evolution resumed and completed!")
        click.echo(f"Final fitness: {result.fitness_score:.6f}")
        click.echo(f"Total generations: {result.generations_used}")
        click.echo(f"Results saved in: {output_directory}")

    except Exception as e:
        click.echo(f"Error during evolution: {e}", err=True)
        import traceback
        if verbose:
            click.echo(traceback.format_exc())
        return


def main():
    cli()


if __name__ == '__main__':
    main()