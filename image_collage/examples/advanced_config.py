#!/usr/bin/env python3
"""
Advanced configuration example for the Image Collage Generator.

This example shows how to create custom configurations and use advanced features.
"""

import json
import sys
import time
from pathlib import Path

from image_collage.core.collage_generator import CollageGenerator
from image_collage.config.settings import CollageConfig, FitnessWeights, GeneticParams


def create_portrait_config():
    """Create a configuration optimized for portrait images."""
    config = CollageConfig()
    
    config.grid_size = (60, 80)
    config.tile_size = (24, 24)
    
    config.fitness_weights = FitnessWeights(
        color=0.5,      # Higher emphasis on color matching
        luminance=0.3,  # Important for facial features
        texture=0.1,    # Less important for portraits
        edges=0.1       # Minimal edge emphasis
    )
    
    config.genetic_params = GeneticParams(
        population_size=120,
        max_generations=800,
        crossover_rate=0.85,
        mutation_rate=0.04,
        elitism_rate=0.15,  # Preserve more good solutions
        tournament_size=7
    )
    
    config.enable_edge_blending = True
    config.allow_duplicate_tiles = False
    
    return config


def create_landscape_config():
    """Create a configuration optimized for landscape images."""
    config = CollageConfig()
    
    config.grid_size = (80, 60)
    config.tile_size = (32, 32)
    
    config.fitness_weights = FitnessWeights(
        color=0.35,
        luminance=0.25,
        texture=0.25,   # More important for landscapes
        edges=0.15      # Edge details matter
    )
    
    config.genetic_params = GeneticParams(
        population_size=100,
        max_generations=1000,
        crossover_rate=0.8,
        mutation_rate=0.06,
        elitism_rate=0.1,
        tournament_size=5
    )
    
    return config


def create_artistic_config():
    """Create a configuration for artistic/abstract results."""
    config = CollageConfig()
    
    config.grid_size = (50, 50)
    config.tile_size = (40, 40)
    
    config.fitness_weights = FitnessWeights(
        color=0.7,      # Heavy color emphasis
        luminance=0.2,
        texture=0.05,
        edges=0.05
    )
    
    config.genetic_params = GeneticParams(
        population_size=80,
        max_generations=600,
        crossover_rate=0.9,
        mutation_rate=0.08,  # Higher mutation for variety
        elitism_rate=0.05,   # Less elitism for exploration
        tournament_size=3
    )
    
    config.allow_duplicate_tiles = True
    
    return config


def detailed_progress_callback(generation: int, fitness: float, preview_image):
    """Enhanced progress callback with detailed statistics."""
    if generation == 0:
        print(f"{'Gen':<6} {'Fitness':<12} {'Time':<8} {'Status'}")
        print("-" * 40)
    
    if generation % 25 == 0:
        current_time = time.strftime("%H:%M:%S")
        status = "Evolving..."
        if generation == 0:
            status = "Starting"
        
        print(f"{generation:<6} {fitness:<12.6f} {current_time:<8} {status}")


def save_config_examples():
    """Save example configurations to JSON files."""
    configs = {
        "portrait": create_portrait_config(),
        "landscape": create_landscape_config(),
        "artistic": create_artistic_config()
    }
    
    for name, config in configs.items():
        filename = f"examples/config_{name}.json"
        Path("examples").mkdir(exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        print(f"Saved {name} configuration to: {filename}")


def main():
    """Main example function demonstrating advanced configurations."""
    
    if len(sys.argv) < 4:
        print("Usage: python advanced_config.py <target_image> <source_dir> <output_path> [config_type]")
        print("Config types: portrait, landscape, artistic")
        return 1
    
    target_image = sys.argv[1]
    source_directory = sys.argv[2]
    output_path = sys.argv[3]
    config_type = sys.argv[4] if len(sys.argv) > 4 else "portrait"
    
    save_config_examples()
    
    print(f"Image Collage Generator - Advanced Configuration Example")
    print(f"Configuration type: {config_type}")
    print("=" * 60)
    
    config_functions = {
        "portrait": create_portrait_config,
        "landscape": create_landscape_config,
        "artistic": create_artistic_config
    }
    
    if config_type not in config_functions:
        print(f"Unknown configuration type: {config_type}")
        return 1
    
    config = config_functions[config_type]()
    
    print(f"Configuration details:")
    print(f"  Grid size: {config.grid_size}")
    print(f"  Tile size: {config.tile_size}")
    print(f"  Population: {config.genetic_params.population_size}")
    print(f"  Max generations: {config.genetic_params.max_generations}")
    print(f"  Fitness weights: Color={config.fitness_weights.color}, "
          f"Luminance={config.fitness_weights.luminance}, "
          f"Texture={config.fitness_weights.texture}, "
          f"Edges={config.fitness_weights.edges}")
    print()
    
    generator = CollageGenerator(config)
    
    print("Loading target image...")
    if not generator.load_target(target_image):
        print("Error: Could not load target image")
        return 1
    
    print("Loading source images...")
    source_count = generator.load_sources(source_directory)
    print(f"Loaded {source_count} source images")
    
    if source_count == 0:
        print("Error: No source images found")
        return 1
    
    grid_tiles = config.grid_size[0] * config.grid_size[1]
    if source_count < grid_tiles and not config.allow_duplicate_tiles:
        print(f"Warning: {grid_tiles} tiles needed but only {source_count} sources available")
        print("Consider enabling duplicate tiles or reducing grid size")
    
    print("\nStarting advanced collage generation...")
    start_time = time.time()
    
    try:
        result = generator.generate(callback=detailed_progress_callback)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nGeneration completed!")
        print(f"Generations used: {result.generations_used}")
        print(f"Final fitness: {result.fitness_score:.6f}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Average time per generation: {elapsed_time/result.generations_used:.3f} seconds")
        
        print(f"\nSaving collage to: {output_path}")
        if generator.export(result, output_path, "PNG"):
            print("Success! Advanced collage saved.")
            
            comparison_path = output_path.replace('.png', '_comparison.png')
            if generator.renderer.create_comparison_image(
                generator.target_image, result.collage_image, comparison_path
            ):
                print(f"Comparison image saved to: {comparison_path}")
                
        else:
            print("Error: Failed to save collage")
            return 1
            
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during generation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())