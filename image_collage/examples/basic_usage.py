#!/usr/bin/env python3
"""
Basic usage example for the Image Collage Generator.

This example demonstrates the core functionality of creating a photomosaic
collage using the genetic algorithm-based approach.
"""

import sys
import time
from pathlib import Path

from image_collage.core.collage_generator import CollageGenerator
from image_collage.config.settings import PresetConfigs


def progress_callback(generation: int, fitness: float, preview_image):
    """Callback function to display generation progress."""
    if generation % 50 == 0 or generation < 10:
        print(f"Generation {generation:4d}: Fitness = {fitness:.6f}")


def main():
    """Main example function."""
    
    target_image_path = "examples/target_image.jpg"
    source_directory = "examples/source_images/"
    output_path = "examples/output_collage.png"
    
    if not Path(target_image_path).exists():
        print(f"Please place a target image at: {target_image_path}")
        return 1
    
    if not Path(source_directory).exists():
        print(f"Please create source images directory at: {source_directory}")
        return 1
    
    print("Image Collage Generator - Basic Usage Example")
    print("=" * 50)
    
    config = PresetConfigs.balanced()
    config.grid_size = (40, 40)
    config.genetic_params.max_generations = 300
    config.genetic_params.population_size = 80
    
    print(f"Configuration:")
    print(f"  Grid size: {config.grid_size}")
    print(f"  Population: {config.genetic_params.population_size}")
    print(f"  Max generations: {config.genetic_params.max_generations}")
    print()
    
    generator = CollageGenerator(config)
    
    print("Loading target image...")
    if not generator.load_target(target_image_path):
        print("Error: Could not load target image")
        return 1
    
    print("Loading source images...")
    source_count = generator.load_sources(source_directory)
    print(f"Loaded {source_count} source images")
    
    if source_count == 0:
        print("Error: No source images found")
        return 1
    
    print("\nStarting collage generation...")
    start_time = time.time()
    
    try:
        result = generator.generate(callback=progress_callback)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nGeneration completed!")
        print(f"Generations used: {result.generations_used}")
        print(f"Final fitness: {result.fitness_score:.6f}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        
        print(f"\nSaving collage to: {output_path}")
        if generator.export(result, output_path, "PNG"):
            print("Success! Collage saved.")
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