#!/usr/bin/env python3
"""
GPU-accelerated collage generation example.

This example demonstrates how to use GPU acceleration with dual RTX 4090s
for maximum performance and quality.
"""

import sys
import time
from pathlib import Path

from image_collage.core.collage_generator import CollageGenerator
from image_collage.config.settings import PresetConfigs, CollageConfig, GPUConfig


def gpu_progress_callback(generation: int, fitness: float, preview_image):
    """Enhanced progress callback for GPU generation."""
    if generation == 0:
        print(f"{'Generation':<10} {'Fitness':<15} {'Status'}")
        print("-" * 50)
    
    if generation % 100 == 0 or generation < 10:
        status = "GPU Processing..."
        if generation == 0:
            status = "GPU Initialized"
        print(f"{generation:<10} {fitness:<15.8f} {status}")


def demonstrate_gpu_presets():
    """Show different GPU preset configurations."""
    
    print("GPU Acceleration Presets:")
    print("=" * 50)
    
    # GPU preset
    gpu_config = PresetConfigs.gpu_accelerated()
    print(f"\nGPU Preset:")
    print(f"  Grid: {gpu_config.grid_size}")
    print(f"  Generations: {gpu_config.genetic_params.max_generations}")
    print(f"  Population: {gpu_config.genetic_params.population_size}")
    print(f"  GPU Devices: {gpu_config.gpu_config.gpu_devices}")
    print(f"  Batch Size: {gpu_config.gpu_config.gpu_batch_size}")
    
    # Extreme preset
    extreme_config = PresetConfigs.extreme_quality()
    print(f"\nExtreme Preset:")
    print(f"  Grid: {extreme_config.grid_size} ({extreme_config.grid_size[0] * extreme_config.grid_size[1]:,} tiles)")
    print(f"  Generations: {extreme_config.genetic_params.max_generations}")
    print(f"  Population: {extreme_config.genetic_params.population_size}")
    print(f"  GPU Devices: {extreme_config.gpu_config.gpu_devices}")
    print(f"  Batch Size: {extreme_config.gpu_config.gpu_batch_size}")


def create_custom_gpu_config():
    """Create a custom GPU configuration optimized for dual RTX 4090s."""
    
    config = CollageConfig()
    
    # Grid configuration for extreme detail
    config.grid_size = (400, 400)  # 160,000 tiles!
    config.tile_size = (24, 24)
    
    # Genetic algorithm parameters
    config.genetic_params.population_size = 1000
    config.genetic_params.max_generations = 10000
    config.genetic_params.mutation_rate = 0.03
    config.genetic_params.crossover_rate = 0.9
    config.genetic_params.elitism_rate = 0.2
    
    # GPU configuration for dual RTX 4090s
    config.gpu_config = GPUConfig(
        enable_gpu=True,
        gpu_devices=[0, 1],  # Use both GPUs
        gpu_batch_size=2048,  # Large batch for maximum GPU utilization
        gpu_memory_limit_gb=22.0,  # Leave some headroom
        auto_mixed_precision=True
    )
    
    # Quality settings
    config.genetic_params.convergence_threshold = 0.00001
    config.genetic_params.early_stopping_patience = 500
    config.allow_duplicate_tiles = False  # No duplicates for ultimate quality
    
    return config


def check_gpu_availability():
    """Check if GPU acceleration is available."""
    try:
        import cupy as cp
        
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"CUDA Devices Found: {device_count}")
        
        for i in range(device_count):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props['name'].decode('utf-8')
                memory_gb = props['totalGlobalMem'] / (1024**3)
                print(f"  GPU {i}: {name} ({memory_gb:.1f} GB)")
        
        return device_count > 0
        
    except ImportError:
        print("CuPy not installed. Install with: pip install cupy")
        return False
    except Exception as e:
        print(f"GPU initialization failed: {e}")
        return False


def main():
    """Main GPU acceleration demonstration."""
    
    if len(sys.argv) < 4:
        print("Usage: python gpu_accelerated.py <target_image> <source_dir> <output_path> [mode]")
        print("Modes: gpu, extreme, custom, demo")
        return 1
    
    target_image = sys.argv[1]
    source_directory = sys.argv[2]
    output_path = sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "gpu"
    
    print("Image Collage Generator - GPU Acceleration Demo")
    print("=" * 60)
    
    # Check GPU availability
    if not check_gpu_availability():
        print("GPU acceleration not available. Exiting.")
        return 1
    
    print()
    
    if mode == "demo":
        demonstrate_gpu_presets()
        return 0
    
    # Select configuration
    if mode == "gpu":
        config = PresetConfigs.gpu_accelerated()
        print("Using GPU Accelerated preset")
    elif mode == "extreme":
        config = PresetConfigs.extreme_quality()
        print("Using Extreme Quality preset")
    elif mode == "custom":
        config = create_custom_gpu_config()
        print("Using Custom GPU configuration")
    else:
        print(f"Unknown mode: {mode}")
        return 1
    
    print(f"Configuration:")
    print(f"  Grid size: {config.grid_size} ({config.grid_size[0] * config.grid_size[1]:,} tiles)")
    print(f"  Population: {config.genetic_params.population_size}")
    print(f"  Max generations: {config.genetic_params.max_generations}")
    print(f"  GPU devices: {config.gpu_config.gpu_devices}")
    print(f"  Batch size: {config.gpu_config.gpu_batch_size}")
    print()
    
    # Initialize generator
    print("Initializing GPU-accelerated generator...")
    generator = CollageGenerator(config)
    
    # Load images
    print("Loading target image...")
    if not generator.load_target(target_image):
        print("Error: Could not load target image")
        return 1
    
    print("Loading source images...")
    source_count = generator.load_sources(source_directory)
    print(f"Loaded {source_count:,} source images")
    
    if source_count == 0:
        print("Error: No source images found")
        return 1
    
    # Check if we have enough sources for no-duplicates mode
    total_tiles = config.grid_size[0] * config.grid_size[1]
    if source_count < total_tiles and not config.allow_duplicate_tiles:
        print(f"Warning: Need {total_tiles:,} tiles but only {source_count:,} sources available")
        print("Enabling duplicate tiles...")
        config.allow_duplicate_tiles = True
    
    print("\nStarting GPU-accelerated generation...")
    print("This may take some time for extreme quality settings...")
    
    start_time = time.time()
    
    try:
        result = generator.generate(callback=gpu_progress_callback)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nGPU generation completed!")
        print(f"Generations used: {result.generations_used:,}")
        print(f"Final fitness: {result.fitness_score:.8f}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Average time per generation: {elapsed_time/result.generations_used:.3f} seconds")
        
        # Show GPU memory stats if available
        if hasattr(generator.fitness_evaluator, 'get_gpu_memory_usage'):
            memory_stats = generator.fitness_evaluator.get_gpu_memory_usage()
            print(f"\nGPU Memory Usage:")
            for device_id, stats in memory_stats.items():
                print(f"  GPU {device_id}: {stats['used_gb']:.1f} GB / {stats['total_gb']:.1f} GB "
                      f"({stats['utilization']*100:.1f}%)")
        
        print(f"\nSaving ultra-high quality collage to: {output_path}")
        if generator.export(result, output_path, "PNG"):
            print("GPU-accelerated collage saved successfully!")
            
            # Create comparison image
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
        print(f"Error during GPU generation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())