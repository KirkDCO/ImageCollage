# GPU Optimization Guide

This guide provides comprehensive information about GPU acceleration in the Image Collage Generator, including optimization strategies, hardware recommendations, and troubleshooting tips.

## Overview

The Image Collage Generator supports CUDA GPU acceleration through CuPy, providing massive performance improvements (1000x+ speedup) for large-scale collage generation. GPU acceleration is particularly effective for:

- Large grid sizes (100x100 tiles and above)
- High population sizes (200+ individuals)
- Complex fitness evaluations
- Multiple generations (1000+ generations)

## Hardware Requirements

### Minimum Requirements
- **NVIDIA GPU**: GTX 1060 or better with 6GB+ VRAM
- **CUDA**: Version 11.0 or higher
- **Memory**: 8GB system RAM + 6GB GPU VRAM
- **Storage**: SSD recommended for image I/O

### Recommended Configuration
- **NVIDIA GPU**: RTX 3080/4080 or better with 12GB+ VRAM
- **CUDA**: Version 12.0 or higher
- **Memory**: 16GB system RAM + 12GB GPU VRAM
- **Multi-GPU**: Dual RTX 4090s for extreme workloads

### Optimal Configuration (Dual GPU)
- **Primary GPU**: RTX 4090 (24GB VRAM)
- **Secondary GPU**: RTX 4090 (24GB VRAM)
- **Total VRAM**: 48GB for massive collages
- **System RAM**: 32GB+ for large source collections

## Installation

### Basic GPU Support
```bash
# Install CUDA toolkit (if not already installed)
# Visit: https://developer.nvidia.com/cuda-downloads

# Install CuPy for your CUDA version
pip install cupy-cuda12x  # For CUDA 12.x
# or
pip install cupy-cuda11x  # For CUDA 11.x

# Install Image Collage Generator with GPU support
pip install -e ".[gpu]"
```

### Verify Installation
```bash
# Test GPU availability
python -c "import cupy; print(f'GPU detected: {cupy.cuda.get_device_count()} devices')"

# Test with Image Collage Generator
image-collage generate target.jpg sources/ test.png --preset demo --gpu --verbose
```

## GPU Configuration

### Basic GPU Usage
```bash
# Enable GPU acceleration
image-collage generate target.jpg sources/ output.png --gpu

# Specify GPU devices (for multi-GPU systems)
image-collage generate target.jpg sources/ output.png --gpu --gpu-devices "0,1"

# Set GPU batch size
image-collage generate target.jpg sources/ output.png --gpu --gpu-batch-size 1024
```

### YAML Configuration
```yaml
# === GPU ACCELERATION ===
gpu_acceleration:
  enable_gpu: true
  gpu_devices: [0, 1]          # Use GPUs 0 and 1
  gpu_batch_size: 1024         # Process 1024 items per batch
  gpu_memory_limit_gb: 16.0    # Limit GPU memory usage
  auto_mixed_precision: true   # Enable FP16 for memory efficiency
```

## Optimization Strategies

### 1. Batch Size Optimization

**Problem**: Low GPU utilization (7% usage observed)
**Solution**: Increase batch size to fully utilize GPU cores

```yaml
# === GPU ACCELERATION ===
gpu_acceleration:
  gpu_batch_size: 2048    # Start with 2048
  # Increase gradually: 1024 → 2048 → 4096 → 8192
```

**Optimization Process**:
1. Start with `gpu_batch_size: 1024`
2. Monitor GPU utilization with `nvidia-smi`
3. Increase batch size until GPU utilization > 80%
4. If you get OOM errors, reduce batch size slightly

### 2. Memory Management

**VRAM Usage Guidelines**:
- **8GB VRAM**: `gpu_batch_size: 512`, grid up to 75x75
- **12GB VRAM**: `gpu_batch_size: 1024`, grid up to 100x100
- **16GB VRAM**: `gpu_batch_size: 2048`, grid up to 150x150
- **24GB VRAM**: `gpu_batch_size: 4096`, grid up to 200x200

**Memory Optimization**:
```yaml
# === GPU ACCELERATION ===
gpu_acceleration:
  gpu_memory_limit_gb: 14.0        # Leave 2GB for system
  auto_mixed_precision: true       # Use FP16 to halve memory usage
  enable_memory_pool: true         # Reuse allocated memory
```

### 3. Multi-GPU Strategies

**Population Splitting Strategy**:
- Population is divided equally across available GPUs
- Each GPU processes its subset independently
- Results are synchronized after each generation

**Example Configuration**:
```yaml
# === GPU ACCELERATION ===
gpu_acceleration:
  enable_gpu: true
  gpu_devices: [0, 1]              # Use two GPUs
  gpu_batch_size: 1024             # Per GPU batch size
  gpu_memory_limit_gb: 20.0        # Per GPU memory limit
```

**Load Balancing**:
```bash
# Monitor GPU utilization
watch nvidia-smi

# If uneven utilization, try:
image-collage generate target.jpg sources/ output.png \
  --gpu --gpu-devices "0,1" \
  --gpu-batch-size 1536 \
  --population-size 400    # Ensure even division across GPUs
```

### 4. Workload-Specific Optimization

**Small Grids (≤50x50)**:
```yaml
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [50, 50]
  tile_size: [32, 32]
  allow_duplicate_tiles: true

# === GPU ACCELERATION ===
gpu_acceleration:
  gpu_batch_size: 512      # Smaller batches for small grids
  enable_gpu: true
```

**Large Grids (≥150x150)**:
```yaml
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [200, 200]
  tile_size: [32, 32]
  allow_duplicate_tiles: true

# === GPU ACCELERATION ===
gpu_acceleration:
  gpu_batch_size: 4096     # Large batches for maximum throughput
  gpu_memory_limit_gb: 20.0
  auto_mixed_precision: true
```

**Ultra-Large Grids (300x300+)**:
```yaml
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [300, 300]
  tile_size: [32, 32]
  allow_duplicate_tiles: true

# === GPU ACCELERATION ===
gpu_acceleration:
  gpu_batch_size: 8192     # Maximum batch size
  gpu_devices: [0, 1]      # Dual GPU required
  gpu_memory_limit_gb: 22.0
```

## Performance Tuning

### GPU Utilization Monitoring

```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Look for:
# - GPU utilization > 80%
# - Memory usage 70-90% of available
# - Temperature < 85°C
```

### Batch Size Tuning Process

1. **Start Conservative**:
   ```bash
   image-collage generate target.jpg sources/ test.png \
     --preset demo --gpu --gpu-batch-size 512 --verbose
   ```

2. **Increase Gradually**:
   ```bash
   # Try 1024
   image-collage generate target.jpg sources/ test.png \
     --preset demo --gpu --gpu-batch-size 1024 --verbose

   # Try 2048
   image-collage generate target.jpg sources/ test.png \
     --preset demo --gpu --gpu-batch-size 2048 --verbose
   ```

3. **Find Optimal Point**:
   - GPU utilization > 80%
   - No out-of-memory errors
   - Best processing time per generation

### Memory Pressure Handling

**Automatic Scaling**:
```yaml
# === GPU ACCELERATION ===
gpu_acceleration:
  auto_memory_scaling: true        # Automatically adjust batch size
  memory_pressure_threshold: 0.9   # Scale down at 90% memory usage
  fallback_to_cpu: true           # Fall back to CPU if OOM
```

**Manual Memory Management**:
```bash
# Clear GPU memory between runs
python -c "import cupy; cupy.get_default_memory_pool().free_all_blocks()"

# Monitor memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Common Optimization Scenarios

### Scenario 1: Low GPU Utilization (7%)

**Symptoms**:
- GPU utilization < 20%
- Long processing times despite GPU enabled
- GPU memory usage very low

**Solutions**:
```yaml
# === GPU ACCELERATION ===
gpu_acceleration:
  gpu_batch_size: 4096     # Increase from default 1024
  enable_gpu: true
  gpu_devices: [0]         # Ensure single GPU first
```

### Scenario 2: Out of Memory Errors

**Symptoms**:
- CUDA out of memory errors
- Process crashes during generation
- GPU memory usage at 100%

**Solutions**:
```yaml
# === GPU ACCELERATION ===
gpu_acceleration:
  gpu_batch_size: 512           # Reduce batch size
  gpu_memory_limit_gb: 14.0     # Set conservative limit
  auto_mixed_precision: true    # Enable FP16
```

### Scenario 3: Multi-GPU Imbalance

**Symptoms**:
- One GPU at 100%, another at 20%
- Uneven processing times
- Suboptimal overall performance

**Solutions**:
```yaml
# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  population_size: 400      # Ensure even division

# === GPU ACCELERATION ===
gpu_acceleration:
  gpu_devices: [0, 1]
  gpu_batch_size: 1024      # Per-GPU batch size
```

### Scenario 4: CPU Bottleneck

**Symptoms**:
- GPU utilization good but overall slow
- High CPU usage during GPU operations
- I/O wait times

**Solutions**:
```yaml
# === PERFORMANCE SETTINGS ===
performance:
  enable_parallel_processing: true
  num_processes: 8              # Match CPU cores
  cache_size_mb: 2048          # Increase cache
```

## Hardware-Specific Recommendations

### RTX 4090 (24GB VRAM)
```yaml
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [300, 300]     # Large grids supported
  tile_size: [32, 32]
  allow_duplicate_tiles: true

# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  population_size: 200      # Large populations

# === GPU ACCELERATION ===
gpu_acceleration:
  gpu_batch_size: 8192
  gpu_memory_limit_gb: 22.0
  auto_mixed_precision: true
```

### RTX 3080 (10GB VRAM)
```yaml
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [150, 150]     # Medium-large grids
  tile_size: [32, 32]
  allow_duplicate_tiles: true

# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  population_size: 150

# === GPU ACCELERATION ===
gpu_acceleration:
  gpu_batch_size: 2048
  gpu_memory_limit_gb: 9.0
  auto_mixed_precision: true
```

### GTX 1660 (6GB VRAM)
```yaml
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [100, 100]     # Medium grids
  tile_size: [32, 32]
  allow_duplicate_tiles: true

# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  population_size: 100

# === GPU ACCELERATION ===
gpu_acceleration:
  gpu_batch_size: 1024
  gpu_memory_limit_gb: 5.0
  auto_mixed_precision: true
```

### Dual RTX 4090 Setup (48GB Total VRAM)
```yaml
# === BASIC SETTINGS ===
basic_settings:
  grid_size: [400, 400]     # Extreme grid sizes
  tile_size: [32, 32]
  allow_duplicate_tiles: true

# === GENETIC ALGORITHM PARAMETERS ===
genetic_algorithm:
  population_size: 400      # Large populations
  max_generations: 5000     # Long evolution runs

# === GPU ACCELERATION ===
gpu_acceleration:
  enable_gpu: true
  gpu_devices: [0, 1]
  gpu_batch_size: 4096      # Per GPU
  gpu_memory_limit_gb: 22.0 # Per GPU
  auto_mixed_precision: true
```

## Troubleshooting

### Common Issues and Solutions

**Issue**: GPU not detected
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check CuPy installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

**Issue**: Out of memory errors
```bash
# Reduce batch size
image-collage generate target.jpg sources/ output.png \
  --gpu --gpu-batch-size 256

# Enable mixed precision
echo "auto_mixed_precision: true" >> config.yaml
```

**Issue**: Slow performance despite GPU
```bash
# Check if actually using GPU
image-collage generate target.jpg sources/ output.png \
  --gpu --verbose

# Monitor GPU usage
nvidia-smi -l 1
```

### Performance Benchmarking

```bash
# Benchmark different batch sizes
for batch_size in 512 1024 2048 4096; do
  echo "Testing batch size: $batch_size"
  time image-collage generate target.jpg sources/ test_$batch_size.png \
    --preset demo --gpu --gpu-batch-size $batch_size
done
```

### Memory Profiling

```python
# Profile GPU memory usage
import cupy
from cupy.cuda import MemoryPool

# Before generation
mempool = cupy.get_default_memory_pool()
print(f"GPU memory before: {mempool.used_bytes() / 1e9:.2f} GB")

# Run generation...

print(f"GPU memory after: {mempool.used_bytes() / 1e9:.2f} GB")
```

## Best Practices

### 1. Start Conservative
- Begin with smaller batch sizes and increase gradually
- Monitor GPU utilization and memory usage
- Test with demo preset before large runs

### 2. Match Hardware to Workload
- Small grids (≤50x50): Single GPU, moderate batch sizes
- Large grids (100x100+): High-end GPU or multi-GPU setup
- Ultra grids (300x300+): Dual high-end GPUs required

### 3. Balance Memory and Throughput
- Use 70-90% of available GPU memory
- Enable mixed precision for memory-constrained setups
- Leave memory headroom for operating system

### 4. Monitor and Adjust
- Use `nvidia-smi` to monitor utilization
- Adjust batch sizes based on performance
- Consider CPU bottlenecks and I/O limitations

### 5. Multi-GPU Considerations
- Ensure population size divides evenly across GPUs
- Use identical GPU models for best performance
- Monitor load balancing between devices

## Future GPU Enhancements

The following optimizations are planned for future releases:

- **Dynamic Memory Management**: Automatic batch size adjustment based on available memory
- **GPU Streams**: Concurrent execution of multiple operations
- **Advanced Memory Pooling**: Optimized memory allocation patterns
- **Tensor Core Utilization**: Leverage specialized AI acceleration units
- **Multi-Node GPU**: Distributed computing across multiple machines

## Support and Resources

- **NVIDIA Developer Documentation**: https://developer.nvidia.com/cuda
- **CuPy Documentation**: https://docs.cupy.dev/
- **GPU Monitoring Tools**: `nvidia-smi`, `nvtop`, `gpustat`
- **Performance Profiling**: `nsight-systems`, `nsight-compute`

For GPU-related issues, please include the following information:
- GPU model and VRAM amount
- CUDA version (`nvcc --version`)
- CuPy version (`python -c "import cupy; print(cupy.__version__)"`)
- Error messages and stack traces
- Configuration file used