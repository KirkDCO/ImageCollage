# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Image Collage Generator.

## 🚨 Quick Diagnostic Commands

```bash
# Check if installation is working
image-collage --version

# Test with minimal demo
image-collage demo target.jpg sources/ --verbose

# Check GPU availability
python -c "import cupy; print(f'GPU devices: {cupy.cuda.get_device_count()}')"

# Memory check
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB')"
```

## 📋 Common Issues and Solutions

### Installation Problems

#### Issue: `ModuleNotFoundError: No module named 'image_collage'`
**Solution:**
```bash
# Ensure proper installation
pip install -e .

# Or with all features
pip install -e ".[gpu,visualization,dev]"

# Check if package is installed
pip list | grep image-collage
```

#### Issue: `ImportError: cannot import name 'cupy'`
**Solution:**
```bash
# Install CuPy for your CUDA version
pip install cupy-cuda12x  # For CUDA 12.x
pip install cupy-cuda11x  # For CUDA 11.x

# Check CUDA installation
nvidia-smi
nvcc --version
```

#### Issue: Missing visualization dependencies
**Solution:**
```bash
# Install visualization packages
pip install matplotlib seaborn pandas

# Or install complete package
pip install -e ".[visualization]"
```

### Generation Problems

#### Issue: "No source images found"
**Symptoms:**
- Error message about empty source directory
- Generation fails immediately

**Solutions:**
```bash
# Check directory exists and has images
ls -la source_images/
find source_images/ -name "*.jpg" -o -name "*.png" | head -10

# Supported formats: JPG, JPEG, PNG, BMP, TIFF
# Try with verbose flag to see what's being loaded
image-collage generate target.jpg sources/ output.png --verbose
```

#### Issue: "Target image cannot be loaded"
**Symptoms:**
- Error loading target image
- Unsupported format warnings

**Solutions:**
```bash
# Check image file
file target.jpg
identify target.jpg  # ImageMagick tool

# Convert if needed
convert target.bmp target.jpg
```

#### Issue: Generation extremely slow
**Symptoms:**
- Takes hours for small grids
- High CPU usage, low progress

**Solutions:**
```bash
# Use faster preset
image-collage generate target.jpg sources/ output.png --preset quick

# Enable parallel processing
image-collage generate target.jpg sources/ output.png --parallel

# Reduce grid size
image-collage generate target.jpg sources/ output.png --grid-size 30 30

# Check system resources
top
htop
```

### Memory Issues

#### Issue: "Out of memory" errors
**Symptoms:**
- Python crashes with MemoryError
- System becomes unresponsive
- Process killed by OS

**Solutions:**
```bash
# Reduce memory usage
image-collage generate target.jpg sources/ output.png \
  --grid-size 50 50 \
  --population-size 50 \
  --preset quick

# Limit source images
image-collage generate target.jpg sources/ output.png \
  --max-source-images 1000

# Check available memory
free -h
```

#### Issue: GPU out of memory
**Symptoms:**
- CUDA out of memory errors
- Generation crashes when using --gpu

**Solutions:**
```bash
# Reduce GPU batch size
image-collage generate target.jpg sources/ output.png \
  --gpu --gpu-batch-size 256

# Use mixed precision
image-collage generate target.jpg sources/ output.png \
  --gpu --config gpu_config.yaml

# Check GPU memory
nvidia-smi
```

### GPU Problems

#### Issue: GPU not detected
**Symptoms:**
- "No CUDA devices found"
- GPU flags ignored

**Diagnostic Steps:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Test CuPy
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

# Check PyTorch CUDA (if installed)
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions:**
```bash
# Install/update NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-535

# Install CUDA toolkit
# Visit: https://developer.nvidia.com/cuda-downloads

# Reinstall CuPy
pip uninstall cupy
pip install cupy-cuda12x
```

#### Issue: Low GPU utilization
**Symptoms:**
- GPU usage < 20% in nvidia-smi
- Slow generation despite GPU enabled

**Solutions:**
```bash
# Increase batch size
image-collage generate target.jpg sources/ output.png \
  --gpu --gpu-batch-size 2048

# Use larger grid
image-collage generate target.jpg sources/ output.png \
  --gpu --grid-size 100 100

# Monitor GPU usage
watch nvidia-smi
```

### Configuration Issues

#### Issue: "Invalid configuration file"
**Symptoms:**
- YAML parsing errors
- Configuration validation failures

**Solutions:**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Use preset as template
image-collage export-config working_config.yaml --preset balanced

# Check for common YAML issues
# - Indentation (use spaces, not tabs)
# - Colons followed by spaces
# - No trailing spaces
```

#### Issue: Unexpected parameter values
**Symptoms:**
- Parameters ignored
- Warnings about invalid values

**Solutions:**
```bash
# Check parameter ranges
# population_size: 10-1000
# mutation_rate: 0.01-0.5
# grid_size: [10,10] to [500,500]

# Use verbose mode to see actual values
image-collage generate target.jpg sources/ output.png --config config.yaml --verbose
```

### Quality Issues

#### Issue: Poor results/Low fitness
**Symptoms:**
- Collage doesn't resemble target
- Low final fitness scores
- Blurry or pixelated results

**Solutions:**
```bash
# Increase generations
image-collage generate target.jpg sources/ output.png --generations 2000

# Use more source images
# Aim for 5-10x more sources than grid tiles

# Improve source collection diversity
image-collage analyze target.jpg sources/

# Adjust fitness weights
# For portraits: increase color_weight
# For landscapes: balance color and texture
```

#### Issue: Collage too abstract
**Symptoms:**
- Can't recognize subject in result
- Very low detail

**Solutions:**
```bash
# Increase grid size
image-collage generate target.jpg sources/ output.png --grid-size 100 100

# Use high quality preset
image-collage generate target.jpg sources/ output.png --preset high

# Disable duplicates for more variety
image-collage generate target.jpg sources/ output.png --no-duplicates
```

### Performance Issues

#### Issue: Generation stalled/stuck
**Symptoms:**
- Fitness not improving
- Same fitness for many generations

**Solutions:**
```bash
# Enable intelligent restart (⚠️ WARNING: Not functional - uses basic restart instead)
image-collage generate target.jpg sources/ output.png \
  --enable-intelligent-restart

# Use basic restart (this actually works)
image-collage generate target.jpg sources/ output.png \
  --preset advanced  # Has aggressive restart settings

# Use fitness sharing
image-collage generate target.jpg sources/ output.png \
  --enable-fitness-sharing

# Increase mutation rate
image-collage generate target.jpg sources/ output.png \
  --mutation-rate 0.1
```

#### Issue: CPU bottleneck
**Symptoms:**
- High CPU usage
# One or few cores at 100%

**Solutions:**
```bash
# Enable parallel processing
image-collage generate target.jpg sources/ output.png \
  --parallel --num-processes 8

# Reduce population size
image-collage generate target.jpg sources/ output.png \
  --population-size 50

# Use SSD for source images
# Move source collection to SSD
```

## 🔧 Advanced Debugging

### Enable Comprehensive Logging

```bash
# Maximum verbosity
image-collage generate target.jpg sources/ output.png \
  --verbose \
  --diagnostics debug_analysis/ \
  --save-animation debug.gif

# Check log files
ls -la debug_analysis/
cat debug_analysis/summary.txt
```

### Memory Profiling

```python
# Create memory_test.py
import psutil
import os
from image_collage import CollageGenerator
from image_collage.config import PresetConfigs

def monitor_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Initial memory: {monitor_memory():.1f} MB")

config = PresetConfigs.quick()
generator = CollageGenerator(config)

print(f"After generator: {monitor_memory():.1f} MB")

generator.load_target("target.jpg")
print(f"After target: {monitor_memory():.1f} MB")

generator.load_sources("sources/")
print(f"After sources: {monitor_memory():.1f} MB")

result = generator.generate()
print(f"After generation: {monitor_memory():.1f} MB")
```

### GPU Memory Profiling

```python
# Create gpu_test.py
import cupy

def check_gpu_memory():
    mempool = cupy.get_default_memory_pool()
    return mempool.used_bytes() / 1e9  # GB

print(f"GPU memory used: {check_gpu_memory():.2f} GB")

# Run generation here

print(f"GPU memory after: {check_gpu_memory():.2f} GB")
```

### Performance Benchmarking

```bash
# Test different settings
for preset in demo quick balanced high; do
    echo "Testing $preset preset..."
    time image-collage generate target.jpg sources/ test_$preset.png \
      --preset $preset \
      --verbose 2>&1 | grep "Processing time"
done
```

## 📊 System Requirements Check

### Minimum System Check
```bash
# Check Python version (3.8+ required)
python --version

# Check available RAM (8GB+ recommended)
free -h

# Check CPU cores (4+ recommended)
nproc

# Check available disk space (2GB+ recommended)
df -h .

# Check image libraries
python -c "import cv2, PIL, numpy; print('Image libraries OK')"
```

### Optimal System Check
```bash
# Check for optimal specs
echo "=== SYSTEM ANALYSIS ==="
echo "Python: $(python --version)"
echo "CPU Cores: $(nproc)"
echo "RAM: $(free -h | grep 'Mem:' | awk '{print $2}')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No NVIDIA GPU')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep 'release' || echo 'CUDA not found')"
echo "Storage: $(df -h . | tail -1 | awk '{print $4}') available"
```

## 🆘 Getting Help

### Collecting Debug Information

Before reporting issues, collect this information:

```bash
# System info
image-collage --version
python --version
pip list | grep -E "(cupy|numpy|opencv|pillow)"
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU"

# Test with minimal example
image-collage demo target.jpg sources/ --verbose > debug.log 2>&1

# Include:
# - Full error message and stack trace
# - System specifications
# - Command that failed
# - Sample target and source images (if possible)
# - Configuration file used
```

### Common Error Messages

| Error | Likely Cause | Quick Fix |
|-------|-------------|-----------|
| `CUDA out of memory` | GPU memory exceeded | Reduce `--gpu-batch-size` |
| `No source images found` | Empty/wrong directory | Check source path with `ls` |
| `Invalid grid size` | Grid too large/small | Use values between [10,10] and [300,300] |
| `Convergence failed` | Population stuck | Add `--enable-restart` |
| `Permission denied` | File/directory access | Check file permissions |
| `Module not found` | Missing dependencies | Reinstall with `pip install -e .` |

### Performance Optimization Checklist

- ✅ Use SSD storage for source images
- ✅ Enable parallel processing (`--parallel`)
- ✅ Use appropriate preset for your needs
- ✅ Match grid size to your source collection size
- ✅ Enable GPU acceleration if available
- ✅ Monitor system resources during generation
- ✅ Use checkpoints for long runs (`--save-checkpoints`)

### Best Practices

1. **Start Small**: Always test with demo preset first
2. **Monitor Resources**: Watch CPU, memory, and GPU usage
3. **Use Checkpoints**: Enable for runs > 30 minutes
4. **Optimize Sources**: Curate diverse, high-quality source collections
5. **Save Configs**: Export and save working configurations
6. **Regular Backups**: Save important results and configurations

---

If you continue to experience issues after following this guide, please create an issue with full debug information including system specs, error messages, and the exact command used.