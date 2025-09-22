# Contributing to Image Collage Generator

Welcome to the Image Collage Generator project! We're excited to have you contribute to this sophisticated genetic algorithm-powered photomosaic creation tool.

## 📚 **Table of Contents**

1. [Getting Started](#getting-started)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Architecture & Guidelines](#code-architecture--guidelines)
4. [Contribution Workflow](#contribution-workflow)
5. [Testing Guidelines](#testing-guidelines)
6. [Performance Requirements](#performance-requirements)
7. [Documentation Standards](#documentation-standards)
8. [Review Process](#review-process)
9. [Community Guidelines](#community-guidelines)

---

## 🚀 **Getting Started**

### **Before You Begin**

**🚨 MANDATORY**: Read these documents before making ANY code changes:
- [`CODING_GUIDELINES.md`](CODING_GUIDELINES.md) - Universal architecture principles (REQUIRED)
- [`DEVELOPER_GUIDELINES.md`](DEVELOPER_GUIDELINES.md) - Human workflow patterns
- [`CLAUDE.md`](CLAUDE.md) - Project overview and architecture
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Detailed system architecture

### **Quick Start Checklist**

```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/image-collage-generator.git
cd image-collage-generator

# 2. Set up development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev,gpu,visualization]"

# 3. Run compliance audit (if available)
./scripts/dry_audit.sh || echo "No audit script available yet"

# 4. Verify installation
image-collage --version
image-collage demo examples/target.jpg examples/sources/ --verbose

# 5. Run tests (when implemented)
# pytest image_collage/tests/
```

### **First-Time Contributors**

Start with these **good first issues**:
- Documentation improvements
- Adding type hints to existing functions
- Writing unit tests for utils/ functions
- Performance optimizations (following sampling guidelines)
- Bug fixes in configuration handling

---

## 🛠️ **Development Environment Setup**

### **System Requirements**

- **Python**: 3.8+ (tested on 3.8, 3.9, 3.10, 3.11, 3.13)
- **OS**: Linux, macOS, Windows (WSL recommended)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 2GB+ free space
- **GPU** (optional): NVIDIA GPU with CUDA 11.x or 12.x for acceleration

### **Installation Options**

```bash
# Minimal installation (CPU only)
pip install -e .

# With GPU support (recommended if you have NVIDIA GPU)
pip install -e ".[gpu]"

# Full development setup
pip install -e ".[dev,gpu,visualization]"

# Development with all optional dependencies
pip install -e ".[dev,gpu,visualization,profiling]"
```

### **Development Tools Setup**

```bash
# Code formatting
pip install black isort

# Linting
pip install flake8 mypy

# Testing
pip install pytest pytest-cov pytest-benchmark

# Optional: Pre-commit hooks
pip install pre-commit
pre-commit install
```

### **IDE Configuration**

**VS Code** (recommended settings in `.vscode/settings.json.example`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

---

## 🏗️ **Code Architecture & Guidelines**

### **🚨 CRITICAL: Mandatory Session Startup**

**EVERY development session MUST start with:**

```bash
# 1. Acknowledge guidelines
echo "✅ I have read and will follow CODING_GUIDELINES.md"

# 2. Audit current state
./scripts/dry_audit.sh || echo "Creating audit script needed"

# 3. Search before create
grep -r "function_name" image_collage/ --include="*.py"

# 4. Review utils structure
ls -la image_collage/utils/
grep -r "def.*calculate" image_collage/utils/
```

### **Core Architecture Principles**

#### **1. Utils-First Development** (MANDATORY)
- **All reusable logic MUST go in `image_collage/utils/`**
- **Search existing utils before implementing new functions**
- **Import from utils, don't duplicate functionality**

```python
# ✅ CORRECT: Use existing utility
from image_collage.utils.diversity_metrics import calculate_hamming_diversity

# ❌ WRONG: Duplicate implementation
def my_hamming_diversity():  # Don't create if utils version exists!
```

#### **2. DRY Compliance** (MANDATORY)
```bash
# ALWAYS search before implementing
grep -r "calculate.*diversity" image_collage/ --include="*.py"
grep -r "fitness.*evaluation" image_collage/ --include="*.py"
```

#### **3. Performance Requirements**
- **No O(n²) algorithms without sampling for n > 50**
- **Sample large datasets (max 1000 operations) for expensive calculations**
- **Use GPU acceleration where appropriate**

```python
# ✅ CORRECT: Sample large datasets
if len(population) > 50:
    sample_size = min(1000, len(population))
    sample = random.sample(population, sample_size)
    result = expensive_calculation(sample)
```

### **Module Organization**

```
image_collage/
├── utils/              # ⭐ CENTRALIZED UTILITIES (search here first!)
│   ├── diversity_metrics.py    # All diversity calculations
│   ├── color_tile_generator.py # Color generation utilities
│   └── __init__.py             # Exports all utilities
├── core/               # Main CollageGenerator class
├── genetic/            # Genetic Algorithm Engine
├── preprocessing/      # Image loading and feature extraction
├── fitness/           # Multi-metric fitness evaluation
├── rendering/         # Collage rendering and output
├── cache/             # Performance caching system
├── config/            # Configuration management
├── cli/               # Command-line interface
├── diagnostics/       # Evolution diagnostics
├── lineage/           # Genealogical tracking
└── tests/             # Test files (contribute here!)
```

---

## 🔄 **Contribution Workflow**

### **Step-by-Step Process**

1. **Issue Discussion**
   ```bash
   # Create or comment on an issue first
   # Discuss approach before coding
   # Get maintainer approval for major changes
   ```

2. **Branch Creation**
   ```bash
   git checkout -b feature/descriptive-name
   # Use prefixes: feature/, bugfix/, docs/, performance/
   ```

3. **Development**
   ```bash
   # Follow session startup checklist
   # Search existing implementations
   # Follow utils-first development
   # Add tests for new functionality
   ```

4. **Quality Checks**
   ```bash
   # Code formatting
   black image_collage/
   isort image_collage/

   # Linting
   flake8 image_collage/
   mypy image_collage/

   # Run tests
   pytest image_collage/tests/ -v --cov=image_collage

   # Performance verification
   python benchmark_script.py  # If applicable
   ```

5. **Commit Standards**
   ```bash
   # Use conventional commits
   git commit -m "feat: add spatial diversity calculation to utils"
   git commit -m "fix: resolve LRU cache performance bottleneck"
   git commit -m "docs: update lineage tracking examples"
   git commit -m "perf: optimize fitness evaluation sampling"
   git commit -m "test: add comprehensive diversity metrics tests"
   ```

6. **Pull Request**
   - Fill out PR template completely
   - Include performance impact analysis
   - Add screenshots for UI/output changes
   - Reference related issues

### **Branch Naming Conventions**

```bash
feature/spatial-diversity-manager    # New features
bugfix/lineage-tracking-integration  # Bug fixes
performance/cache-optimization       # Performance improvements
docs/contributing-guide             # Documentation
test/fitness-evaluation-coverage    # Test additions
refactor/diversity-path-consolidation # Code refactoring
```

### **Commit Message Format**

```
type(scope): brief description

Detailed explanation if needed.

- List specific changes
- Performance impact
- Breaking changes

Closes #123
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

---

## 🧪 **Testing Guidelines**

### **Testing Philosophy**

- **Write tests for ALL new utils/ functions**
- **Add integration tests for major features**
- **Include performance benchmarks for expensive operations**
- **Test edge cases and error conditions**

### **Test Organization**

```
tests/
├── unit/
│   ├── test_utils/
│   │   ├── test_diversity_metrics.py
│   │   └── test_color_tile_generator.py
│   ├── test_genetic/
│   ├── test_fitness/
│   └── test_config/
├── integration/
│   ├── test_full_generation.py
│   ├── test_checkpoint_resume.py
│   └── test_gpu_acceleration.py
├── performance/
│   ├── benchmark_diversity_metrics.py
│   ├── benchmark_fitness_evaluation.py
│   └── benchmark_cache_performance.py
└── fixtures/
    ├── sample_target.jpg
    ├── sample_sources/
    └── test_configs/
```

### **Writing Tests**

```python
# Example test structure
import pytest
from image_collage.utils.diversity_metrics import calculate_hamming_diversity

class TestHammingDiversity:
    def test_identical_populations(self):
        """Test that identical populations have zero diversity."""
        population = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        assert calculate_hamming_diversity(population) == 0.0

    def test_completely_different_populations(self):
        """Test maximum diversity case."""
        population = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        result = calculate_hamming_diversity(population)
        assert 0.0 < result <= 1.0

    def test_large_population_sampling(self):
        """Test that large populations are properly sampled."""
        large_population = [[i] * 100 for i in range(2000)]
        # Should complete quickly due to sampling
        result = calculate_hamming_diversity(large_population)
        assert isinstance(result, float)

    @pytest.mark.benchmark
    def test_performance(self, benchmark):
        """Benchmark diversity calculation performance."""
        population = [[random.randint(1, 100) for _ in range(100)]
                     for _ in range(1000)]
        result = benchmark(calculate_hamming_diversity, population)
        assert result is not None
```

### **Running Tests**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=image_collage --cov-report=html

# Run performance tests
pytest tests/performance/ -v

# Run specific test file
pytest tests/unit/test_utils/test_diversity_metrics.py -v

# Run benchmarks
pytest tests/performance/ --benchmark-only
```

---

## ⚡ **Performance Requirements**

### **Mandatory Performance Standards**

1. **Algorithmic Complexity**
   - No O(n²) without sampling for n > 50
   - All utils functions must handle large inputs efficiently
   - Use random sampling (max 1000 operations) for expensive calculations

2. **Memory Management**
   - Functions must not cause memory leaks
   - Large arrays should be explicitly deleted when done
   - GPU memory must be properly managed

3. **Benchmarking Requirements**
   ```python
   # All performance-critical functions need benchmarks
   @pytest.mark.benchmark
   def test_function_performance(benchmark):
       large_input = generate_large_test_data()
       result = benchmark(your_function, large_input)
       # Function should complete in < 1 second for typical inputs
   ```

### **GPU Acceleration Guidelines**

```python
# Use GPU acceleration appropriately
try:
    import cupy as cp
    # GPU implementation
    def gpu_fitness_evaluation(data):
        gpu_data = cp.asarray(data)
        # GPU operations
        return cp.asnumpy(result)
except ImportError:
    # CPU fallback
    def cpu_fitness_evaluation(data):
        # CPU implementation
        return result
```

---

## 📖 **Documentation Standards**

### **Required Documentation**

1. **Docstrings** (Google style):
   ```python
   def calculate_spatial_diversity(population, tile_positions, sample_size=None):
       """Calculate spatial diversity of population arrangements.

       Measures how spatially diverse the tile arrangements are across
       the population, focusing on local neighborhood patterns.

       Args:
           population (List[List[int]]): Population of tile arrangements
           tile_positions (List[Tuple[int, int]]): Grid positions for each tile
           sample_size (Optional[int]): Maximum population size to sample.
               If None, uses sampling threshold from config.

       Returns:
           float: Spatial diversity score between 0.0 and 1.0

       Raises:
           ValueError: If population is empty or positions don't match

       Example:
           >>> pop = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
           >>> positions = [(0,0), (0,1), (1,0)]
           >>> diversity = calculate_spatial_diversity(pop, positions)
           >>> print(f"Spatial diversity: {diversity:.3f}")
           Spatial diversity: 0.847
       """
   ```

2. **README Updates**:
   - Update CLI examples for new features
   - Add configuration parameters
   - Include performance notes

3. **Architecture Documentation**:
   - Update `docs/ARCHITECTURE.md` for new components
   - Add new modules to component diagrams
   - Document integration points

### **Code Comments**

```python
# ✅ GOOD: Explain WHY, not what
# Use sampling to prevent O(n²) performance degradation on large populations
if len(population) > self.config.sampling_threshold:
    sample = random.sample(population, self.config.max_sample_size)

# ❌ BAD: Obvious comments
# Set x to 5
x = 5
```

---

## 🔍 **Review Process**

### **Pull Request Requirements**

**Every PR must include:**

- [ ] **Performance impact analysis**: "Adds 0.1s per generation for new feature"
- [ ] **Architecture compliance**: "Follows utils-first pattern, no code duplication"
- [ ] **Test coverage**: "95% coverage maintained, added 12 new tests"
- [ ] **Documentation updates**: "Updated CLI help and ARCHITECTURE.md"
- [ ] **Breaking changes**: "None" or detailed explanation
- [ ] **GPU compatibility**: "CPU fallback implemented"

### **Review Checklist for Maintainers**

**Code Quality:**
- [ ] Follows CODING_GUIDELINES.md principles
- [ ] No duplicate implementations (checked with grep)
- [ ] Uses existing utils/ functions where appropriate
- [ ] Proper error handling and edge cases

**Performance:**
- [ ] No O(n²) algorithms without sampling
- [ ] Large dataset handling verified
- [ ] GPU acceleration implemented where beneficial
- [ ] Memory usage tested

**Architecture:**
- [ ] Follows established module organization
- [ ] Clean integration with existing systems
- [ ] Configuration properly handled
- [ ] CLI interface updated if needed

**Testing:**
- [ ] Adequate test coverage (>90% for new code)
- [ ] Performance benchmarks included
- [ ] Integration tests for major features
- [ ] Edge cases covered

**Documentation:**
- [ ] Docstrings follow Google style
- [ ] README updated for user-facing changes
- [ ] Architecture docs updated
- [ ] Examples provided for complex features

### **Approval Process**

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: Minimum 1 maintainer approval
3. **Performance Review**: Performance impact assessed
4. **Documentation Review**: All docs updated appropriately
5. **Final Approval**: Maintainer merge approval

---

## 🤝 **Community Guidelines**

### **Code of Conduct**

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide actionable feedback
- **Be patient**: Remember contributors have different experience levels
- **Be collaborative**: Work together to improve the project

### **Getting Help**

**For Development Questions:**
- Create a GitHub Discussion
- Ask in pull request comments
- Reference relevant documentation

**For Bug Reports:**
- Use issue templates
- Include minimal reproduction case
- Provide system information
- Follow debugging guides in `docs/DEBUGGING.md`

**For Feature Requests:**
- Discuss in GitHub Issues first
- Provide use cases and examples
- Consider performance implications
- Review existing roadmap

### **Recognition**

**Contributors will be recognized for:**
- Code contributions (features, fixes, performance improvements)
- Documentation improvements
- Test coverage improvements
- Performance optimizations
- Bug reports with detailed analysis
- Helping other contributors

---

## 🎯 **Quick Reference**

### **Before Every Session**
```bash
echo "✅ Read CODING_GUIDELINES.md"
./scripts/dry_audit.sh
grep -r "function_name" image_collage/ --include="*.py"
```

### **Before Every Commit**
```bash
black image_collage/
flake8 image_collage/
pytest tests/ -x
```

### **Common Commands**
```bash
# Search existing implementations
grep -r "diversity" image_collage/utils/ --include="*.py"

# Check performance
python -m pytest tests/performance/ --benchmark-only

# Full quality check
black . && flake8 . && pytest --cov=image_collage
```

---

## 📞 **Contact & Support**

- **Issues**: [GitHub Issues](https://github.com/project/image-collage-generator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/project/image-collage-generator/discussions)
- **Documentation**: [`docs/`](docs/) folder
- **Architecture Questions**: See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Development Setup**: See [`DEVELOPER_GUIDELINES.md`](DEVELOPER_GUIDELINES.md)

---

**Thank you for contributing to the Image Collage Generator! Together we're building a sophisticated, high-performance genetic algorithm system for creating beautiful photomosaic art.** 🎨🧬✨