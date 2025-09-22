# ImageCollage System Architecture

## Overview

The ImageCollage system is a sophisticated genetic algorithm-based photomosaic generator with comprehensive analysis capabilities. This document provides a detailed architectural view to support debugging and development.

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                CLI Layer (Entry Point)                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  cli/main.py          │  cli/helpers.py                                               │
│  - generate()         │  - create_generator()                                         │
│  - resume()          │  - load_images()                                              │
│  - analyze()         │  - create_progress_callback()                                 │
│  - export_config()   │  - calculate_selection_pressure()                             │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            Core Orchestration Layer                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  core/collage_generator.py                                                            │
│  - CollageGenerator (Main Controller)                                                 │
│  - CollageResult (Output Data Structure)                                              │
│  - generate() / resume_from_checkpoint()                                              │
│  - _initialize_analysis_components()                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐
│    Configuration        │    │    Data Processing      │    │   Evolution Engine      │
│        Layer           │    │        Layer           │    │        Layer           │
├─────────────────────────┤    ├─────────────────────────┤    ├─────────────────────────┤
│ config/settings.py      │    │ preprocessing/          │    │ genetic/ga_engine.py    │
│ - CollageConfig         │    │   image_processor.py    │    │ - GeneticAlgorithmEngine│
│ - GeneticParams         │    │ cache/manager.py        │    │ - Selection & Evolution │
│ - FitnessWeights        │    │ - CacheManager          │    │ - Population Management │
│ - PresetConfigs         │    │ - LRU Image Cache       │    │                         │
└─────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘
                                              │                         │
                                              ▼                         ▼
                               ┌─────────────────────────┐    ┌─────────────────────────┐
                               │   Fitness Evaluation    │    │   Genetic Operators     │
                               │        Layer           │    │        Layer           │
                               ├─────────────────────────┤    ├─────────────────────────┤
                               │ fitness/evaluator.py    │    │ genetic/                │
                               │ fitness/gpu_evaluator.py│    │ - comprehensive_diversity│
                               │ - CPU/GPU Fitness       │    │ - spatial_diversity     │
                               │ - Multi-metric scoring  │    │ - fitness_sharing       │
                               │ - CIEDE2000, Texture    │    │ - intelligent_restart   │
                               └─────────────────────────┘    │ - island_model          │
                                                             │ - diversity_dashboard   │
                                                             └─────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            Analysis & Monitoring Layer                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  diagnostics/           │  lineage/               │  checkpoints/                     │
│  - collector.py         │  - tracker.py           │  - manager.py                     │
│  - visualizer.py        │  - visualizer.py        │  - state.py                       │
│  - GenerationStats      │  - fitness_components.py│  - EvolutionState                 │
│  - DiagnosticsData      │  - Genealogy Tracking   │  - Crash Recovery                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              Output & Utilities Layer                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  rendering/renderer.py  │  utils/                                                     │
│  - Renderer             │  - diversity_metrics.py                                    │
│  - Final Image Assembly │  - color_tile_generator.py                                 │
│  - Preview Generation   │  - Centralized Utility Functions                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow Architecture

### 1. Initialization Flow
```
CLI → CollageGenerator.__init__() → Component Initialization
                                 ├── ImageProcessor
                                 ├── CacheManager
                                 ├── FitnessEvaluator (CPU/GPU)
                                 ├── GeneticAlgorithmEngine
                                 └── Renderer
```

### 2. Generation Flow
```
generate() → load_target() → load_sources() → GA Evolution Loop
                                            ├── fitness_evaluation()
                                            ├── selection()
                                            ├── crossover()
                                            ├── mutation()
                                            ├── diversity_analysis()
                                            ├── checkpoint_saving()
                                            └── progress_callback()
```

### 3. Analysis Flow
```
Evolution Loop → DiagnosticsCollector → GenerationStats
              → LineageTracker        → Individual Tracking
              → DiversityDashboard    → Real-time Monitoring
              → CheckpointManager     → State Persistence
```

## 📦 Detailed Component Architecture

### Core Components

#### CollageGenerator (Central Orchestrator)
**Location**: `core/collage_generator.py`
**Dependencies**: All major subsystems
**Responsibilities**:
- System orchestration and workflow management
- Component initialization and lifecycle management
- Evolution loop coordination
- Analysis system integration
- Error handling and fallback mechanisms

**Key Methods**:
- `generate()` - Main generation workflow
- `resume_from_checkpoint()` - Crash recovery workflow
- `_initialize_analysis_components()` - Analysis setup
- `load_target()` / `load_sources()` - Image loading

#### GeneticAlgorithmEngine (Evolution Core)
**Location**: `genetic/ga_engine.py`
**Dependencies**: Config, Diversity modules
**Responsibilities**:
- Population initialization and management
- Selection, crossover, and mutation operations
- Fitness-based evolution
- Integration with advanced genetic operators

**Key Methods**:
- `initialize_population()`
- `evolve_population()`
- `get_population()` / `set_population()`

### Advanced Genetic Operators

#### ComprehensiveDiversityManager
**Location**: `genetic/comprehensive_diversity.py`
**Purpose**: Multi-metric diversity analysis
**Metrics**: Hamming distance, entropy, clustering, spatial patterns

#### SpatialDiversityManager
**Location**: `genetic/spatial_diversity.py`
**Purpose**: Image-specific spatial diversity analysis
**Metrics**: Local patterns, edge diversity, contiguous regions

#### IslandModel
**Location**: `genetic/island_model.py`
**Purpose**: Multi-population evolution with migration
**Features**: Inter-island migration, population isolation, diversity preservation

#### IntelligentRestart
**Location**: `genetic/intelligent_restart.py`
**Purpose**: Stagnation detection and population restart
**Features**: Automatic restart triggers, elite preservation

#### FitnessSharing
**Location**: `genetic/fitness_sharing.py`
**Purpose**: Diversity-based fitness adjustment
**Features**: Phenotypic similarity, shared fitness calculation

#### DiversityDashboard
**Location**: `genetic/diversity_dashboard.py`
**Purpose**: Real-time evolution monitoring
**Features**: Live metrics, alert system, periodic reporting

### Fitness Evaluation System

#### FitnessEvaluator (CPU)
**Location**: `fitness/evaluator.py`
**Metrics**:
- **Color Similarity** (40%): CIEDE2000 color difference
- **Luminance Matching** (25%): Brightness distribution
- **Texture Correlation** (20%): Local binary patterns
- **Edge Preservation** (15%): Sobel operator edge detection

#### GPUFitnessEvaluator (GPU-Accelerated)
**Location**: `fitness/gpu_evaluator.py`
**Features**: CuPy-based GPU acceleration, multi-GPU support, batch processing

### Analysis & Monitoring Systems

#### DiagnosticsCollector
**Location**: `diagnostics/collector.py`
**Purpose**: Comprehensive evolution data collection
**Data Collected**:
- GenerationStats with 50+ metrics per generation
- Genetic operation effectiveness
- Population dynamics and diversity trends
- Performance and timing data

#### DiagnosticsVisualizer
**Location**: `diagnostics/visualizer.py`
**Outputs**: 10+ visualization plots + 3 data files
**Visualizations**:
- Dashboard overview, fitness evolution, genetic operations
- Population analysis, performance metrics, diversity trends
- Evolution grids, spatial analysis, advanced metrics

#### LineageTracker
**Location**: `lineage/tracker.py`
**Purpose**: Complete genealogical tracking
**Features**:
- Individual genealogy, parent-child relationships
- Birth method tracking, age distribution
- Migration event recording, survival analysis

#### LineageVisualizer
**Location**: `lineage/visualizer.py`
**Outputs**: 16+ genealogical visualizations
**Visualizations**:
- Lineage trees, population dynamics, diversity evolution
- Migration patterns, survival curves, dominance analysis
- Genealogy networks, evolutionary timelines

### Persistence & Recovery Systems

#### CheckpointManager
**Location**: `checkpoints/manager.py`
**Purpose**: Crash recovery and resume functionality
**Features**:
- Automatic state snapshots, configurable intervals
- Smart cleanup, metadata preservation
- Deterministic resume with random state restoration

#### EvolutionState
**Location**: `checkpoints/state.py`
**Purpose**: Complete evolution state encapsulation
**Preserved Data**:
- Population, fitness history, random states
- Configuration, timing, generation counters
- Optional: diagnostics state, lineage data, evolution frames

### Support Systems

#### ImageProcessor
**Location**: `preprocessing/image_processor.py`
**Purpose**: Image loading, preprocessing, and feature extraction
**Features**: Multi-format support, automatic resizing, feature caching

#### CacheManager
**Location**: `cache/manager.py`
**Purpose**: Performance optimization through caching
**Features**: LRU cache for images and features, memory management

#### Renderer
**Location**: `rendering/renderer.py`
**Purpose**: Final collage assembly and preview generation
**Features**: High-quality rendering, evolution frame generation, format conversion

## 🔧 Configuration Architecture

### Configuration Hierarchy
```
CollageConfig (Root)
├── GeneticParams (Evolution parameters)
├── FitnessWeights (Fitness component weights)
├── GPUConfig (GPU acceleration settings)
├── IslandModelConfig (Multi-population settings)
├── DashboardConfig (Real-time monitoring)
└── CheckpointConfig (Persistence settings)
```

### Configuration Flow
```
YAML/JSON File → CollageConfig.load_from_file() → Component Initialization
CLI Arguments → Parameter Override → Configuration Validation
PresetConfigs → Predefined Templates → Quick Setup
```

## 🚨 Known Architectural Issues

### 1. Circular Import Dependencies
**Issue**: `core/collage_generator.py` ↔ `diagnostics/` modules
**Resolution**: Runtime imports moved inside functions
**Impact**: DIAGNOSTICS_AVAILABLE flag reliability

### 2. Configuration Path Resolution
**Issue**: CLI path parameters not properly applied to components
**Symptoms**: Hardcoded "lineage" instead of "lineage_comprehensive"
**Root Cause**: Parameter passing chain breaks in CLI → Generator transition

### 3. Multiple Diversity Calculation Paths (TECH_DEBT)
**Issue**: Three redundant diversity calculation approaches
**Location**: `core/collage_generator.py:716-778`
**Impact**: Maintenance burden, potential inconsistencies

### 4. Hardcoded Getattr Fallbacks (TECH_DEBT)
**Issue**: Hardcoded diversity metric defaults never used
**Location**: `core/collage_generator.py:376-384`
**Impact**: Code confusion, maintenance overhead

## 🔄 Component Interaction Patterns

### Initialization Pattern
1. **Configuration Loading**: YAML → CollageConfig
2. **Component Construction**: Config → Individual components
3. **Dependency Injection**: Components reference each other via config
4. **Availability Checking**: Runtime capability detection (GPU, diagnostics)

### Evolution Loop Pattern
1. **Population Management**: GA Engine handles genetic operations
2. **Fitness Evaluation**: Delegated to CPU/GPU evaluator
3. **Analysis Collection**: Multiple collectors gather metrics
4. **Progress Reporting**: Callbacks to CLI layer
5. **State Persistence**: Checkpoints save complete state

### Analysis Integration Pattern
1. **Data Collection**: Components push data to collectors
2. **Metric Calculation**: Centralized calculation in utils/
3. **Visualization Generation**: Batch processing at evolution end
4. **Output Organization**: Timestamped directory structure

## 🎯 Performance Architecture

### GPU Acceleration Path
```
GPUFitnessEvaluator → CuPy Arrays → Batch Processing → Multi-GPU Support
```

### Parallel Processing Path
```
MultiProcessing Pool → Fitness Evaluation → Population-level Parallelism
```

### Caching Strategy
```
LRU Cache → Image Features → Preprocessed Data → Memory Management
```

### Memory Management
```
Configurable Limits → Automatic Quality Reduction → Resource Monitoring
```

## 🔍 Debugging Guide

### Common Debug Points
1. **Configuration Issues**: Check CLI parameter passing in `cli/main.py:204-210`
2. **Import Failures**: Verify runtime imports in `core/collage_generator.py`
3. **GPU Problems**: Check GPU availability flags and fallback mechanisms
4. **Analysis Missing**: Verify analysis component initialization
5. **Path Resolution**: Trace config values through CLI → Generator → Components

### Key Debugging Files
- `core/collage_generator.py` - Central orchestration
- `cli/main.py` - CLI parameter processing
- `config/settings.py` - Configuration system
- `genetic/ga_engine.py` - Evolution core
- `diagnostics/collector.py` - Analysis data collection

### Debug Information Sources
- CLI verbose output (`--verbose`)
- Diversity dashboard real-time updates
- Checkpoint metadata files
- Diagnostics JSON data exports
- Evolution state inspection

This architecture diagram should provide comprehensive guidance for debugging and understanding the ImageCollage system's complex interactions and data flows.