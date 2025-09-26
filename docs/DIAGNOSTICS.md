# Diagnostic System Documentation

## Overview

The Image Collage Generator includes a comprehensive diagnostic system that provides detailed analysis of genetic algorithm evolution, including population dynamics, fitness progression, genetic operations effectiveness, and diversity metrics.

## Migration Event Annotations

### Introduction

Migration event annotations provide visual indicators on diagnostic plots showing when island model migrations occur during genetic algorithm evolution. These annotations help researchers and users understand the timing and impact of population exchanges between islands.

### Implementation Status

✅ **FULLY IMPLEMENTED** - Working as of 2025-09-23

**Supported Plots**: 7 out of 10 diagnostic plots now include migration annotations:

1. **`fitness_evolution.png`** ⭐ - **Full annotations** (detailed markers + text)
2. **`genetic_operations.png`** ✅ - **Simple vertical lines**
3. **`performance_metrics.png`** ✅ - **Simple vertical lines**
4. **`population_analysis.png`** ✅ - **Simple vertical lines**
5. **`comprehensive_diversity.png`** ✅ - **Simple vertical lines**
6. **`spatial_diversity.png`** ✅ - **Simple vertical lines**
7. **`advanced_metrics.png`** ✅ - **Simple vertical lines**

**Not Annotated** (non-time-series plots):
- `dashboard.png` - Summary dashboard (no time axis)
- `evolution_grid.png` - Heatmap visualization
- `fitness_distribution.png` - Statistical distribution analysis

### Visual Features

#### Full Annotations (fitness_evolution.png)
- **Purple vertical dotted lines** at migration generations
- **Purple star markers** with size scaled by event count
- **Text annotations** with migration details (e.g., "3 Migrations\nGen 8")
- **Legend entry** for "Migration Events"

#### Simple Vertical Lines (all other time-series plots)
- **Purple vertical dotted lines** (`:` linestyle, `alpha=0.6`)
- **Single legend entry** per plot for "Migration Events"
- **Minimal visual impact** while clearly marking migration timing

### Configuration Requirements

```yaml
# Enable island model (required for migration events)
genetic_algorithm:
  enable_island_model: true
  island_model_num_islands: 3
  island_model_migration_interval: 4    # Migrate every 4 generations
  island_model_migration_rate: 0.15

# Enable diagnostics (required for visualization)
basic_settings:
  enable_diagnostics: true
  diagnostics_output_dir: "diagnostics_folder"
```

### Data Recording

Migration events are automatically recorded when:
1. **Island model is enabled** in configuration
2. **Migration interval** is reached (e.g., every N generations)
3. **Diagnostics collector** is active

Each migration event records:
```json
{
  "generation": 4,
  "source_island": 0,
  "target_island": 1,
  "migrant_fitness": 0.295,
  "num_migrants": 1,
  "timestamp": 0.22245192527770996
}
```

### Technical Implementation

#### Key Components

1. **DiagnosticsCollector.record_migration_event()**
   - Records migration events with metadata
   - Automatically called during island model evolution
   - Exports events to JSON for analysis

2. **DiagnosticsVisualizer._add_migration_annotations()**
   - Full annotation system for fitness evolution plot
   - Includes detailed markers, text, and arrows

3. **DiagnosticsVisualizer._add_simple_migration_lines()**
   - Simple vertical line system for other time-series plots
   - Minimal visual impact, maximum information

4. **GA Engine Integration**
   - Automatically detects island model migrations
   - Records events via diagnostics collector
   - Handles multiple migration events per generation

#### Integration Pipeline

```
Island Model Manager → GA Engine → Diagnostics Collector → Diagnostics Visualizer
     ↓                    ↓              ↓                      ↓
Migration occurs → Event recorded → Saved to JSON → Visual annotations
```

### Testing and Validation

#### Test Migration Annotations

```bash
# Create test configuration with island model
image-collage generate target.jpg sources/ test.png \
  --config island_model_config.yaml \
  --diagnostics test_diagnostics/ \
  --verbose

# Verify migration events were recorded
grep -A 10 "migration_events" test_diagnostics/diagnostics_data.json

# Verify visual annotations on plots
ls test_diagnostics/*.png
# Should show purple vertical lines on time-series plots
```

#### Expected Results

- **Migration events** appear in `diagnostics_data.json`
- **Purple vertical lines** visible on 6 time-series plots
- **Detailed annotations** on fitness evolution plot
- **Migration timing** aligns with configured interval

### Performance Impact

- **Recording**: <0.1% overhead per migration event
- **Storage**: ~100 bytes per migration event in JSON
- **Visualization**: ~0.5 seconds additional rendering time
- **Memory**: Minimal impact (events stored as simple dictionaries)

### Troubleshooting

#### Problem: No Migration Events Recorded

**Symptoms**:
- Empty `migration_events` array in JSON
- No vertical lines on diagnostic plots

**Diagnosis**:
```bash
# Check island model configuration
grep -A 5 "enable_island_model" config.yaml
grep "migration_interval" config.yaml

# Check evolution logs for migration messages
grep -i "migration.*perform" verbose_output.log
```

**Common Causes**:
1. **Island model not enabled**: `enable_island_model: false`
2. **Migration interval too large**: No migrations occurred during run
3. **Diagnostics not enabled**: Migration recording requires diagnostics
4. **Configuration parsing error**: Migration interval not read correctly

**Solutions**:
```yaml
# Ensure proper configuration
genetic_algorithm:
  enable_island_model: true
  island_model_migration_interval: 10  # Reasonable interval
  island_model_migration_rate: 0.1

basic_settings:
  enable_diagnostics: true
```

#### Problem: Migration Events Recorded But Not Visualized

**Symptoms**:
- Migration events present in JSON
- No visual lines on plots

**Diagnosis**:
```bash
# Check migration event count
grep -c "generation.*source_island" diagnostics/diagnostics_data.json

# Verify plot generation timestamp
ls -la diagnostics/*.png
```

**Common Causes**:
1. **Plot caching**: Old plots not regenerated
2. **Visualization error**: Exception during annotation rendering
3. **Generation bounds**: Migration generations outside plot range

**Solutions**:
- Delete existing diagnostic plots to force regeneration
- Check verbose output for visualization errors
- Verify migration generations fall within evolution range

#### Problem: Migration Interval Not Working

**Symptoms**:
- Migrations occur at wrong generations
- Too many/too few migration events

**Root Cause**:
Migration interval configuration not properly read from YAML.

**Critical Fix Required**:
```python
# In genetic/ga_engine.py, ensure this line exists:
if self.use_island_model:
    migration_interval = getattr(self.config.genetic_params, 'island_model_migration_interval', 20)
    self.migration_interval = migration_interval  # This line is critical!
```

**Verification**:
```python
# Add debug logging to verify interval
print(f"Migration interval set to: {self.migration_interval}")
print(f"Migration will occur at generations: {[i for i in range(0, max_gens, self.migration_interval) if i > 0]}")
```

### Advanced Usage

#### Custom Migration Analysis

```python
import json

# Load migration data
with open('diagnostics/diagnostics_data.json', 'r') as f:
    data = json.load(f)

migration_events = data['migration_events']

# Analyze migration patterns
migration_generations = [event['generation'] for event in migration_events]
migration_fitness = [event['migrant_fitness'] for event in migration_events]

print(f"Migrations occurred at generations: {sorted(set(migration_generations))}")
print(f"Average migrant fitness: {sum(migration_fitness)/len(migration_fitness):.4f}")
```

#### Integration with External Tools

Migration event data can be exported for analysis in:
- **R/ggplot2**: Time-series analysis with migration markers
- **Python/Matplotlib**: Custom visualization scripts
- **Excel/CSV**: Statistical analysis and reporting
- **Jupyter Notebooks**: Interactive migration pattern exploration

### Future Enhancements

#### Planned Features

1. **Migration Success Tracking**
   - Track fitness improvements from migrants
   - Success rate analysis by island pair
   - Migration impact visualization

2. **Inter-Island Diversity Analysis**
   - Diversity metrics between islands
   - Migration-driven diversity changes
   - Island specialization visualization

3. **Migration Flow Diagrams**
   - Network diagrams showing migration patterns
   - Island connectivity analysis
   - Migration frequency heatmaps

4. **Adaptive Migration Intervals**
   - Dynamic migration timing based on stagnation
   - Performance-based migration rate adjustment
   - Smart migration scheduling

#### Configuration Extensions

Future configuration options may include:
```yaml
# Advanced migration analysis (planned)
migration_analysis:
  track_migration_success: true
  analyze_inter_island_diversity: true
  enable_migration_flow_diagrams: true
  adaptive_migration_intervals: true
  migration_impact_threshold: 0.01
```

### Research Applications

Migration event annotations are particularly valuable for:

- **Evolutionary Algorithm Research**: Understanding migration timing impact
- **Population Dynamics Studies**: Analyzing gene flow between subpopulations
- **Optimization Performance Analysis**: Migration effects on convergence
- **Multi-Population Algorithm Development**: Design and tuning guidance
- **Academic Publications**: Visual evidence of island model behavior

### Contributing

To extend migration annotations:

1. **Add new plot types**: Extend `_add_simple_migration_lines()` calls
2. **Enhance annotation styles**: Modify visual parameters in helper methods
3. **Add migration metrics**: Extend `record_migration_event()` data collection
4. **Improve performance**: Optimize annotation rendering for large datasets

Report issues or suggest enhancements through the project's issue tracking system.

---

*Last updated: 2025-09-23*
*Implementation version: 0.1.0*
*Diagnostic system version: 0.1.0*