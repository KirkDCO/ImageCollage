from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class FitnessWeights:
    color: float = 0.4
    luminance: float = 0.25
    texture: float = 0.2
    edges: float = 0.15
    
    def __post_init__(self):
        total = self.color + self.luminance + self.texture + self.edges
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Fitness weights must sum to 1.0, got {total}")


@dataclass
class GeneticParams:
    population_size: int = 100
    max_generations: int = 1000
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    elitism_rate: float = 0.1
    tournament_size: int = 5

    # Termination criteria
    convergence_threshold: float = 0.001
    early_stopping_patience: int = 50
    target_fitness: float = 0.0

    # Advanced evolution parameters
    enable_adaptive_parameters: bool = True
    stagnation_threshold: int = 50
    diversity_threshold: float = 0.4
    restart_threshold: int = 50
    restart_ratio: float = 0.3
    enable_advanced_crossover: bool = True
    enable_advanced_mutation: bool = True

    # Comprehensive diversity features
    enable_comprehensive_diversity: bool = True
    enable_spatial_diversity: bool = True
    enable_island_model: bool = False
    island_model_num_islands: int = 4
    island_model_migration_interval: int = 20
    island_model_migration_rate: float = 0.1

    def __post_init__(self):
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("Crossover rate must be between 0.0 and 1.0")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        if not 0.0 <= self.elitism_rate <= 1.0:
            raise ValueError("Elitism rate must be between 0.0 and 1.0")
        if not 0.0 <= self.diversity_threshold <= 1.0:
            raise ValueError("Diversity threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.restart_ratio <= 1.0:
            raise ValueError("Restart ratio must be between 0.0 and 1.0")


@dataclass
class GPUConfig:
    enable_gpu: bool = False
    gpu_devices: List[int] = field(default_factory=lambda: [0])
    gpu_batch_size: int = 256
    gpu_memory_limit_gb: float = 20.0
    auto_mixed_precision: bool = True
    
    def __post_init__(self):
        if self.enable_gpu and not self.gpu_devices:
            self.gpu_devices = [0]


@dataclass
class CollageConfig:
    grid_size: Tuple[int, int] = (50, 50)
    tile_size: Tuple[int, int] = (32, 32)

    genetic_params: GeneticParams = field(default_factory=GeneticParams)
    fitness_weights: FitnessWeights = field(default_factory=FitnessWeights)
    gpu_config: GPUConfig = field(default_factory=GPUConfig)

    max_source_images: int = 10000
    cache_size_mb: int = 1024

    allow_duplicate_tiles: bool = True
    enable_edge_blending: bool = False
    enable_parallel_processing: bool = True
    num_processes: int = 4

    output_quality: int = 95
    preview_frequency: int = 10

    # Reproducibility
    random_seed: int = None

    # Lineage tracking
    enable_lineage_tracking: bool = False
    lineage_output_dir: Optional[str] = None

    # Diagnostics and analysis
    enable_diagnostics: bool = False
    diagnostics_output_dir: Optional[str] = None

    # Checkpoint system
    enable_checkpoints: bool = False
    checkpoint_interval: int = 25
    max_checkpoints: int = 5
    checkpoint_dir: Optional[str] = None

    # Advanced diversity management
    enable_fitness_sharing: bool = False
    fitness_sharing_radius: float = 5.0
    fitness_sharing_alpha: float = 1.0
    enable_crowding_replacement: bool = False
    crowding_factor: float = 2.0

    # Intelligent restart system
    enable_intelligent_restart: bool = False
    restart_diversity_threshold: float = 0.1
    restart_stagnation_threshold: int = 50
    restart_elite_preservation: float = 0.1

    # Real-time diversity dashboard
    enable_diversity_dashboard: bool = False
    dashboard_update_interval: int = 10
    dashboard_alert_critical_diversity: float = 0.1
    dashboard_alert_low_diversity: float = 0.2

    # Fitness component tracking
    enable_component_tracking: bool = False
    track_component_inheritance: bool = False
    
    def update_from_dict(self, params: Dict[str, Any]) -> None:
        """Update configuration from dictionary, handling both flat and organized YAML structures."""

        # Handle organized YAML structure (new format)
        if "basic_settings" in params:
            self._update_from_organized_dict(params)
        else:
            # Handle flat structure (legacy format)
            self._update_from_flat_dict(params)

    def _update_from_organized_dict(self, params: Dict[str, Any]) -> None:
        """Update from organized YAML structure with sections."""

        # Update basic settings
        if "basic_settings" in params:
            basic = params["basic_settings"]
            for key, value in basic.items():
                if key in ["grid_size", "tile_size"] and isinstance(value, list):
                    setattr(self, key, tuple(value))
                elif hasattr(self, key):
                    setattr(self, key, value)

        # Update genetic algorithm parameters
        if "genetic_algorithm" in params:
            ga_params = params["genetic_algorithm"]

            # Extract genetic params for GeneticParams object
            genetic_dict = {}
            for key, value in ga_params.items():
                if not key.startswith("#") and hasattr(GeneticParams, key):
                    genetic_dict[key] = value

            # Update existing genetic_params with new values
            for key, value in genetic_dict.items():
                if hasattr(self.genetic_params, key):
                    setattr(self.genetic_params, key, value)

        # Update fitness evaluation
        if "fitness_evaluation" in params:
            fitness = params["fitness_evaluation"]
            fitness_dict = {}
            for key, value in fitness.items():
                if not key.startswith("#"):
                    # Map weight names to FitnessWeights fields
                    if key == "color_weight":
                        fitness_dict["color"] = value
                    elif key == "luminance_weight":
                        fitness_dict["luminance"] = value
                    elif key == "texture_weight":
                        fitness_dict["texture"] = value
                    elif key == "edges_weight":
                        fitness_dict["edges"] = value

            if fitness_dict:
                # Update existing fitness_weights with new values
                for key, value in fitness_dict.items():
                    if hasattr(self.fitness_weights, key):
                        setattr(self.fitness_weights, key, value)

        # Update GPU acceleration
        if "gpu_acceleration" in params:
            gpu = params["gpu_acceleration"]
            for key, value in gpu.items():
                if not key.startswith("#") and hasattr(self.gpu_config, key):
                    setattr(self.gpu_config, key, value)

        # Update performance settings
        if "performance" in params:
            perf = params["performance"]
            for key, value in perf.items():
                if not key.startswith("#") and hasattr(self, key):
                    setattr(self, key, value)

        # Update output settings
        if "output" in params:
            output = params["output"]
            for key, value in output.items():
                if not key.startswith("#") and hasattr(self, key):
                    setattr(self, key, value)

        # Update checkpoint system settings
        if "checkpoint_system" in params:
            checkpoint = params["checkpoint_system"]
            for key, value in checkpoint.items():
                if not key.startswith("#") and hasattr(self, key):
                    setattr(self, key, value)

        # Update diversity management settings
        if "diversity_management" in params:
            diversity = params["diversity_management"]
            for key, value in diversity.items():
                if not key.startswith("#") and hasattr(self, key):
                    setattr(self, key, value)

        # Update intelligent restart settings
        if "intelligent_restart" in params:
            restart = params["intelligent_restart"]
            for key, value in restart.items():
                if not key.startswith("#") and hasattr(self, key):
                    setattr(self, key, value)

        # Update diversity dashboard settings
        if "diversity_dashboard" in params:
            dashboard = params["diversity_dashboard"]
            for key, value in dashboard.items():
                if not key.startswith("#") and hasattr(self, key):
                    setattr(self, key, value)

        # Update component tracking settings
        if "component_tracking" in params:
            components = params["component_tracking"]
            for key, value in components.items():
                if not key.startswith("#") and hasattr(self, key):
                    setattr(self, key, value)

    def _update_from_flat_dict(self, params: Dict[str, Any]) -> None:
        """Update from flat dictionary structure (legacy format)."""
        for key, value in params.items():
            if key == "fitness_weights" and isinstance(value, dict):
                self.fitness_weights = FitnessWeights(**value)
            elif key == "genetic_params" and isinstance(value, dict):
                self.genetic_params = GeneticParams(**value)
            elif key == "gpu_config" and isinstance(value, dict):
                self.gpu_config = GPUConfig(**value)
            elif key in ["grid_size", "tile_size"] and isinstance(value, list):
                # Convert list back to tuple for internal use
                setattr(self, key, tuple(value))
            elif hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary with organized structure for YAML export."""
        from datetime import datetime

        config_dict = {}

        # Header with metadata
        config_dict["# Image Collage Generator Configuration"] = None
        config_dict["# Generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_dict["# Documentation"] = "https://docs.anthropic.com/claude-code"

        # === BASIC SETTINGS ===
        config_dict["basic_settings"] = {
            "grid_size": list(self.grid_size),  # [width, height] in tiles
            "tile_size": list(self.tile_size),  # [width, height] in pixels
            "allow_duplicate_tiles": self.allow_duplicate_tiles,
            "random_seed": self.random_seed,
            "enable_lineage_tracking": self.enable_lineage_tracking,
            "lineage_output_dir": self.lineage_output_dir,
            "enable_diagnostics": self.enable_diagnostics,
            "diagnostics_output_dir": self.diagnostics_output_dir,
        }

        # === GENETIC ALGORITHM PARAMETERS ===
        genetic_dict = {
            "# Core GA Parameters": None,
            "population_size": self.genetic_params.population_size,
            "max_generations": self.genetic_params.max_generations,
            "crossover_rate": self.genetic_params.crossover_rate,
            "mutation_rate": self.genetic_params.mutation_rate,
            "elitism_rate": self.genetic_params.elitism_rate,
            "tournament_size": self.genetic_params.tournament_size,
            "# Termination Criteria": None,
            "convergence_threshold": self.genetic_params.convergence_threshold,
            "early_stopping_patience": self.genetic_params.early_stopping_patience,
            "target_fitness": self.genetic_params.target_fitness,
        }

        # Add advanced diversity parameters if they exist
        if hasattr(self.genetic_params, 'enable_adaptive_parameters'):
            genetic_dict["# Advanced Evolution Features"] = None
            genetic_dict["enable_adaptive_parameters"] = self.genetic_params.enable_adaptive_parameters
            genetic_dict["enable_advanced_crossover"] = self.genetic_params.enable_advanced_crossover
            genetic_dict["enable_advanced_mutation"] = self.genetic_params.enable_advanced_mutation
            genetic_dict["stagnation_threshold"] = self.genetic_params.stagnation_threshold
            genetic_dict["diversity_threshold"] = self.genetic_params.diversity_threshold
            genetic_dict["restart_threshold"] = self.genetic_params.restart_threshold
            genetic_dict["restart_ratio"] = self.genetic_params.restart_ratio

        # Add comprehensive diversity features if they exist
        if hasattr(self.genetic_params, 'enable_comprehensive_diversity'):
            genetic_dict["# Comprehensive Diversity Management"] = None
            genetic_dict["enable_comprehensive_diversity"] = self.genetic_params.enable_comprehensive_diversity
            genetic_dict["enable_spatial_diversity"] = self.genetic_params.enable_spatial_diversity

        # Add island model parameters if they exist
        if hasattr(self.genetic_params, 'enable_island_model'):
            genetic_dict["# Multi-Population Island Model"] = None
            genetic_dict["enable_island_model"] = self.genetic_params.enable_island_model
            genetic_dict["island_model_num_islands"] = self.genetic_params.island_model_num_islands
            genetic_dict["island_model_migration_interval"] = self.genetic_params.island_model_migration_interval
            genetic_dict["island_model_migration_rate"] = self.genetic_params.island_model_migration_rate

        config_dict["genetic_algorithm"] = genetic_dict

        # === FITNESS EVALUATION ===
        config_dict["fitness_evaluation"] = {
            "# Fitness component weights (must sum to 1.0)": None,
            "color_weight": self.fitness_weights.color,
            "luminance_weight": self.fitness_weights.luminance,
            "texture_weight": self.fitness_weights.texture,
            "edges_weight": self.fitness_weights.edges,
            "# Total weight check": f"Sum = {self.fitness_weights.color + self.fitness_weights.luminance + self.fitness_weights.texture + self.fitness_weights.edges}",
        }

        # === GPU ACCELERATION ===
        config_dict["gpu_acceleration"] = {
            "# GPU settings for CUDA acceleration": None,
            "enable_gpu": self.gpu_config.enable_gpu,
            "gpu_devices": self.gpu_config.gpu_devices,
            "gpu_batch_size": self.gpu_config.gpu_batch_size,
            "gpu_memory_limit_gb": self.gpu_config.gpu_memory_limit_gb,
            "auto_mixed_precision": self.gpu_config.auto_mixed_precision,
        }

        # === PERFORMANCE SETTINGS ===
        config_dict["performance"] = {
            "# Processing and optimization settings": None,
            "enable_parallel_processing": self.enable_parallel_processing,
            "num_processes": self.num_processes,
            "cache_size_mb": self.cache_size_mb,
            "max_source_images": self.max_source_images,
            "preview_frequency": self.preview_frequency,
        }

        # === OUTPUT SETTINGS ===
        config_dict["output"] = {
            "# Image output and quality settings": None,
            "output_quality": self.output_quality,
            "enable_edge_blending": self.enable_edge_blending,
        }

        # === CHECKPOINT SYSTEM ===
        config_dict["checkpoint_system"] = {
            "# Crash recovery and resuming settings": None,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_interval": self.checkpoint_interval,
            "max_checkpoints": self.max_checkpoints,
            "checkpoint_dir": self.checkpoint_dir,
        }

        # === ADVANCED DIVERSITY MANAGEMENT ===
        config_dict["diversity_management"] = {
            "# Advanced diversity preservation techniques": None,
            "enable_fitness_sharing": self.enable_fitness_sharing,
            "fitness_sharing_radius": self.fitness_sharing_radius,
            "fitness_sharing_alpha": self.fitness_sharing_alpha,
            "enable_crowding_replacement": self.enable_crowding_replacement,
            "crowding_factor": self.crowding_factor,
        }

        # === INTELLIGENT RESTART SYSTEM ===
        config_dict["intelligent_restart"] = {
            "# Automatic population restart for stagnation prevention": None,
            "enable_intelligent_restart": self.enable_intelligent_restart,
            "restart_diversity_threshold": self.restart_diversity_threshold,
            "restart_stagnation_threshold": self.restart_stagnation_threshold,
            "restart_elite_preservation": self.restart_elite_preservation,
        }

        # === DIVERSITY DASHBOARD ===
        config_dict["diversity_dashboard"] = {
            "# Real-time diversity monitoring and alerts": None,
            "enable_diversity_dashboard": self.enable_diversity_dashboard,
            "dashboard_update_interval": self.dashboard_update_interval,
            "dashboard_alert_critical_diversity": self.dashboard_alert_critical_diversity,
            "dashboard_alert_low_diversity": self.dashboard_alert_low_diversity,
        }

        # === COMPONENT TRACKING ===
        config_dict["component_tracking"] = {
            "# Fitness component evolution analysis": None,
            "enable_component_tracking": self.enable_component_tracking,
            "track_component_inheritance": self.track_component_inheritance,
        }

        return config_dict

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to YAML file with organized structure and comments."""
        path = Path(file_path)
        config_dict = self.to_dict()

        with open(path, 'w') as f:
            # Write organized YAML with preserved structure
            self._write_organized_yaml(config_dict, f)

    def _write_organized_yaml(self, config_dict: Dict[str, Any], file) -> None:
        """Write YAML with custom formatting and preserved comments."""

        # Write header comments
        if "# Image Collage Generator Configuration" in config_dict:
            file.write("# Image Collage Generator Configuration\n")
            file.write(f"# Generated: {config_dict['# Generated']}\n")
            file.write(f"# Documentation: {config_dict['# Documentation']}\n\n")

        # Write each main section
        sections = [
            ("basic_settings", "BASIC SETTINGS"),
            ("genetic_algorithm", "GENETIC ALGORITHM PARAMETERS"),
            ("fitness_evaluation", "FITNESS EVALUATION"),
            ("gpu_acceleration", "GPU ACCELERATION"),
            ("performance", "PERFORMANCE SETTINGS"),
            ("output", "OUTPUT SETTINGS"),
            ("checkpoint_system", "CHECKPOINT SYSTEM"),
            ("diversity_management", "ADVANCED DIVERSITY MANAGEMENT"),
            ("intelligent_restart", "INTELLIGENT RESTART SYSTEM"),
            ("diversity_dashboard", "DIVERSITY DASHBOARD"),
            ("component_tracking", "COMPONENT TRACKING")
        ]

        for section_key, section_title in sections:
            if section_key in config_dict:
                file.write(f"# === {section_title} ===\n")
                file.write(f"{section_key}:\n")

                section_data = config_dict[section_key]
                for key, value in section_data.items():
                    if key.startswith("#"):
                        # Write comment
                        if value is None:
                            file.write(f"  {key}\n")
                        else:
                            file.write(f"  # {value}\n")
                    else:
                        # Write data
                        if isinstance(value, list):
                            file.write(f"  {key}: {value}\n")
                        elif isinstance(value, str):
                            file.write(f"  {key}: \"{value}\"\n")
                        else:
                            file.write(f"  {key}: {value}\n")

                file.write("\n")

    @classmethod
    def load_from_file(cls, file_path: str) -> 'CollageConfig':
        """Load configuration from YAML file."""
        path = Path(file_path)

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = cls()
        config.update_from_dict(config_dict)
        return config


class PresetConfigs:
    @staticmethod
    def quick() -> CollageConfig:
        config = CollageConfig()
        config.grid_size = (20, 20)
        config.genetic_params.population_size = 50
        config.fitness_weights = FitnessWeights(color=0.6, luminance=0.4, texture=0.0, edges=0.0)
        return config

    @staticmethod
    def demo() -> CollageConfig:
        """Ultra-fast demo configuration for testing and demonstration."""
        config = CollageConfig()
        config.grid_size = (15, 20)  # 300 tiles - good for demo (width, height)
        config.genetic_params.max_generations = 30
        config.genetic_params.population_size = 30
        config.genetic_params.mutation_rate = 0.1  # Higher for faster evolution
        config.genetic_params.enable_adaptive_parameters = True
        config.genetic_params.enable_advanced_crossover = True
        config.genetic_params.enable_advanced_mutation = True
        config.genetic_params.restart_threshold = 20  # Shorter for demo
        config.genetic_params.stagnation_threshold = 15
        config.fitness_weights = FitnessWeights(color=0.7, luminance=0.3, texture=0.0, edges=0.0)
        config.genetic_params.early_stopping_patience = 10  # Stop early if converged
        config.genetic_params.convergence_threshold = 0.01  # Less strict convergence
        return config
    
    @staticmethod
    def balanced() -> CollageConfig:
        return CollageConfig()
    
    @staticmethod
    def high() -> CollageConfig:
        config = CollageConfig()
        config.grid_size = (100, 100)
        config.genetic_params.max_generations = 1500
        config.genetic_params.population_size = 200
        config.genetic_params.convergence_threshold = 0.0005
        config.genetic_params.early_stopping_patience = 100
        return config
    
    @staticmethod
    def gpu() -> CollageConfig:
        config = CollageConfig()
        config.grid_size = (150, 150)
        config.genetic_params.max_generations = 3000
        config.genetic_params.population_size = 300
        config.gpu_config.enable_gpu = True
        config.gpu_config.gpu_devices = [0, 1]  # Use both GPUs
        config.gpu_config.gpu_batch_size = 512
        config.genetic_params.convergence_threshold = 0.0001
        config.genetic_params.early_stopping_patience = 150
        return config
    
    @staticmethod
    def extreme() -> CollageConfig:
        config = CollageConfig()
        config.grid_size = (300, 300)  # 90,000 tiles
        config.genetic_params.max_generations = 5000
        config.genetic_params.population_size = 500
        config.gpu_config.enable_gpu = True
        config.gpu_config.gpu_devices = [0, 1]
        config.gpu_config.gpu_batch_size = 1024
        config.genetic_params.convergence_threshold = 0.00005
        config.genetic_params.early_stopping_patience = 250
        return config

    @staticmethod
    def advanced() -> CollageConfig:
        """Advanced evolution configuration for preventing convergence and maintaining diversity."""
        config = CollageConfig()
        config.grid_size = (60, 60)
        config.genetic_params.max_generations = 1500
        config.genetic_params.population_size = 150
        config.genetic_params.mutation_rate = 0.08
        config.genetic_params.crossover_rate = 0.85
        config.genetic_params.elitism_rate = 0.12
        config.genetic_params.enable_adaptive_parameters = True
        config.genetic_params.enable_advanced_crossover = True
        config.genetic_params.enable_advanced_mutation = True
        config.genetic_params.restart_threshold = 40
        config.genetic_params.stagnation_threshold = 30
        config.genetic_params.diversity_threshold = 0.5  # Higher diversity requirement
        config.genetic_params.restart_ratio = 0.4  # More aggressive restart
        config.genetic_params.early_stopping_patience = 80
        config.genetic_params.convergence_threshold = 0.001
        return config

    @staticmethod
    def ultra() -> CollageConfig:
        """Ultra-advanced configuration with all DIVERSITY.md techniques implemented."""
        config = CollageConfig()
        config.grid_size = (80, 80)  # 6,400 tiles - challenging but manageable
        config.genetic_params.max_generations = 2000
        config.genetic_params.population_size = 200
        config.genetic_params.mutation_rate = 0.06
        config.genetic_params.crossover_rate = 0.85
        config.genetic_params.elitism_rate = 0.10

        # Enable ALL advanced diversity features
        config.genetic_params.enable_adaptive_parameters = True
        config.genetic_params.enable_advanced_crossover = True
        config.genetic_params.enable_advanced_mutation = True
        config.genetic_params.enable_comprehensive_diversity = True
        config.genetic_params.enable_spatial_diversity = True

        # Multi-population island model
        config.genetic_params.enable_island_model = True
        config.genetic_params.island_model_num_islands = 6
        config.genetic_params.island_model_migration_interval = 15
        config.genetic_params.island_model_migration_rate = 0.15

        # Aggressive diversity preservation
        config.genetic_params.diversity_threshold = 0.6  # Higher requirement
        config.genetic_params.restart_threshold = 30    # Faster restart
        config.genetic_params.restart_ratio = 0.5       # More aggressive restart

        config.genetic_params.early_stopping_patience = 100
        config.genetic_params.convergence_threshold = 0.0005
        return config

    @staticmethod
    def get_preset(name: str) -> CollageConfig:
        presets = {
            "demo": PresetConfigs.demo(),
            "quick": PresetConfigs.quick(),
            "balanced": PresetConfigs.balanced(),
            "high": PresetConfigs.high(),
            "advanced": PresetConfigs.advanced(),
            "ultra": PresetConfigs.ultra(),
            "gpu": PresetConfigs.gpu(),
            "extreme": PresetConfigs.extreme(),
        }
        
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        
        return presets[name]