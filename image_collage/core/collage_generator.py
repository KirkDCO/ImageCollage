from typing import Optional, Callable, Dict, Any, List
import os
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from ..preprocessing.image_processor import ImageProcessor
from ..genetic.ga_engine import GeneticAlgorithmEngine
from ..fitness.evaluator import FitnessEvaluator
from ..rendering.renderer import Renderer
from ..cache.manager import CacheManager
from ..config.settings import CollageConfig

try:
    from ..fitness.gpu_evaluator import GPUFitnessEvaluator
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Move diagnostics imports inside functions to avoid circular import
DIAGNOSTICS_AVAILABLE = True  # Assume available, check at runtime

try:
    from ..checkpoints.manager import CheckpointManager
    from ..checkpoints.state import EvolutionState
    CHECKPOINTS_AVAILABLE = True
except ImportError:
    CHECKPOINTS_AVAILABLE = False

try:
    from ..genetic.diversity_dashboard import DiversityDashboard, DashboardConfig
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


@dataclass
class CollageResult:
    """Result from collage generation with comprehensive output data.

    Attributes:
        collage_image: Final rendered collage as numpy array
        fitness_score: Best fitness score achieved
        generations_used: Number of generations completed
        processing_time: Total processing time in seconds
        best_individual: Best genetic individual (grid arrangement)
        fitness_history: List of best fitness scores per generation
        evolution_frames: Optional list of preview images for animation
        evolution_generation_numbers: Corresponding generation numbers for frames
    """
    collage_image: np.ndarray
    fitness_score: float
    generations_used: int
    processing_time: float
    best_individual: np.ndarray
    fitness_history: List[float]
    evolution_frames: Optional[List[np.ndarray]] = None
    evolution_generation_numbers: Optional[List[int]] = None


class CollageGenerator:
    def __init__(self, config: Optional[CollageConfig] = None):
        self.config = config or CollageConfig()
        self.target_image: Optional[np.ndarray] = None
        self.source_images: List[np.ndarray] = []
        self.source_features: List[Dict[str, Any]] = []
        
        self.image_processor = ImageProcessor(self.config)
        self.cache_manager = CacheManager(self.config.cache_size_mb)
        
        # Initialize appropriate fitness evaluator
        if self.config.gpu_config.enable_gpu and GPU_AVAILABLE:
            try:
                self.fitness_evaluator = GPUFitnessEvaluator(
                    self.config, 
                    gpu_devices=self.config.gpu_config.gpu_devices
                )
                # Update config with actual devices used (may have changed due to fallback)
                actual_devices = self.fitness_evaluator.gpu_devices
                self.config.gpu_config.gpu_devices = actual_devices
                print(f"GPU acceleration enabled on devices: {actual_devices}")
            except Exception as e:
                print(f"GPU initialization failed, falling back to CPU: {e}")
                self.fitness_evaluator = FitnessEvaluator(self.config)
        else:
            self.fitness_evaluator = FitnessEvaluator(self.config)
            
        self.ga_engine = GeneticAlgorithmEngine(self.config)
        self.renderer = Renderer(self.config)
        
    def load_target(self, image_path: str) -> bool:
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Target image not found: {image_path}")
            
            self.target_image = self.image_processor.load_image(str(image_path))
            
            if self.target_image is None:
                return False
                
            target_features = self.image_processor.extract_features(self.target_image)
            self.fitness_evaluator.set_target(self.target_image, target_features)
            
            return True
            
        except Exception as e:
            print(f"Error loading target image: {e}")
            return False
    
    def load_sources(self, directory_path: str) -> int:
        try:
            directory_path = Path(directory_path)
            if not directory_path.exists():
                raise FileNotFoundError(f"Source directory not found: {directory_path}")
            
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = set()  # Use set to avoid duplicates

            # Handle all case variations of file extensions - case insensitive search
            for ext in supported_formats:
                # Find all files ending with this extension (case insensitive)
                for file_path in directory_path.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() == ext:
                        image_files.add(file_path)

            image_files = list(image_files)  # Convert back to list
            image_files.sort()  # Ensure consistent order across runs for deterministic fitness
            
            self.source_images = []
            self.source_features = []
            
            loaded_count = 0
            for image_file in image_files:
                cache_key = f"source_{image_file.name}_{image_file.stat().st_mtime}"
                
                cached_data = self.cache_manager.get(cache_key)
                if cached_data:
                    image, features = cached_data
                else:
                    image = self.image_processor.load_image(str(image_file))
                    if image is not None:
                        image = self.image_processor.normalize_to_tile_size(image)
                        features = self.image_processor.extract_features(image)
                        self.cache_manager.put(cache_key, (image, features))
                    else:
                        continue
                
                self.source_images.append(image)
                self.source_features.append(features)
                loaded_count += 1
                
                if loaded_count >= self.config.max_source_images:
                    break
            
            return loaded_count
            
        except Exception as e:
            print(f"Error loading source images: {e}")
            return 0
    
    def configure_algorithm(self, params: Dict[str, Any]) -> None:
        self.config.update_from_dict(params)
        
        self.ga_engine.update_config(self.config)
        self.fitness_evaluator.update_config(self.config)
        self.renderer.update_config(self.config)
    
    def generate(self, callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
                 save_evolution: bool = False, evolution_interval: int = 10,
                 diagnostics_folder: Optional[str] = None,
                 output_folder: Optional[str] = None,
                 save_checkpoints: bool = False,
                 checkpoint_interval: int = 10) -> CollageResult:
        """Generate photomosaic collage using genetic algorithm.

        Args:
            callback: Optional progress callback function(generation, fitness, preview)
            save_evolution: Whether to save evolution frames for animation
            evolution_interval: Generation interval for saving evolution frames
            diagnostics_folder: Path to save 8+ enhanced diagnostic reports
            output_folder: Path to save configuration and other outputs

        Returns:
            CollageResult: Complete results including collage, metrics, and optional
                         evolution frames with generation numbers for animation

        Features:
            - Enhanced diagnostics with proper alignment and accurate metrics
            - Evolution animation frames with generation number tracking
            - GPU acceleration with multi-device support
            - Comprehensive performance analysis
        """
        if self.target_image is None:
            raise ValueError("Target image not loaded")
        
        if not self.source_images:
            raise ValueError("No source images loaded")
        
        start_time = self._get_time()

        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            import random
            random.seed(self.config.random_seed)

        # Save configuration early so users can see what settings will be used
        if output_folder:
            try:
                config_path = Path(output_folder) / "config.yaml"
                self.config.save_to_file(str(config_path))
                print(f"Configuration saved to: {config_path}")
            except Exception as e:
                print(f"Warning: Failed to save configuration: {e}")

        # Initialize diagnostics if requested
        diagnostics_collector = None
        if diagnostics_folder:
            # Create diagnostics directory early for consistency
            try:
                Path(diagnostics_folder).mkdir(parents=True, exist_ok=True)
                print(f"Diagnostics directory created: {diagnostics_folder}")
            except Exception as e:
                print(f"Warning: Failed to create diagnostics directory: {e}")

            try:
                from ..diagnostics import DiagnosticsCollector
                diagnostics_collector = DiagnosticsCollector(self.config)
                diagnostics_collector.start_evolution()

                # Set diagnostics collector in GA engine for migration tracking
                self.ga_engine.set_diagnostics_collector(diagnostics_collector)
            except ImportError as e:
                print(f"Warning: Diagnostics not available: {e}")
                diagnostics_collector = None

        # Initialize lineage tracking if requested
        lineage_tracker = None
        lineage_visualizer = None
        if self.config.enable_lineage_tracking:
            try:
                from ..lineage import LineageTracker, LineageVisualizer
                from ..lineage.fitness_components import FitnessComponentTracker

                # Create lineage output directory - prioritize config setting
                if self.config.lineage_output_dir:
                    if output_folder:
                        # Use configured name within output folder
                        lineage_dir = Path(output_folder) / self.config.lineage_output_dir
                    else:
                        # Use configured path as-is
                        lineage_dir = Path(self.config.lineage_output_dir)
                elif output_folder:
                    # Default to "lineage" within output folder
                    lineage_dir = Path(output_folder) / "lineage"
                else:
                    lineage_dir = Path("lineage_output")

                lineage_tracker = LineageTracker(str(lineage_dir))

                # Create fitness component tracker if enabled
                fitness_component_tracker = None
                if self.config.enable_component_tracking:
                    fitness_component_tracker = FitnessComponentTracker(['color', 'luminance', 'texture', 'edges'])

                lineage_visualizer = LineageVisualizer(lineage_tracker, str(lineage_dir), fitness_component_tracker)

                # Connect lineage tracker to GA engine
                self.ga_engine.set_lineage_tracker(lineage_tracker)
                print(f"Lineage tracking enabled - output: {lineage_dir}")

            except ImportError as e:
                print(f"Warning: Lineage tracking not available: {e}")
                self.config.enable_lineage_tracking = False

        self.ga_engine.initialize_population(len(self.source_images))

        # Initialize checkpoint manager if requested
        checkpoint_manager = None
        if save_checkpoints and CHECKPOINTS_AVAILABLE and output_folder:
            try:
                checkpoint_dir = Path(output_folder) / "checkpoints"
                checkpoint_manager = CheckpointManager(
                    str(checkpoint_dir),
                    save_interval=checkpoint_interval,
                    max_checkpoints=self.config.max_checkpoints
                )
                print(f"Checkpoint saving enabled: {checkpoint_dir}")
            except Exception as e:
                print(f"Warning: Failed to initialize checkpoint manager: {e}")
                save_checkpoints = False

        # Initialize diversity dashboard if enabled
        diversity_dashboard = None
        if self.config.enable_diversity_dashboard and DASHBOARD_AVAILABLE:
            try:
                dashboard_config = DashboardConfig(
                    update_interval=self.config.dashboard_update_interval,
                    alert_thresholds={
                        'critical_diversity': self.config.dashboard_alert_critical_diversity,
                        'low_diversity': self.config.dashboard_alert_low_diversity,
                        'stagnation_generations': 30,
                        'high_selection_pressure': 0.9,
                        'low_fitness_variance': 0.001
                    }
                )
                dashboard_output_dir = output_folder if output_folder else None
                diversity_dashboard = DiversityDashboard(dashboard_config, dashboard_output_dir)
            except Exception as e:
                print(f"Warning: Failed to initialize diversity dashboard: {e}")
                self.config.enable_diversity_dashboard = False

        # Initialize lineage tracking with initial population
        individual_ids = None
        if lineage_tracker:
            population = self.ga_engine.get_population()
            # Get initial fitness scores
            initial_fitness = []
            for individual in population:
                fitness = self.fitness_evaluator.evaluate(
                    individual, self.source_images, self.source_features
                )
                initial_fitness.append(fitness)

            individual_ids = lineage_tracker.initialize_population(population, initial_fitness)

        best_fitness = float('inf')
        best_individual = None
        fitness_history = []
        generations_without_improvement = 0
        evolution_frames = [] if save_evolution else None
        evolution_generation_numbers = [] if save_evolution else None
        
        for generation in range(self.config.genetic_params.max_generations):
            # Start generation tracking for diagnostics
            if diagnostics_collector:
                diagnostics_collector.start_generation(generation)

            population = self.ga_engine.get_population()
            
            # Use batch processing for GPU multi-device evaluation
            if (hasattr(self.fitness_evaluator, 'evaluate_population_batch') and 
                hasattr(self.fitness_evaluator, 'gpu_devices') and 
                len(self.fitness_evaluator.gpu_devices) > 1):
                
                fitness_scores = self.fitness_evaluator.evaluate_population_batch(
                    population, self.source_images, self.source_features
                )
            else:
                # Standard individual evaluation
                fitness_scores = []
                for individual in population:
                    fitness = self.fitness_evaluator.evaluate(
                        individual, self.source_images, self.source_features
                    )
                    fitness_scores.append(fitness)
            
            current_best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            current_best_individual = population[current_best_idx].copy()
            
            if current_best_fitness < best_fitness:
                improvement = best_fitness - current_best_fitness
                best_fitness = current_best_fitness
                best_individual = current_best_individual.copy()

                # Only reset counter if improvement is significant
                # Small improvements count toward early stopping
                if improvement >= self.config.genetic_params.convergence_threshold:
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
            else:
                generations_without_improvement += 1
            
            fitness_history.append(best_fitness)

            # Record generation data for diagnostics
            if diagnostics_collector:
                diagnostics_collector.record_generation(population, fitness_scores)

            # Update diversity dashboard if enabled
            if diversity_dashboard:
                # Gather diversity metrics from diagnostics collector
                diversity_metrics = {}
                if diagnostics_collector and hasattr(diagnostics_collector, 'data') and diagnostics_collector.data.generations:
                    latest_gen = diagnostics_collector.data.generations[-1]

                    # Calculate unique individuals directly (override if diagnostics shows 0)
                    fitness_array = np.array(fitness_scores)
                    from ..cli.helpers import calculate_unique_individuals
                    unique_individuals = calculate_unique_individuals(population)
                    unique_ratio = unique_individuals / len(population) if population else 0.0

                    # Use max of diagnostics fitness_variance and calculated variance
                    calculated_variance = np.var(fitness_array)

                    diversity_metrics = {
                        'normalized_diversity': getattr(latest_gen, 'normalized_diversity', 0.5),
                        'hamming_distance_avg': getattr(latest_gen, 'hamming_distance_avg', 0.5),
                        'position_wise_entropy': getattr(latest_gen, 'position_wise_entropy', 1.0),
                        'fitness_variance': max(latest_gen.fitness_variance, calculated_variance),
                        'spatial_diversity_score': latest_gen.spatial_diversity_score,
                        'unique_individuals_ratio': max(latest_gen.unique_individuals_ratio, unique_ratio),
                        'fitness_range': latest_gen.worst_fitness - latest_gen.best_fitness
                    }
                else:
                    # Fallback basic diversity calculation
                    fitness_array = np.array(fitness_scores)
                    from ..cli.helpers import calculate_unique_individuals
                    from ..utils.diversity_metrics import calculate_hamming_distance_average, calculate_position_wise_entropy

                    unique_individuals = calculate_unique_individuals(population)
                    unique_ratio = unique_individuals / len(population) if population else 0.0

                    # Calculate actual diversity metrics
                    hamming_diversity = calculate_hamming_distance_average(population)
                    position_entropy = calculate_position_wise_entropy(population)

                    diversity_metrics = {
                        'normalized_diversity': unique_ratio,
                        'hamming_distance_avg': hamming_diversity,
                        'position_wise_entropy': position_entropy,
                        'fitness_variance': np.var(fitness_array),
                        'spatial_diversity_score': hamming_diversity,  # Use hamming as spatial proxy
                        'unique_individuals_ratio': unique_individuals / len(population) if population else 0.0,
                        'fitness_range': np.max(fitness_array) - np.min(fitness_array) if len(fitness_array) > 0 else 0.0
                    }

                # Gather population state information
                fitness_array = np.array(fitness_scores)
                population_state = {
                    'population_size': len(population),
                    'selection_pressure': np.std(fitness_array) / np.mean(fitness_array) if np.mean(fitness_array) != 0 else 0.0,
                    'best_fitness': best_fitness,
                    'generations_without_improvement': generations_without_improvement
                }

                # Update dashboard
                alerts = diversity_dashboard.update_dashboard(generation, diversity_metrics, population_state)

            # Save evolution frame if requested
            if save_evolution and generation % evolution_interval == 0:
                frame_collage = self.renderer.render_preview(
                    best_individual, self.source_images
                )
                evolution_frames.append(frame_collage)
                evolution_generation_numbers.append(generation)

            if callback:
                preview_collage = self.renderer.render_preview(
                    best_individual, self.source_images
                )
                callback(generation, best_fitness, preview_collage)
            
            if generations_without_improvement >= self.config.genetic_params.early_stopping_patience:
                break
            
            if best_fitness <= self.config.genetic_params.target_fitness:
                break

            # Track lineage before evolution
            if lineage_tracker and individual_ids:
                # Calculate diversity and selection pressure for lineage tracking
                try:
                    diversity_score = self.ga_engine.get_population_diversity()
                except Exception as e:
                    # Calculate diversity manually as fallback
                    from ..cli.helpers import calculate_unique_individuals
                    unique_individuals = calculate_unique_individuals(population)
                    from ..cli.helpers import calculate_population_diversity
                    diversity_score = calculate_population_diversity(population)

                # Calculate selection pressure using standardized definition
                from ..cli.helpers import calculate_selection_pressure
                selection_pressure = calculate_selection_pressure(fitness_scores)

                lineage_tracker.track_generation(
                    population, fitness_scores, individual_ids,
                    diversity_score, selection_pressure
                )

            # Save checkpoint if requested and at checkpoint interval
            if checkpoint_manager and checkpoint_manager.should_save_checkpoint(generation):
                try:
                    # Create evolution state
                    evolution_state = EvolutionState.create_from_current_state(
                        generation=generation,
                        population=population,
                        fitness_scores=fitness_scores,
                        best_individual=best_individual,
                        best_fitness=best_fitness,
                        fitness_history=fitness_history,
                        generations_without_improvement=generations_without_improvement,
                        start_time=start_time,
                        config_dict=self.config.to_dict(),
                        evolution_frames=evolution_frames,
                        evolution_generation_numbers=evolution_generation_numbers,
                        ga_engine_state=self.ga_engine.get_state() if hasattr(self.ga_engine, 'get_state') else {},
                        diagnostics_state=diagnostics_collector.get_state() if diagnostics_collector and hasattr(diagnostics_collector, 'get_state') else None,
                        lineage_state=lineage_tracker.get_state() if lineage_tracker and hasattr(lineage_tracker, 'get_state') else None
                    )

                    checkpoint_path = checkpoint_manager.save_checkpoint(evolution_state)
                    print(f"Checkpoint saved at generation {generation}: {checkpoint_path}")

                except Exception as e:
                    print(f"Warning: Failed to save checkpoint at generation {generation}: {e}")

            self.ga_engine.evolve_population(fitness_scores)

            # Update individual IDs after evolution (simplified - in a full implementation
            # we would track specific crossover and mutation operations)
            if lineage_tracker:
                new_population = self.ga_engine.get_population()
                individual_ids = [f"gen_{generation+1}_ind_{i}" for i in range(len(new_population))]
        
        final_collage = self.renderer.render_final(best_individual, self.source_images)
        processing_time = self._get_time() - start_time
        
        # Generate diagnostics if requested
        if diagnostics_collector:
            diagnostics_collector.finish_evolution()
            diagnostics_collector.save_to_folder(diagnostics_folder)

            # Create visualizations
            try:
                from ..diagnostics import DiagnosticsVisualizer
                visualizer = DiagnosticsVisualizer()
                visualizer.create_full_report(diagnostics_collector, diagnostics_folder)
                print(f"Diagnostics saved to: {diagnostics_folder}")
            except Exception as e:
                print(f"Warning: Failed to create diagnostics visualizations: {e}")
                import traceback

        # Generate lineage analysis if requested
        if lineage_tracker and lineage_visualizer:
            try:
                # Export lineage data
                lineage_tracker.export_data()

                # Generate all lineage visualizations
                lineage_visualizer.generate_all_plots()

                print(f"Lineage analysis saved to: {lineage_tracker.output_dir}")
            except Exception as e:
                print(f"Warning: Failed to create lineage analysis: {e}")
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")

        # Export dashboard data if available
        if diversity_dashboard and output_folder:
            try:
                dashboard_export_path = Path(output_folder) / "diversity_dashboard_data.json"
                diversity_dashboard.export_dashboard_data(str(dashboard_export_path))
                print(f"Dashboard data exported to: {dashboard_export_path}")
            except Exception as e:
                print(f"Warning: Failed to export dashboard data: {e}")

        # Clean up GPU memory if using GPU
        if hasattr(self.fitness_evaluator, 'clear_gpu_cache'):
            self.fitness_evaluator.clear_gpu_cache()

        return CollageResult(
            collage_image=final_collage,
            fitness_score=best_fitness,
            generations_used=generation + 1,
            processing_time=processing_time,
            best_individual=best_individual,
            fitness_history=fitness_history,
            evolution_frames=evolution_frames,
            evolution_generation_numbers=evolution_generation_numbers
        )

    def resume_from_checkpoint(self, checkpoint_path: str,
                               callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
                               save_evolution: bool = False, evolution_interval: int = 10,
                               diagnostics_folder: Optional[str] = None,
                               output_folder: Optional[str] = None,
                               save_checkpoints: bool = True,
                               checkpoint_interval: int = 10) -> CollageResult:
        """Resume evolution from a saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file to resume from
            callback: Optional progress callback function(generation, fitness, preview)
            save_evolution: Whether to save evolution frames for animation
            evolution_interval: Generation interval for saving evolution frames
            diagnostics_folder: Path to save enhanced diagnostic reports
            output_folder: Output folder for saving configuration and checkpoints
            save_checkpoints: Whether to continue saving checkpoints
            checkpoint_interval: Generation interval for saving checkpoints

        Returns:
            CollageResult with evolution continued from checkpoint
        """
        if not CHECKPOINTS_AVAILABLE:
            raise ImportError("Checkpoint functionality not available. Install required dependencies.")

        print(f"Resuming evolution from checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint_manager = CheckpointManager(str(Path(checkpoint_path).parent))
        evolution_state, metadata = checkpoint_manager.load_checkpoint(checkpoint_path)

        # Set the resume generation for safe checkpoint management
        checkpoint_manager.set_resume_generation(evolution_state.generation)

        # Update configuration from checkpoint
        self.config.update_from_dict(evolution_state.config_dict)

        # Reconfigure components with loaded config
        self.ga_engine.update_config(self.config)
        self.fitness_evaluator.update_config(self.config)
        self.renderer.update_config(self.config)

        # Ensure GA engine knows about the number of source images
        if self.source_images:
            self.ga_engine.num_source_images = len(self.source_images)

        # Restore population and fitness state
        self.ga_engine.set_population(evolution_state.population)

        # Restore GA engine internal state
        if evolution_state.ga_engine_state:
            self.ga_engine.set_state(evolution_state.ga_engine_state)

        # Resume evolution from checkpoint state
        start_generation = evolution_state.generation + 1
        best_individual = evolution_state.best_individual
        best_fitness = evolution_state.best_fitness
        fitness_history = evolution_state.fitness_history.copy()
        generations_without_improvement = evolution_state.generations_without_improvement

        # Restore evolution frames if they exist
        evolution_frames = evolution_state.evolution_frames.copy() if evolution_state.evolution_frames else ([] if save_evolution else None)
        evolution_generation_numbers = evolution_state.evolution_generation_numbers.copy() if evolution_state.evolution_generation_numbers else ([] if save_evolution else None)

        # Initialize all analysis components
        diagnostics_collector, lineage_tracker, lineage_visualizer, individual_ids, diversity_dashboard = self._initialize_analysis_components(
            output_folder, diagnostics_folder, evolution_state
        )

        # Continue with standard generation loop
        for generation in range(start_generation, self.config.genetic_params.max_generations):
            population = self.ga_engine.get_population()

            # Use stored fitness scores for first generation after resume, then re-evaluate
            if generation == start_generation and hasattr(evolution_state, 'fitness_scores') and evolution_state.fitness_scores:
                # Use restored fitness scores from checkpoint
                fitness_scores = evolution_state.fitness_scores.copy()
                print(f"Using restored fitness scores from checkpoint (generation {evolution_state.generation})")
            else:
                # Standard fitness evaluation for subsequent generations
                fitness_scores = []
                for individual in population:
                    fitness = self.fitness_evaluator.evaluate(
                        individual, self.source_images, self.source_features
                    )
                    fitness_scores.append(fitness)

            current_best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            current_best_individual = population[current_best_idx].copy()

            if current_best_fitness < best_fitness:
                improvement = best_fitness - current_best_fitness
                best_fitness = current_best_fitness
                best_individual = current_best_individual.copy()

                # Only reset counter if improvement is significant
                # Small improvements count toward early stopping
                if improvement >= self.config.genetic_params.convergence_threshold:
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
            else:
                generations_without_improvement += 1

            fitness_history.append(best_fitness)

            # Save evolution frame if requested
            if save_evolution and generation % evolution_interval == 0:
                frame_collage = self.renderer.render_preview(
                    best_individual, self.source_images
                )
                evolution_frames.append(frame_collage)
                evolution_generation_numbers.append(generation)

            # Progress callback
            if callback:
                preview_collage = self.renderer.render_preview(best_individual, self.source_images)
                callback(generation, best_fitness, preview_collage)

            # Save checkpoint if requested
            if save_checkpoints and checkpoint_manager.should_save_checkpoint(generation):
                try:
                    evolution_state_current = EvolutionState.create_from_current_state(
                        generation=generation,
                        population=population,
                        fitness_scores=fitness_scores,
                        best_individual=best_individual,
                        best_fitness=best_fitness,
                        fitness_history=fitness_history,
                        generations_without_improvement=generations_without_improvement,
                        start_time=evolution_state.start_time,  # Keep original start time
                        config_dict=self.config.to_dict(),
                        evolution_frames=evolution_frames,
                        evolution_generation_numbers=evolution_generation_numbers,
                        ga_engine_state=self.ga_engine.get_state() if hasattr(self.ga_engine, 'get_state') else {},
                        diagnostics_state=diagnostics_collector.get_state() if diagnostics_collector and hasattr(diagnostics_collector, 'get_state') else None,
                        lineage_state=lineage_tracker.get_state() if lineage_tracker and hasattr(lineage_tracker, 'get_state') else None
                    )

                    checkpoint_path_new = checkpoint_manager.save_checkpoint(evolution_state_current)
                    print(f"Checkpoint saved at generation {generation}: {checkpoint_path_new}")

                except Exception as e:
                    print(f"Warning: Failed to save checkpoint at generation {generation}: {e}")

            # Check for early termination
            if generations_without_improvement >= self.config.genetic_params.early_stopping_patience:
                print(f"Early stopping: No improvement for {self.config.genetic_params.early_stopping_patience} generations")
                break

            # Record generation data for diagnostics
            if diagnostics_collector:
                diagnostics_collector.record_generation(population, fitness_scores)

            # Update diversity dashboard if enabled
            if diversity_dashboard:
                # Only calculate diversity metrics when dashboard needs update
                dashboard_update_interval = getattr(diversity_dashboard.config, 'update_interval', 10)
                if generation % dashboard_update_interval == 0:
                    # Gather diversity metrics from diagnostics collector
                    diversity_metrics = {}
                    if diagnostics_collector and hasattr(diagnostics_collector, 'data') and diagnostics_collector.data.generations:
                        latest_gen = diagnostics_collector.data.generations[-1]

                        # Calculate unique individuals directly (override if diagnostics shows 0)
                        fitness_array = np.array(fitness_scores)
                        from ..cli.helpers import calculate_unique_individuals
                        unique_individuals = calculate_unique_individuals(population)
                        unique_ratio = unique_individuals / len(population) if population else 0.0

                        # Use max of diagnostics fitness_variance and calculated variance
                        calculated_variance = np.var(fitness_array)

                        diversity_metrics = {
                            'normalized_diversity': getattr(latest_gen, 'normalized_diversity', unique_ratio),
                            'hamming_distance_avg': getattr(latest_gen, 'hamming_distance_avg', getattr(latest_gen, 'hamming_diversity', 0.5)),
                            'position_wise_entropy': getattr(latest_gen, 'position_wise_entropy', getattr(latest_gen, 'entropy_diversity', 1.0)),
                            'fitness_variance': calculated_variance,
                            'unique_individuals_ratio': unique_ratio,
                            'cluster_count': getattr(latest_gen, 'cluster_diversity', 1.0),
                            'silhouette_score': 0.0,  # Default value
                            'spatial_autocorr': getattr(latest_gen, 'spatial_diversity', 0.0),
                            'spatial_moran_i': 0.0,  # Default value
                            'spatial_cluster_count': 1  # Default value
                        }
                    else:
                        # Use diagnostics collector to calculate proper metrics
                        if diagnostics_collector:
                            # Ensure num_source_images is set for comprehensive diversity manager
                            if self.source_images:
                                diagnostics_collector.config.num_source_images = len(self.source_images)

                            # Force diagnostics collector to collect metrics for current population
                            temp_data = diagnostics_collector.collect_generation_data(
                                population, fitness_scores, start_generation - 1,
                                evolution_state.generation, evolution_state.generations_without_improvement
                            )
                            diversity_metrics = {
                                'normalized_diversity': temp_data.normalized_diversity,
                                'hamming_distance_avg': temp_data.hamming_distance_avg,
                                'position_wise_entropy': temp_data.position_wise_entropy,
                                'fitness_variance': temp_data.fitness_variance,
                                'unique_individuals_ratio': temp_data.unique_individuals_ratio,
                                'cluster_count': temp_data.cluster_count,
                                'silhouette_score': getattr(temp_data, 'silhouette_score', 0.0),
                                'spatial_autocorr': temp_data.spatial_autocorrelation,
                                'spatial_moran_i': getattr(temp_data, 'spatial_moran_i', 0.0),
                                'spatial_cluster_count': getattr(temp_data, 'spatial_cluster_count', 1),
                                'spatial_diversity_score': temp_data.spatial_diversity_score
                            }
                        else:
                            # Fallback to simplified calculation only if diagnostics not available
                            diversity_metrics = self._calculate_initial_diversity_metrics_from_population(population, fitness_scores)
                else:
                    # For non-update generations, provide minimal metrics
                    diversity_metrics = {
                        'normalized_diversity': 0.5,  # Default placeholder
                        'hamming_distance_avg': 0.0,  # Skip expensive calculation
                        'position_wise_entropy': 0.0,  # Skip expensive calculation
                        'fitness_variance': np.var(fitness_scores) if fitness_scores else 0.0,  # This is cheap
                        'unique_individuals_ratio': 1.0,  # Default placeholder
                        'cluster_count': 1.0,  # Default placeholder
                        'silhouette_score': 0.0,  # Default value
                        'spatial_autocorr': 0.0,  # Skip expensive calculation
                        'spatial_moran_i': 0.0,  # Default value
                        'spatial_cluster_count': 1,  # Default value
                        'spatial_diversity_score': 0.0  # Skip expensive calculation
                    }

                # Create population state for dashboard
                population_state = {
                    'population_size': len(population),
                    'generation': generation,
                    'best_fitness': current_best_fitness,
                    'average_fitness': np.mean(fitness_scores),
                    'worst_fitness': np.max(fitness_scores)
                }

                # Update dashboard and get alerts
                alerts = diversity_dashboard.update_dashboard(generation, diversity_metrics, population_state)

                # Print alerts if any
                for alert in alerts:
                    print(f"Diversity Alert: {alert}")

            # Track lineage data if enabled
            if lineage_tracker and individual_ids:
                # Calculate diversity and selection pressure for lineage tracking
                try:
                    diversity_score = self.ga_engine.get_population_diversity()
                except Exception as e:
                    # Calculate diversity manually as fallback
                    from ..cli.helpers import calculate_unique_individuals
                    unique_individuals = calculate_unique_individuals(population)
                    from ..cli.helpers import calculate_population_diversity
                    diversity_score = calculate_population_diversity(population)

                # Calculate selection pressure using standardized definition
                from ..cli.helpers import calculate_selection_pressure
                selection_pressure = calculate_selection_pressure(fitness_scores)

                lineage_tracker.track_generation(
                    population, fitness_scores, individual_ids,
                    diversity_score, selection_pressure
                )

            self.ga_engine.evolve_population(fitness_scores)

            # Update individual IDs after evolution for lineage tracking
            if lineage_tracker and individual_ids:
                new_population = self.ga_engine.get_population()
                individual_ids = [f"gen_{generation+1}_ind_{i}" for i in range(len(new_population))]

        # Save diagnostics if enabled
        if diagnostics_collector:
            try:
                from ..diagnostics import DiagnosticsVisualizer
                diagnostics_collector.end_evolution()
                diagnostics_collector.save_to_folder(diagnostics_folder)

                # Generate visual reports
                visualizer = DiagnosticsVisualizer()
                visualizer.create_full_report(diagnostics_collector, diagnostics_folder)
                print(f"Diagnostics saved to: {diagnostics_folder}")
            except Exception as e:
                print(f"Warning: Failed to save diagnostics: {e}")

        # Save lineage analysis if enabled
        if lineage_tracker and lineage_visualizer:
            try:
                # Export lineage data
                lineage_tracker.export_data()

                # Generate all lineage visualizations
                lineage_visualizer.generate_all_plots()

                print(f"Lineage analysis saved to: {lineage_tracker.output_dir}")
            except Exception as e:
                print(f"Warning: Failed to save lineage analysis: {e}")

        # Export diversity dashboard data if enabled
        if diversity_dashboard and output_folder:
            try:
                dashboard_export_path = Path(output_folder) / "diversity_dashboard_data.json"
                diversity_dashboard.export_dashboard_data(str(dashboard_export_path))
                print(f"Diversity dashboard data exported to: {dashboard_export_path}")
            except Exception as e:
                print(f"Warning: Failed to export diversity dashboard data: {e}")

        # Create final result
        final_collage = self.renderer.render_final(best_individual, self.source_images)
        total_processing_time = evolution_state.total_processing_time + (self._get_time() - evolution_state.checkpoint_time)

        print(f"Evolution resumed successfully from generation {evolution_state.generation} to {generation}")

        return CollageResult(
            collage_image=final_collage,
            fitness_score=best_fitness,
            generations_used=generation + 1,
            processing_time=total_processing_time,
            best_individual=best_individual,
            fitness_history=fitness_history,
            evolution_frames=evolution_frames,
            evolution_generation_numbers=evolution_generation_numbers
        )

    def export(self, result: CollageResult, output_path: str, format: str = 'PNG') -> bool:
        try:
            return self.renderer.save_collage(result.collage_image, output_path, format)
        except Exception as e:
            print(f"Error exporting collage: {e}")
            return False
    
    def _get_time(self) -> float:
        import time
        return time.time()

    def _initialize_analysis_components(self, output_folder: Optional[str], diagnostics_folder: Optional[str] = None, evolution_state=None):
        """Initialize diagnostics, lineage tracking, and diversity dashboard components.

        Args:
            output_folder: Main output directory
            diagnostics_folder: Specific diagnostics directory (optional)
            evolution_state: Previous evolution state for resume (optional)

        Returns:
            Tuple of (diagnostics_collector, lineage_tracker, lineage_visualizer, individual_ids, diversity_dashboard)
        """
        diagnostics_collector = None
        lineage_tracker = None
        lineage_visualizer = None
        individual_ids = None
        diversity_dashboard = None

        # Initialize diagnostics if requested
        if diagnostics_folder:
            # Create diagnostics directory early for consistency
            try:
                Path(diagnostics_folder).mkdir(parents=True, exist_ok=True)
                print(f"Diagnostics directory created: {diagnostics_folder}")
            except Exception as e:
                print(f"Warning: Failed to create diagnostics directory: {e}")

            try:
                from ..diagnostics import DiagnosticsCollector
                diagnostics_collector = DiagnosticsCollector(self.config)

                # Set diagnostics collector in GA engine for migration tracking
                self.ga_engine.set_diagnostics_collector(diagnostics_collector)
            except ImportError as e:
                print(f"Warning: Diagnostics not available: {e}")
                diagnostics_collector = None

            # Ensure num_source_images is set for comprehensive diversity manager
            if diagnostics_collector and self.source_images:
                diagnostics_collector.config.num_source_images = len(self.source_images)

            if diagnostics_collector:
                diagnostics_collector.start_evolution()

        # Initialize lineage tracking if requested
        if self.config.enable_lineage_tracking:
            try:
                from ..lineage import LineageTracker, LineageVisualizer
                from ..lineage.fitness_components import FitnessComponentTracker

                # Create lineage output directory - prioritize config setting
                if self.config.lineage_output_dir:
                    if output_folder:
                        # Use configured name within output folder
                        lineage_dir = Path(output_folder) / self.config.lineage_output_dir
                    else:
                        # Use configured path as-is
                        lineage_dir = Path(self.config.lineage_output_dir)
                elif output_folder:
                    # Default to "lineage" within output folder
                    lineage_dir = Path(output_folder) / "lineage"
                else:
                    lineage_dir = Path("lineage_output")

                lineage_tracker = LineageTracker(str(lineage_dir))

                # Create fitness component tracker if enabled
                fitness_component_tracker = None
                if self.config.enable_component_tracking:
                    fitness_component_tracker = FitnessComponentTracker(['color', 'luminance', 'texture', 'edges'])

                lineage_visualizer = LineageVisualizer(lineage_tracker, str(lineage_dir), fitness_component_tracker)

                # Connect lineage tracker to GA engine
                self.ga_engine.set_lineage_tracker(lineage_tracker)
                print(f"Lineage tracking enabled for resume - output: {lineage_dir}")

                # Initialize lineage tracking with current population
                if evolution_state:
                    # Resuming from checkpoint
                    initial_fitness = [np.inf] * len(evolution_state.population)  # Placeholder fitness
                    individual_ids = lineage_tracker.initialize_population(evolution_state.population, initial_fitness)
                else:
                    # Fresh start - will be initialized later with real population
                    pass

            except ImportError as e:
                print(f"Warning: Lineage tracking not available: {e}")
                self.config.enable_lineage_tracking = False

        # Initialize diversity dashboard if enabled
        if self.config.enable_diversity_dashboard and DASHBOARD_AVAILABLE:
            try:
                from ..genetic.diversity_dashboard import DiversityDashboard, DashboardConfig
                dashboard_config = DashboardConfig(
                    update_interval=self.config.dashboard_update_interval,
                    alert_thresholds={
                        'critical_diversity': self.config.dashboard_alert_critical_diversity,
                        'low_diversity': self.config.dashboard_alert_low_diversity,
                        'stagnation_generations': 30,
                        'high_selection_pressure': 0.9,
                        'low_fitness_variance': 0.001
                    }
                )
                dashboard_output_dir = output_folder if output_folder else None

                diversity_dashboard = DiversityDashboard(dashboard_config, dashboard_output_dir)

                # Handle resume case - clean up discontinuous data from metrics file
                if evolution_state and hasattr(evolution_state, 'generation') and dashboard_output_dir:
                    self._cleanup_diversity_metrics_for_resume(Path(dashboard_output_dir), evolution_state.generation)

            except Exception as e:
                print(f"Warning: Failed to initialize diversity dashboard: {e}")
                self.config.enable_diversity_dashboard = False

        return diagnostics_collector, lineage_tracker, lineage_visualizer, individual_ids, diversity_dashboard

    def _cleanup_diversity_metrics_for_resume(self, output_dir: Path, resume_generation: int):
        """Clean up diversity metrics file to prevent generation discontinuity.

        Args:
            output_dir: Output directory containing diversity files
            resume_generation: Generation number we're resuming from
        """
        try:
            import json
            metrics_file = output_dir / "diversity_metrics.json"
            dashboard_log = output_dir / "diversity_dashboard.log"

            if metrics_file.exists():
                # Read all existing metrics
                valid_entries = []
                with open(metrics_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                # Keep only entries up to and including the resume generation
                                if entry.get('generation', 0) <= resume_generation:
                                    valid_entries.append(entry)
                            except json.JSONDecodeError:
                                continue

                # Rewrite the file with only valid entries
                with open(metrics_file, 'w') as f:
                    for entry in valid_entries:
                        json.dump(entry, f, default=str)
                        f.write('\n')

                print(f"Cleaned diversity metrics file: kept {len(valid_entries)} entries up to generation {resume_generation}")

            # Clean up dashboard log similarly
            if dashboard_log.exists():
                valid_log_lines = []
                with open(dashboard_log, 'r') as f:
                    for line in f:
                        # Keep log entries (they don't have structured generation data, so keep them for context)
                        valid_log_lines.append(line.rstrip())

                # For now, keep the log as-is since it's less structured
                # In future, could parse timestamps and clean based on generation correlation

        except Exception as e:
            print(f"Warning: Failed to clean diversity metrics for resume: {e}")

    def _calculate_initial_diversity_metrics_from_population(self, population: List[np.ndarray], fitness_scores: List[float]) -> Dict[str, float]:
        """Calculate comprehensive diversity metrics from loaded population on resume.

        Args:
            population: The restored population
            fitness_scores: Fitness scores for the population

        Returns:
            Dictionary of calculated diversity metrics
        """
        try:
            fitness_array = np.array(fitness_scores)

            # Use helper functions for diversity calculations
            from ..cli.helpers import calculate_unique_individuals, calculate_population_diversity, calculate_hamming_diversity, calculate_position_wise_entropy

            unique_individuals = calculate_unique_individuals(population)
            unique_ratio = calculate_population_diversity(population)

            hamming_diversity = calculate_hamming_diversity(population)

            position_wise_entropy = calculate_position_wise_entropy(population)

            # Calculate spatial diversity (using spatial autocorrelation)
            spatial_diversity = 0.0
            if population:
                # Simplified spatial diversity: measure how much adjacent tiles differ
                for ind in population[:min(10, len(population))]:  # Sample for performance
                    grid_height, grid_width = ind.shape
                    differences = 0
                    total_pairs = 0

                    # Check horizontal neighbors
                    for i in range(grid_height):
                        for j in range(grid_width - 1):
                            if ind[i, j] != ind[i, j + 1]:
                                differences += 1
                            total_pairs += 1

                    # Check vertical neighbors
                    for i in range(grid_height - 1):
                        for j in range(grid_width):
                            if ind[i, j] != ind[i + 1, j]:
                                differences += 1
                            total_pairs += 1

                    if total_pairs > 0:
                        spatial_diversity += differences / total_pairs

                spatial_diversity /= min(10, len(population))

            return {
                'normalized_diversity': unique_ratio,
                'hamming_distance_avg': hamming_diversity,
                'position_wise_entropy': position_wise_entropy,
                'fitness_variance': np.var(fitness_array),
                'unique_individuals_ratio': unique_ratio,
                'cluster_count': max(1.0, unique_ratio * len(population) / 10.0),  # Estimate
                'silhouette_score': 0.0,  # Would need clustering analysis
                'spatial_autocorr': spatial_diversity,
                'spatial_moran_i': 0.0,  # Would need full spatial analysis
                'spatial_cluster_count': max(1, int(spatial_diversity * 10))  # Estimate
            }

        except Exception as e:
            print(f"Warning: Failed to calculate initial diversity metrics: {e}")
            # Return safe defaults
            return {
                'normalized_diversity': 0.5,
                'hamming_distance_avg': 0.5,
                'position_wise_entropy': 1.0,
                'fitness_variance': np.var(fitness_scores) if fitness_scores else 0.0,
                'unique_individuals_ratio': 0.5,
                'cluster_count': 1.0,
                'silhouette_score': 0.0,
                'spatial_autocorr': 0.0,
                'spatial_moran_i': 0.0,
                'spatial_cluster_count': 1
            }