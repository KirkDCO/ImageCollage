import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor
import time

try:
    import cupy as cp
    from cupyx.scipy.spatial.distance import cdist
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# For type annotations when CuPy is not available
if not CUPY_AVAILABLE:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        import cupy as cp

from ..config.settings import CollageConfig
from .evaluator import FitnessEvaluator


class GPUFitnessEvaluator(FitnessEvaluator):
    def __init__(self, config: CollageConfig, gpu_devices: Optional[List[int]] = None):
        super().__init__(config)
        
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU acceleration. Install with: pip install cupy")
        
        self.gpu_devices = gpu_devices or [0]
        self.device_pools = {}
        self.target_tiles_gpu = {}
        self.target_features_gpu = {}
        self.use_batch_processing = False
        
        self._initialize_gpu_devices()
        
    def _initialize_gpu_devices(self):
        """Initialize GPU devices and check availability."""
        available_devices = cp.cuda.runtime.getDeviceCount()
        logging.info(f"Found {available_devices} GPU devices")
        
        for device_id in self.gpu_devices:
            if device_id >= available_devices:
                raise ValueError(f"GPU device {device_id} not available. Only {available_devices} devices found.")
            
            with cp.cuda.Device(device_id):
                props = cp.cuda.runtime.getDeviceProperties(device_id)
                gpu_name = props['name'].decode('utf-8')
                total_mem = props['totalGlobalMem'] / (1024**3)  # GB
                logging.info(f"GPU {device_id}: {gpu_name} ({total_mem:.1f} GB)")
        
        # Check peer access between GPUs if using multiple devices
        if len(self.gpu_devices) > 1:
            self._check_peer_access()
    
    def _check_peer_access(self):
        """Check if peer access is available between GPUs."""
        try:
            # Test peer access between first two devices
            device0, device1 = self.gpu_devices[0], self.gpu_devices[1]
            
            with cp.cuda.Device(device0):
                can_access = cp.cuda.runtime.deviceCanAccessPeer(device0, device1)
                
            if not can_access:
                logging.warning(f"Peer access not available between GPU {device0} and {device1}")
                logging.info("Using batch processing mode (no data sharing between GPUs)")
                self.use_batch_processing = True
            else:
                # Enable peer access
                with cp.cuda.Device(device0):
                    try:
                        cp.cuda.runtime.deviceEnablePeerAccess(device1, 0)
                    except:
                        pass  # Already enabled or not needed
                        
                with cp.cuda.Device(device1):
                    try:
                        cp.cuda.runtime.deviceEnablePeerAccess(device0, 0)
                    except:
                        pass  # Already enabled or not needed
                        
                logging.info(f"Peer access enabled between GPUs {device0} and {device1}")
                self.use_batch_processing = False
                
        except Exception as e:
            logging.warning(f"Error checking peer access: {e}")
            logging.info("Using batch processing mode as fallback")
            self.use_batch_processing = True
    
    def _evaluate_population_multi_gpu_batch(self, population: List[np.ndarray], 
                                           source_images: List[np.ndarray], 
                                           source_features: List[Dict[str, Any]]) -> List[float]:
        """Evaluate population by splitting across multiple GPUs."""
        population_size = len(population)
        num_gpus = len(self.gpu_devices)
        
        # Split population across GPUs
        individuals_per_gpu = population_size // num_gpus
        remainder = population_size % num_gpus
        
        gpu_batches = []
        start_idx = 0
        
        for i, device_id in enumerate(self.gpu_devices):
            # Distribute remainder across first few GPUs
            batch_size = individuals_per_gpu + (1 if i < remainder else 0)
            end_idx = start_idx + batch_size
            
            batch = population[start_idx:end_idx]
            gpu_batches.append((device_id, batch))
            start_idx = end_idx
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for device_id, batch in gpu_batches:
                future = executor.submit(
                    self._evaluate_batch_on_gpu,
                    device_id, batch, source_images, source_features
                )
                futures.append(future)
            
            # Collect results and flatten
            all_fitness_scores = []
            for future in futures:
                batch_scores = future.result()
                all_fitness_scores.extend(batch_scores)
        
        return all_fitness_scores
    
    def _evaluate_batch_on_gpu(self, device_id: int, batch: List[np.ndarray], 
                              source_images: List[np.ndarray], 
                              source_features: List[Dict[str, Any]]) -> List[float]:
        """Evaluate a batch of individuals on a specific GPU."""
        with cp.cuda.Device(device_id):
            batch_scores = []
            
            for individual in batch:
                fitness = self._gpu_fitness_calculation(device_id, individual, source_images, source_features)
                batch_scores.append(fitness)
            
            return batch_scores
    
    def _evaluate_multi_gpu_batch(self, individual: np.ndarray, source_images: List[np.ndarray], 
                                source_features: List[Dict[str, Any]]) -> float:
        """Fallback: evaluate single individual using first GPU when batch method not used."""
        return self._evaluate_single_gpu(individual, source_images, source_features)
                
    def set_target(self, target_image: np.ndarray, target_features: Dict[str, Any]) -> None:
        """Set target image and precompute GPU data."""
        super().set_target(target_image, target_features)
        
        # Precompute target data on each GPU
        for device_id in self.gpu_devices:
            with cp.cuda.Device(device_id):
                self.target_tiles_gpu[device_id] = cp.asarray(self.target_tiles)
                
                # Precompute target features on GPU
                target_colors = self._extract_gpu_colors(self.target_tiles_gpu[device_id])
                self.target_features_gpu[device_id] = {
                    'colors': target_colors,
                    'mean_colors': cp.mean(target_colors, axis=(2, 3))
                }
    
    def evaluate(self, individual: np.ndarray, source_images: List[np.ndarray], 
                source_features: List[Dict[str, Any]]) -> float:
        """GPU-accelerated fitness evaluation with multi-GPU support."""
        if not self.target_tiles_gpu:
            raise ValueError("Target image not set for GPU evaluation")
        
        grid_height, grid_width = individual.shape
        
        if len(self.gpu_devices) == 1:
            return self._evaluate_single_gpu(individual, source_images, source_features)
        else:
            return self._evaluate_multi_gpu_batch(individual, source_images, source_features)
    
    def evaluate_population_batch(self, population: List[np.ndarray], source_images: List[np.ndarray], 
                                source_features: List[Dict[str, Any]]) -> List[float]:
        """Batch evaluate entire population using multiple GPUs."""
        if not self.target_tiles_gpu:
            raise ValueError("Target image not set for GPU evaluation")
        
        if len(self.gpu_devices) == 1:
            # Single GPU - evaluate sequentially
            return [self._evaluate_single_gpu(ind, source_images, source_features) for ind in population]
        else:
            # Multi-GPU batch processing
            return self._evaluate_population_multi_gpu_batch(population, source_images, source_features)
    
    def _evaluate_single_gpu(self, individual: np.ndarray, source_images: List[np.ndarray], 
                           source_features: List[Dict[str, Any]]) -> float:
        """Single GPU evaluation."""
        device_id = self.gpu_devices[0]
        
        with cp.cuda.Device(device_id):
            return self._gpu_fitness_calculation(
                device_id, individual, source_images, source_features
            )
    
    def _evaluate_multi_gpu(self, individual: np.ndarray, source_images: List[np.ndarray], 
                          source_features: List[Dict[str, Any]]) -> float:
        """Multi-GPU evaluation by splitting work across devices."""
        grid_height, grid_width = individual.shape
        total_tiles = grid_height * grid_width
        tiles_per_gpu = total_tiles // len(self.gpu_devices)
        
        # Split individual into chunks for each GPU
        individual_flat = individual.flatten()
        chunks = []
        
        for i, device_id in enumerate(self.gpu_devices):
            start_idx = i * tiles_per_gpu
            if i == len(self.gpu_devices) - 1:  # Last GPU gets remaining tiles
                end_idx = total_tiles
            else:
                end_idx = (i + 1) * tiles_per_gpu
            
            chunk_indices = individual_flat[start_idx:end_idx]
            chunks.append((device_id, start_idx, chunk_indices))
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=len(self.gpu_devices)) as executor:
            futures = []
            for device_id, start_idx, chunk_indices in chunks:
                future = executor.submit(
                    self._gpu_chunk_evaluation,
                    device_id, start_idx, chunk_indices, individual.shape,
                    source_images, source_features
                )
                futures.append(future)
            
            # Collect results
            total_fitness = 0.0
            total_count = 0
            
            for future in futures:
                chunk_fitness, chunk_count = future.result()
                total_fitness += chunk_fitness
                total_count += chunk_count
        
        return total_fitness / total_count
    
    def _gpu_chunk_evaluation(self, device_id: int, start_idx: int, chunk_indices: np.ndarray,
                            grid_shape: tuple, source_images: List[np.ndarray], 
                            source_features: List[Dict[str, Any]]) -> tuple:
        """Evaluate a chunk of tiles on specific GPU."""
        with cp.cuda.Device(device_id):
            grid_height, grid_width = grid_shape
            
            # Convert flat indices back to 2D coordinates
            chunk_fitness = 0.0
            chunk_count = len(chunk_indices)
            
            for flat_idx, source_idx in enumerate(chunk_indices):
                # Convert flat index to 2D coordinates
                actual_idx = start_idx + flat_idx
                i = actual_idx // grid_width
                j = actual_idx % grid_width
                
                if i < grid_height and j < grid_width:
                    target_tile_gpu = self.target_tiles_gpu[device_id][i, j]
                    
                    # Always create source tile on current device to avoid cross-device issues
                    source_tile_gpu = cp.asarray(source_images[source_idx])
                    
                    tile_fitness = self._gpu_tile_fitness(target_tile_gpu, source_tile_gpu)
                    chunk_fitness += tile_fitness
            
            # Ensure result is returned as Python float, not GPU array
            return float(chunk_fitness), chunk_count
    
    def _gpu_fitness_calculation(self, device_id: int, individual: np.ndarray,
                               source_images: List[np.ndarray],
                               source_features: List[Dict[str, Any]]) -> float:
        """Core GPU fitness calculation."""
        grid_height, grid_width = individual.shape
        total_fitness = 0.0
        
        # Batch process tiles for better GPU utilization
        batch_size = min(256, grid_height * grid_width)  # Adjust based on GPU memory
        
        for batch_start in range(0, grid_height * grid_width, batch_size):
            batch_end = min(batch_start + batch_size, grid_height * grid_width)
            batch_fitness = 0.0

            for flat_idx in range(batch_start, batch_end):
                i = flat_idx // grid_width
                j = flat_idx % grid_width

                # Bounds checking to prevent index errors
                if i >= grid_height or j >= grid_width:
                    continue

                source_idx = individual[i, j]
                target_tile_gpu = self.target_tiles_gpu[device_id][i, j]
                source_tile_gpu = cp.asarray(source_images[source_idx])
                
                tile_fitness = self._gpu_tile_fitness(target_tile_gpu, source_tile_gpu)
                batch_fitness += tile_fitness
            
            total_fitness += batch_fitness
        
        return total_fitness / (grid_height * grid_width)
    
    def _gpu_tile_fitness(self, target_tile_gpu: 'cp.ndarray', source_tile_gpu: 'cp.ndarray') -> float:
        """GPU-accelerated tile fitness calculation."""
        weights = self.config.fitness_weights
        
        # Color similarity (dominant component)
        color_fitness = self._gpu_color_similarity(target_tile_gpu, source_tile_gpu) * weights.color
        
        # Luminance similarity
        luminance_fitness = self._gpu_luminance_similarity(target_tile_gpu, source_tile_gpu) * weights.luminance
        
        # Simplified texture and edge for GPU efficiency
        texture_fitness = self._gpu_texture_similarity(target_tile_gpu, source_tile_gpu) * weights.texture
        edge_fitness = self._gpu_edge_similarity(target_tile_gpu, source_tile_gpu) * weights.edges
        
        return color_fitness + luminance_fitness + texture_fitness + edge_fitness
    
    def _gpu_color_similarity(self, target_gpu: 'cp.ndarray', source_gpu: 'cp.ndarray') -> float:
        """GPU-accelerated color similarity using mean color distance."""
        target_mean = cp.mean(target_gpu, axis=(0, 1))
        source_mean = cp.mean(source_gpu, axis=(0, 1))
        
        # Euclidean distance in RGB space (normalized)
        distance = cp.linalg.norm(target_mean - source_mean)
        return float(distance.get()) / (255.0 * float(cp.sqrt(3).get()))
    
    def _gpu_luminance_similarity(self, target_gpu: 'cp.ndarray', source_gpu: 'cp.ndarray') -> float:
        """GPU-accelerated luminance similarity."""
        # Convert to grayscale using standard weights
        target_gray = cp.dot(target_gpu, cp.array([0.299, 0.587, 0.114]))
        source_gray = cp.dot(source_gpu, cp.array([0.299, 0.587, 0.114]))
        
        target_mean = cp.mean(target_gray)
        source_mean = cp.mean(source_gray)
        
        return float(cp.abs(target_mean - source_mean).get()) / 255.0
    
    def _gpu_texture_similarity(self, target_gpu: 'cp.ndarray', source_gpu: 'cp.ndarray') -> float:
        """Simplified GPU texture similarity using standard deviation."""
        target_std = cp.std(target_gpu)
        source_std = cp.std(source_gpu)
        
        return float(cp.abs(target_std - source_std).get()) / 255.0
    
    def _gpu_edge_similarity(self, target_gpu: 'cp.ndarray', source_gpu: 'cp.ndarray') -> float:
        """GPU-accelerated edge similarity using gradient magnitude."""
        # Simple gradient-based edge detection
        target_gray = cp.dot(target_gpu, cp.array([0.299, 0.587, 0.114]))
        source_gray = cp.dot(source_gpu, cp.array([0.299, 0.587, 0.114]))
        
        # Compute gradients
        target_grad = cp.gradient(target_gray)
        source_grad = cp.gradient(source_gray)
        
        target_magnitude = cp.sqrt(target_grad[0]**2 + target_grad[1]**2)
        source_magnitude = cp.sqrt(source_grad[0]**2 + source_grad[1]**2)
        
        target_edge = cp.mean(target_magnitude)
        source_edge = cp.mean(source_magnitude)
        
        return float(cp.abs(target_edge - source_edge).get()) / 255.0
    
    def _extract_gpu_colors(self, tiles_gpu: 'cp.ndarray') -> 'cp.ndarray':
        """Extract color information for efficient GPU processing."""
        return tiles_gpu.astype(cp.float32) / 255.0
    
    def get_gpu_memory_usage(self) -> Dict[int, Dict[str, float]]:
        """Get memory usage statistics for all GPUs."""
        memory_stats = {}
        
        for device_id in self.gpu_devices:
            with cp.cuda.Device(device_id):
                mempool = cp.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                total_bytes = mempool.total_bytes()
                
                memory_stats[device_id] = {
                    'used_gb': used_bytes / (1024**3),
                    'total_gb': total_bytes / (1024**3),
                    'utilization': used_bytes / total_bytes if total_bytes > 0 else 0.0
                }
        
        return memory_stats
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        for device_id in self.gpu_devices:
            with cp.cuda.Device(device_id):
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                cp.cuda.Stream.null.synchronize()