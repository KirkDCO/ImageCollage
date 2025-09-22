import numpy as np
from typing import List, Dict, Any, Optional
import cv2
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage import feature, filters
from skimage.color import rgb2gray
import logging
from multiprocessing import Pool
import functools

from ..config.settings import CollageConfig
from ..preprocessing.image_processor import ImageProcessor


class FitnessEvaluator:
    def __init__(self, config: CollageConfig):
        self.config = config
        self.target_image: Optional[np.ndarray] = None
        self.target_features: Optional[Dict[str, Any]] = None
        self.target_tiles: Optional[np.ndarray] = None
        self.target_tile_features: Optional[np.ndarray] = None
        self.image_processor = ImageProcessor(config)
        
    def update_config(self, config: CollageConfig) -> None:
        self.config = config
        
    def set_target(self, target_image: np.ndarray, target_features: Dict[str, Any]) -> None:
        self.target_image = target_image
        self.target_features = target_features
        
        self.target_tiles = self.image_processor.split_target_into_tiles(target_image)
        self.target_tile_features = self.image_processor.extract_tile_features(self.target_tiles)
    
    def evaluate(self, individual: np.ndarray, source_images: List[np.ndarray], 
                source_features: List[Dict[str, Any]]) -> float:
        if self.target_tiles is None or self.target_tile_features is None:
            raise ValueError("Target image not set")
        
        grid_height, grid_width = individual.shape
        total_fitness = 0.0
        
        if self.config.enable_parallel_processing and self.config.num_processes > 1:
            fitness = self._evaluate_parallel(individual, source_images, source_features)
        else:
            fitness = self._evaluate_sequential(individual, source_images, source_features)
        
        return fitness
    
    def _evaluate_sequential(self, individual: np.ndarray, source_images: List[np.ndarray], 
                           source_features: List[Dict[str, Any]]) -> float:
        grid_height, grid_width = individual.shape
        total_fitness = 0.0
        
        for i in range(grid_height):
            for j in range(grid_width):
                source_idx = individual[i, j]
                target_tile = self.target_tiles[i, j]
                target_tile_features = self.target_tile_features[i, j]
                source_tile = source_images[source_idx]
                source_tile_features = source_features[source_idx]
                
                tile_fitness = self._evaluate_tile_fitness(
                    target_tile, target_tile_features,
                    source_tile, source_tile_features
                )
                total_fitness += tile_fitness
        
        return total_fitness / (grid_height * grid_width)
    
    def _evaluate_parallel(self, individual: np.ndarray, source_images: List[np.ndarray], 
                          source_features: List[Dict[str, Any]]) -> float:
        grid_height, grid_width = individual.shape
        
        tasks = []
        for i in range(grid_height):
            for j in range(grid_width):
                source_idx = individual[i, j]
                tasks.append((
                    self.target_tiles[i, j],
                    self.target_tile_features[i, j],
                    source_images[source_idx],
                    source_features[source_idx]
                ))
        
        with Pool(processes=self.config.num_processes) as pool:
            fitness_scores = pool.map(self._evaluate_tile_fitness_wrapper, tasks)
        
        return sum(fitness_scores) / len(fitness_scores)
    
    def _evaluate_tile_fitness_wrapper(self, args) -> float:
        return self._evaluate_tile_fitness(*args)
    
    def _evaluate_tile_fitness(self, target_tile: np.ndarray, target_features: Dict[str, Any],
                              source_tile: np.ndarray, source_features: Dict[str, Any]) -> float:
        weights = self.config.fitness_weights
        
        color_fitness = self._color_similarity(target_tile, source_tile) * weights.color
        luminance_fitness = self._luminance_similarity(target_features, source_features) * weights.luminance
        texture_fitness = self._texture_similarity(target_features, source_features) * weights.texture
        edge_fitness = self._edge_similarity(target_features, source_features) * weights.edges
        
        total_fitness = color_fitness + luminance_fitness + texture_fitness + edge_fitness
        
        return total_fitness
    
    def _color_similarity(self, target_tile: np.ndarray, source_tile: np.ndarray) -> float:
        try:
            target_lab = rgb2lab(target_tile.astype(np.float64) / 255.0)
            source_lab = rgb2lab(source_tile.astype(np.float64) / 255.0)
            
            target_mean = np.mean(target_lab.reshape(-1, 3), axis=0)
            source_mean = np.mean(source_lab.reshape(-1, 3), axis=0)
            
            delta_e = deltaE_ciede2000(
                target_mean.reshape(1, 1, 3),
                source_mean.reshape(1, 1, 3)
            )[0, 0]
            
            return min(delta_e / 100.0, 1.0)
            
        except Exception as e:
            logging.warning(f"Error in color similarity calculation: {e}")
            target_mean = np.mean(target_tile, axis=(0, 1))
            source_mean = np.mean(source_tile, axis=(0, 1))
            return np.linalg.norm(target_mean - source_mean) / (255.0 * np.sqrt(3))
    
    def _luminance_similarity(self, target_features: Dict[str, Any], 
                            source_features: Dict[str, Any]) -> float:
        target_lum = target_features['luminance']
        source_lum = source_features['luminance']
        
        mean_diff = abs(target_lum['mean'] - source_lum['mean'])
        std_diff = abs(target_lum['std'] - source_lum['std'])
        
        return (mean_diff + std_diff * 0.5) / 2.0
    
    def _texture_similarity(self, target_features: Dict[str, Any], 
                          source_features: Dict[str, Any]) -> float:
        try:
            target_texture = target_features['texture']
            source_texture = source_features['texture']
            
            lbp_diff = np.sum(np.abs(
                target_texture['lbp_histogram'] - source_texture['lbp_histogram']
            )) / 2.0
            
            contrast_diff = abs(
                target_texture['glcm_contrast'] - source_texture['glcm_contrast']
            ) / 100.0
            
            energy_diff = abs(
                target_texture['glcm_energy'] - source_texture['glcm_energy']
            )
            
            return (lbp_diff * 0.6 + contrast_diff * 0.3 + energy_diff * 0.1)
            
        except Exception as e:
            logging.warning(f"Error in texture similarity calculation: {e}")
            return 0.5
    
    def _edge_similarity(self, target_features: Dict[str, Any], 
                        source_features: Dict[str, Any]) -> float:
        try:
            target_edges = target_features['edges']
            source_edges = source_features['edges']
            
            magnitude_diff = abs(
                target_edges['edge_magnitude'] - source_edges['edge_magnitude']
            )
            
            density_diff = abs(
                target_edges['edge_density'] - source_edges['edge_density']
            )
            
            return (magnitude_diff * 0.7 + density_diff * 0.3)
            
        except Exception as e:
            logging.warning(f"Error in edge similarity calculation: {e}")
            return 0.5
    
    def _histogram_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        try:
            correlation = cv2.compareHist(
                hist1.astype(np.float32), 
                hist2.astype(np.float32), 
                cv2.HISTCMP_CORREL
            )
            return 1.0 - correlation
        except:
            return np.sum(np.abs(hist1 - hist2)) / 2.0
    
    def evaluate_global_fitness(self, collage_image: np.ndarray) -> Dict[str, float]:
        if self.target_image is None:
            raise ValueError("Target image not set")
        
        target_resized = cv2.resize(self.target_image, 
                                   (collage_image.shape[1], collage_image.shape[0]))
        
        mse = np.mean((target_resized.astype(float) - collage_image.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        target_gray = rgb2gray(target_resized)
        collage_gray = rgb2gray(collage_image)
        
        ssim_score = self._structural_similarity(target_gray, collage_gray)
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim_score)
        }
    
    def _structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            from skimage.metrics import structural_similarity
            return structural_similarity(img1, img2, data_range=1.0)
        except ImportError:
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            sigma1 = np.var(img1)
            sigma2 = np.var(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            
            return ssim