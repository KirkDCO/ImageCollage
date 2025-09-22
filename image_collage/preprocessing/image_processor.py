import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from skimage import feature, filters
from skimage.color import rgb2gray
import logging

from ..config.settings import CollageConfig


class ImageProcessor:
    def __init__(self, config: CollageConfig):
        self.config = config
        self.tile_size = config.tile_size
        
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Could not load image: {image_path}")
                return None
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None
    
    def validate_image(self, image: np.ndarray) -> bool:
        if image is None:
            return False
            
        height, width = image.shape[:2]
        
        if width < 50 or height < 50:
            logging.warning(f"Image too small: {width}x{height}")
            return False
            
        if width > 4096 or height > 4096:
            logging.warning(f"Image too large: {width}x{height}")
            return False
            
        return True
    
    def normalize_to_tile_size(self, image: np.ndarray) -> np.ndarray:
        target_width, target_height = self.tile_size
        
        current_height, current_width = image.shape[:2]
        
        scale_x = target_width / current_width
        scale_y = target_height / current_height
        scale = min(scale_x, scale_y)
        
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        if new_width == target_width and new_height == target_height:
            return resized
        
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        start_y = (target_height - new_height) // 2
        start_x = (target_width - new_width) // 2
        
        result[start_y:start_y + new_height, start_x:start_x + new_width] = resized
        
        return result
    
    def resize_target_to_grid(self, image: np.ndarray) -> np.ndarray:
        grid_width, grid_height = self.config.grid_size
        tile_width, tile_height = self.tile_size
        
        target_width = grid_width * tile_width
        target_height = grid_height * tile_height
        
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        features = {}
        
        features['color_histogram'] = self._compute_color_histogram(image)
        features['luminance'] = self._compute_luminance_stats(image)
        features['texture'] = self._compute_texture_features(image)
        features['edges'] = self._compute_edge_features(image)
        features['mean_color'] = np.mean(image, axis=(0, 1))
        
        return features
    
    def _compute_color_histogram(self, image: np.ndarray) -> np.ndarray:
        hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])
        
        hist = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        
        hist = hist / (np.sum(hist) + 1e-7)
        
        return hist
    
    def _compute_luminance_stats(self, image: np.ndarray) -> Dict[str, float]:
        gray = rgb2gray(image)
        
        return {
            'mean': float(np.mean(gray)),
            'std': float(np.std(gray)),
            'min': float(np.min(gray)),
            'max': float(np.max(gray)),
        }
    
    def _compute_texture_features(self, image: np.ndarray) -> Dict[str, Any]:
        gray = rgb2gray(image)
        
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return {
                'lbp_histogram': np.zeros(26),
                'glcm_contrast': 0.0,
                'glcm_energy': 0.0,
            }
        
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        try:
            lbp = feature.local_binary_pattern(gray_uint8, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-7)
        except:
            lbp_hist = np.zeros(26)
        
        try:
            glcm = feature.graycomatrix(gray_uint8, [1], [0], 256, symmetric=True, normed=True)
            contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
            energy = feature.graycoprops(glcm, 'energy')[0, 0]
        except:
            contrast = 0.0
            energy = 0.0
        
        return {
            'lbp_histogram': lbp_hist,
            'glcm_contrast': float(contrast),
            'glcm_energy': float(energy),
        }
    
    def _compute_edge_features(self, image: np.ndarray) -> Dict[str, Any]:
        gray = rgb2gray(image)
        
        try:
            edges_sobel = filters.sobel(gray)
            edge_magnitude = np.mean(edges_sobel)
            edge_density = np.sum(edges_sobel > 0.1) / edges_sobel.size
        except:
            edge_magnitude = 0.0
            edge_density = 0.0
        
        return {
            'edge_magnitude': float(edge_magnitude),
            'edge_density': float(edge_density),
        }
    
    def split_target_into_tiles(self, target_image: np.ndarray) -> np.ndarray:
        grid_width, grid_height = self.config.grid_size
        tile_width, tile_height = self.tile_size
        
        resized_target = self.resize_target_to_grid(target_image)
        
        tiles = np.zeros((grid_height, grid_width, tile_height, tile_width, 3), dtype=np.uint8)
        
        for i in range(grid_height):
            for j in range(grid_width):
                start_y = i * tile_height
                end_y = start_y + tile_height
                start_x = j * tile_width
                end_x = start_x + tile_width
                
                tiles[i, j] = resized_target[start_y:end_y, start_x:end_x]
        
        return tiles
    
    def extract_tile_features(self, target_tiles: np.ndarray) -> np.ndarray:
        grid_height, grid_width = target_tiles.shape[:2]
        
        tile_features = []
        
        for i in range(grid_height):
            row_features = []
            for j in range(grid_width):
                tile = target_tiles[i, j]
                features = self.extract_features(tile)
                row_features.append(features)
            tile_features.append(row_features)
        
        return np.array(tile_features, dtype=object)