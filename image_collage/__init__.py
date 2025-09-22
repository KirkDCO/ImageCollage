"""
Image Collage Generator

A genetic algorithm-based tool that creates photomosaic collages by arranging 
a collection of source images to visually approximate a target image.
"""

from .core.collage_generator import CollageGenerator, CollageResult
from .config.settings import CollageConfig, PresetConfigs, FitnessWeights, GeneticParams

__version__ = "0.1.0"
__author__ = "ImageCollage Team"
__email__ = "contact@imagecollage.dev"

__all__ = [
    "CollageGenerator",
    "CollageResult", 
    "CollageConfig",
    "PresetConfigs",
    "FitnessWeights",
    "GeneticParams",
]