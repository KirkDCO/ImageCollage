"""
Color Tile Generator - Creates diverse sets of single-color tiles spanning RGB spectrum.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import colorsys
from typing import List, Tuple, Optional
import math


class ColorTileGenerator:
    """Generates diverse single-color tiles spanning the RGB spectrum."""

    def __init__(self, tile_size: Tuple[int, int] = (32, 32)):
        """
        Initialize color tile generator.

        Args:
            tile_size: Size of generated tiles (width, height)
        """
        self.tile_size = tile_size

    def generate_diverse_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """
        Generate a diverse set of RGB colors spanning the full spectrum.

        Uses a combination of systematic sampling and perceptual spacing to ensure
        good coverage of the RGB color space.

        Args:
            num_colors: Number of colors to generate

        Returns:
            List of RGB tuples (0-255 range)
        """
        if num_colors <= 0:
            return []

        colors = []

        # Always include key colors for small sets
        key_colors = [
            (0, 0, 0),       # Black
            (255, 255, 255), # White
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
        ]

        if num_colors <= len(key_colors):
            return key_colors[:num_colors]

        # Add key colors first
        colors.extend(key_colors)
        remaining = num_colors - len(key_colors)

        if remaining > 0:
            # Generate additional colors using HSV space for better perceptual distribution
            additional_colors = self._generate_hsv_distributed_colors(remaining)
            colors.extend(additional_colors)

        return colors[:num_colors]

    def _generate_hsv_distributed_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate colors distributed evenly in HSV space for better perceptual coverage."""
        colors = []

        # Use golden ratio for hue distribution to avoid clustering
        golden_ratio = (1 + 5**0.5) / 2

        for i in range(num_colors):
            # Distribute hues using golden ratio
            hue = (i * golden_ratio) % 1.0

            # Vary saturation and value to get good coverage
            # Use different saturation/value combinations for better diversity
            sat_val_pairs = [
                (1.0, 1.0),   # Pure bright colors
                (1.0, 0.7),   # Darker pure colors
                (0.7, 1.0),   # Bright but less saturated
                (0.7, 0.7),   # Medium saturation and brightness
                (0.3, 1.0),   # Pale bright colors
                (1.0, 0.4),   # Dark saturated colors
            ]

            sat, val = sat_val_pairs[i % len(sat_val_pairs)]

            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)

            # Convert to 0-255 range
            rgb = (
                int(r * 255),
                int(g * 255),
                int(b * 255)
            )

            colors.append(rgb)

        return colors

    def generate_tiles(self, num_tiles: int, output_directory: str,
                      prefix: str = "color_tile") -> int:
        """
        Generate a set of diverse single-color tiles.

        Args:
            num_tiles: Number of tiles to generate
            output_directory: Directory to save tiles
            prefix: Filename prefix for generated tiles

        Returns:
            Number of tiles actually generated
        """
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate diverse color set
        colors = self.generate_diverse_colors(num_tiles)

        generated_count = 0
        for i, (r, g, b) in enumerate(colors):
            try:
                # Create solid color image
                image = Image.new('RGB', self.tile_size, (r, g, b))

                # Save with descriptive filename
                filename = f"{prefix}_{i:03d}_rgb_{r:03d}_{g:03d}_{b:03d}.png"
                filepath = output_path / filename

                image.save(filepath, 'PNG')
                generated_count += 1

            except Exception as e:
                print(f"Warning: Failed to generate tile {i}: {e}")
                continue

        return generated_count

    def preview_color_palette(self, num_colors: int,
                            output_path: Optional[str] = None) -> np.ndarray:
        """
        Create a preview image showing the color palette that would be generated.

        Args:
            num_colors: Number of colors in the palette
            output_path: Optional path to save preview image

        Returns:
            Numpy array of the preview image
        """
        colors = self.generate_diverse_colors(num_colors)

        # Calculate grid dimensions for preview
        cols = min(16, int(math.ceil(math.sqrt(num_colors))))
        rows = math.ceil(num_colors / cols)

        # Size of each color swatch
        swatch_size = 32

        # Create preview image
        preview_width = cols * swatch_size
        preview_height = rows * swatch_size
        preview = Image.new('RGB', (preview_width, preview_height), (128, 128, 128))

        for i, (r, g, b) in enumerate(colors):
            row = i // cols
            col = i % cols

            # Create color swatch
            swatch = Image.new('RGB', (swatch_size, swatch_size), (r, g, b))

            # Paste into preview
            x = col * swatch_size
            y = row * swatch_size
            preview.paste(swatch, (x, y))

        if output_path:
            preview.save(output_path)

        return np.array(preview)

    def analyze_color_distribution(self, colors: List[Tuple[int, int, int]]) -> dict:
        """
        Analyze the distribution and diversity of a color set.

        Args:
            colors: List of RGB color tuples

        Returns:
            Dictionary with distribution statistics
        """
        if not colors:
            return {}

        # Convert to numpy for analysis
        color_array = np.array(colors)

        # Calculate statistics
        stats = {
            'count': len(colors),
            'rgb_means': {
                'red': float(np.mean(color_array[:, 0])),
                'green': float(np.mean(color_array[:, 1])),
                'blue': float(np.mean(color_array[:, 2]))
            },
            'rgb_std': {
                'red': float(np.std(color_array[:, 0])),
                'green': float(np.std(color_array[:, 1])),
                'blue': float(np.std(color_array[:, 2]))
            },
            'brightness_range': {
                'min': float(np.min(np.sum(color_array, axis=1))),
                'max': float(np.max(np.sum(color_array, axis=1))),
                'mean': float(np.mean(np.sum(color_array, axis=1)))
            }
        }

        # Calculate color space coverage (0-1 scale)
        rgb_coverage = {
            'red': (np.max(color_array[:, 0]) - np.min(color_array[:, 0])) / 255.0,
            'green': (np.max(color_array[:, 1]) - np.min(color_array[:, 1])) / 255.0,
            'blue': (np.max(color_array[:, 2]) - np.min(color_array[:, 2])) / 255.0
        }
        stats['rgb_coverage'] = rgb_coverage
        stats['average_coverage'] = sum(rgb_coverage.values()) / 3.0

        return stats