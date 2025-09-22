import numpy as np
import cv2
from typing import List, Optional, Union
from pathlib import Path
from PIL import Image, ImageFilter
import logging

from ..config.settings import CollageConfig


class Renderer:
    def __init__(self, config: CollageConfig):
        self.config = config
        self.tile_size = config.tile_size
        self.grid_size = config.grid_size
        
    def update_config(self, config: CollageConfig) -> None:
        self.config = config
        self.tile_size = config.tile_size
        self.grid_size = config.grid_size
    
    def render_preview(self, individual: np.ndarray, source_images: List[np.ndarray]) -> np.ndarray:
        return self._render_collage(individual, source_images, preview=True)
    
    def render_final(self, individual: np.ndarray, source_images: List[np.ndarray]) -> np.ndarray:
        return self._render_collage(individual, source_images, preview=False)
    
    def _render_collage(self, individual: np.ndarray, source_images: List[np.ndarray], 
                       preview: bool = False) -> np.ndarray:
        grid_height, grid_width = individual.shape
        tile_width, tile_height = self.tile_size
        
        if preview:
            tile_width = max(8, tile_width // 4)
            tile_height = max(8, tile_height // 4)
        
        collage_width = grid_width * tile_width
        collage_height = grid_height * tile_height
        
        collage = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)
        
        for i in range(grid_height):
            for j in range(grid_width):
                source_idx = individual[i, j]
                source_image = source_images[source_idx]
                
                tile = self._prepare_tile(source_image, tile_width, tile_height)
                
                start_y = i * tile_height
                end_y = start_y + tile_height
                start_x = j * tile_width
                end_x = start_x + tile_width
                
                collage[start_y:end_y, start_x:end_x] = tile
        
        if self.config.enable_edge_blending and not preview:
            collage = self._apply_edge_blending(collage, individual)
        
        return collage
    
    def _prepare_tile(self, source_image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        if source_image.shape[:2] == (target_height, target_width):
            return source_image
        
        return cv2.resize(source_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    def _apply_edge_blending(self, collage: np.ndarray, individual: np.ndarray) -> np.ndarray:
        try:
            pil_image = Image.fromarray(collage)
            
            blended = pil_image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            return np.array(blended)
            
        except Exception as e:
            logging.warning(f"Edge blending failed: {e}")
            return collage
    
    def save_collage(self, collage_image: np.ndarray, output_path: str, 
                    format: str = 'PNG') -> bool:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.upper() == 'PNG':
                cv2.imwrite(str(output_path), cv2.cvtColor(collage_image, cv2.COLOR_RGB2BGR))
            
            elif format.upper() in ['JPEG', 'JPG']:
                quality = self.config.output_quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                cv2.imwrite(str(output_path), cv2.cvtColor(collage_image, cv2.COLOR_RGB2BGR), encode_param)
            
            elif format.upper() == 'TIFF':
                pil_image = Image.fromarray(collage_image)
                pil_image.save(str(output_path), format='TIFF', compression='lzw')
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving collage: {e}")
            return False
    
    def create_evolution_animation(self, evolution_frames: List[np.ndarray],
                                 output_path: str, duration: int = 200,
                                 generation_numbers: Optional[List[int]] = None) -> bool:
        """Create enhanced evolution animation with generation number titles.

        Args:
            evolution_frames: List of collage preview images from evolution
            output_path: Path to save the animated GIF
            duration: Duration per frame in milliseconds
            generation_numbers: List of generation numbers corresponding to frames

        Returns:
            bool: True if animation was created successfully

        Features:
            - Generation number titles on each frame
            - White title bar with centered "Generation N" text
            - Automatic frame numbering if generation_numbers not provided
        """
        try:
            if not evolution_frames:
                logging.warning("No evolution frames provided")
                return False

            frames = []
            for i, frame in enumerate(evolution_frames):
                # Add generation number title to each frame
                frame_with_title = self._add_generation_title(
                    frame,
                    generation_numbers[i] if generation_numbers and i < len(generation_numbers) else i
                )
                frame_pil = Image.fromarray(frame_with_title)
                frames.append(frame_pil)

            # Save as animated GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )
            return True

        except Exception as e:
            logging.error(f"Error creating animation: {e}")
            return False

    def _add_generation_title(self, frame: np.ndarray, generation: int) -> np.ndarray:
        """Add generation number using matplotlib subplots for better layout."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from io import BytesIO

            # Calculate figure size based on frame dimensions
            height, width = frame.shape[:2]
            dpi = 100
            fig_width = width / dpi
            fig_height = (height + 60) / dpi  # Add space for title

            # Create figure with subplots
            fig, (ax_title, ax_grid) = plt.subplots(2, 1, figsize=(fig_width, fig_height),
                                                   gridspec_kw={'height_ratios': [1, 10], 'hspace': 0})

            # Top subplot: Generation title
            ax_title.text(0.5, 0.5, f'Generation {generation}',
                         ha='center', va='center', fontsize=14, fontweight='bold',
                         transform=ax_title.transAxes)
            ax_title.set_xlim(0, 1)
            ax_title.set_ylim(0, 1)
            ax_title.axis('off')

            # Bottom subplot: Grid image
            ax_grid.imshow(frame)
            ax_grid.axis('off')

            # Save to bytes and convert back to numpy array
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Render to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
            buf.seek(0)

            # Convert back to numpy array
            from PIL import Image
            pil_image = Image.open(buf)
            result_frame = np.array(pil_image)

            # Convert RGBA to RGB if needed
            if result_frame.shape[2] == 4:
                result_frame = result_frame[:, :, :3]

            plt.close(fig)
            buf.close()

            return result_frame

        except Exception as e:
            logging.warning(f"Could not add generation title with subplots: {e}")
            # Fallback to original frame
            return frame

    def create_comparison_image(self, target_image: np.ndarray, 
                               collage_image: np.ndarray,
                               output_path: str) -> bool:
        try:
            target_resized = cv2.resize(
                target_image, 
                (collage_image.shape[1], collage_image.shape[0])
            )
            
            comparison = np.hstack([target_resized, collage_image])
            
            return self.save_collage(comparison, output_path, 'PNG')
            
        except Exception as e:
            logging.error(f"Error creating comparison image: {e}")
            return False
    
    def add_metadata(self, image_path: str, metadata: dict) -> bool:
        try:
            pil_image = Image.open(image_path)
            
            from PIL.ExifTags import TAGS
            from PIL.ExifTags import Base
            
            exif_dict = {
                "0th": {},
                "Exif": {},
                "GPS": {},
                "1st": {},
                "thumbnail": None
            }
            
            for key, value in metadata.items():
                if key == "generation_count":
                    exif_dict["0th"][Base.ImageDescription.value] = f"Generation: {value}"
                elif key == "fitness_score":
                    exif_dict["0th"][Base.Software.value] = f"Fitness: {value:.6f}"
                elif key == "processing_time":
                    exif_dict["0th"][Base.Artist.value] = f"Time: {value:.2f}s"
            
            return True
            
        except Exception as e:
            logging.warning(f"Could not add metadata: {e}")
            return False
    
    def render_grid_overlay(self, collage_image: np.ndarray, 
                           line_color: tuple = (128, 128, 128), 
                           line_thickness: int = 1) -> np.ndarray:
        result = collage_image.copy()
        
        grid_height, grid_width = self.grid_size
        tile_width, tile_height = self.tile_size
        
        for i in range(1, grid_height):
            y = i * tile_height
            cv2.line(result, (0, y), (result.shape[1], y), line_color, line_thickness)
        
        for j in range(1, grid_width):
            x = j * tile_width
            cv2.line(result, (x, 0), (x, result.shape[0]), line_color, line_thickness)
        
        return result
    
    def create_high_resolution_output(self, individual: np.ndarray, 
                                    source_images: List[np.ndarray],
                                    scale_factor: int = 2) -> np.ndarray:
        original_tile_size = self.tile_size
        
        self.tile_size = (
            self.tile_size[0] * scale_factor,
            self.tile_size[1] * scale_factor
        )
        
        try:
            high_res_collage = self._render_collage(individual, source_images, preview=False)
            return high_res_collage
        finally:
            self.tile_size = original_tile_size