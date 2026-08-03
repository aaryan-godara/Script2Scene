import os
from pathlib import Path
from typing import List, Tuple
from video_assembler.models import Scene

class ImageMatcher:
    def __init__(self, images_dir: Path):
        self.images_dir = Path(images_dir)
        self.available_images = {f.name: f for f in self.images_dir.iterdir() if f.is_file()}

    def match_scenes(self, scenes: List[Scene]) -> Tuple[List[Scene], List[dict], List[dict]]:
        """
        Matches images to scenes.
        Returns (updated_scenes, errors, warnings)
        """
        errors = []
        warnings = []
        
        for scene in scenes:
            matched_images = []
            
            # 1. Explicit Mapping
            for img_name in scene.images:
                if img_name in self.available_images:
                    matched_images.append(img_name)
                else:
                    errors.append({
                        "type": "MISSING_EXPLICIT_IMAGE",
                        "scene_id": scene.scene_id,
                        "file": img_name
                    })
            
            # 2. Normalized Scene-ID Match (if no images matched explicitly)
            if not matched_images and not scene.images:
                potential_names = [
                    f"scene_{scene.scene_id:03d}.png",
                    f"scene_{scene.scene_id:02d}.png",
                    f"scene_{scene.scene_id}.png",
                    f"{scene.scene_id:03d}.png",
                    f"{scene.scene_id}.png"
                ]
                
                for p_name in potential_names:
                    if p_name in self.available_images:
                        matched_images.append(p_name)
                        break
                        
                if not matched_images:
                    errors.append({
                        "type": "MISSING_IMAGE",
                        "scene_id": scene.scene_id
                    })
                    
            scene.images = matched_images
            
        # Check for unused images (warnings)
        used_images = set(img for scene in scenes for img in scene.images)
        for available_img in self.available_images:
            if available_img not in used_images:
                warnings.append({
                    "type": "UNUSED_IMAGE",
                    "file": available_img
                })
                
        return scenes, errors, warnings
