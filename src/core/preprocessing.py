import torch
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger("EdgeVisionEngine.Preprocessing")

class ImagePreprocessor:
    """
    Standardized preprocessing pipeline optimized for edge vision backbones.
    Handles resizing, center cropping, and ImageNet normalization.
    """
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        logger.info(f"Initialized ImagePreprocessor with target resolution {image_size}x{image_size}")

    def __call__(self, image_path: str) -> torch.Tensor:
        """Loads an image from disk and applies the transformation pipeline, adding a batch dimension."""
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image)
            # Add batch dimension -> shape: (1, 3, H, W)
            return tensor.unsqueeze(0)
        except Exception as e:
            logger.error(f"Failed to preprocess image at {image_path}: {e}")
            raise