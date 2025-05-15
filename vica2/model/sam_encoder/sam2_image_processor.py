# Adapted from https://github.com/facebookresearch/sam2/blob/main/sam2/utils/transforms.py

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Resize
from transformers.image_processing_utils import BaseImageProcessor
from typing import List, Union, Tuple, Dict, Any
from PIL import Image


class SAM2ImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        resolution: int = 1024,
        mask_threshold: float = 0.0,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
        do_resize: bool = True,
        do_normalize: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.to_tensor = ToTensor()
        self.resize = Resize((self.resolution, self.resolution))
        self.normalize = Normalize(mean=self.mean, std=self.std)

    def preprocess(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, Any]:
        if isinstance(images, np.ndarray):
            if images.ndim == 4:
                images = [Image.fromarray(img.astype(np.uint8)) for img in images]
            else:
                raise ValueError("Expected 4D numpy array with shape (B, H, W, C)")

        if not isinstance(images, list):
            raise ValueError("Input images should be a list of np.ndarray or a 4D np.ndarray")

        processed = []
        for img in images:
            img = self.to_tensor(img)
            if self.do_resize:
                img = self.resize(img)
            if self.do_normalize:
                img = self.normalize(img)
            processed.append(img)

        pixel_values = torch.stack(processed, dim=0)

        return {"pixel_values": pixel_values}

    def transform_coords(
        self, coords: torch.Tensor, normalize: bool = False, orig_hw: Tuple[int, int] = None
    ) -> torch.Tensor:
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.resolution
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize: bool = False, orig_hw: Tuple[int, int] = None
    ) -> torch.Tensor:
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks
