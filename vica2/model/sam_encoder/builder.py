from .sam2_encoder import SAM2VisionTower
def build_sam_tower(vision_tower="nkkbr/hiera-base-plus-in-sam2.1", **kwargs):
    return SAM2VisionTower(vision_tower, **kwargs)