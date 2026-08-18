# src/model/utils/transforms.py
import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform

class CustomRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self, in_channels: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels

    def normalize(self, image):
        if self.image_mean is None or self.image_std is None:
            return image
        
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)

        if image.shape[0] != len(mean):
            if len(mean) == 3:
                num_frames = image.shape[0] // 3
                mean = mean.repeat(num_frames)
                std = std.repeat(num_frames)
            else:
                return image

        return (image - mean[:, None, None]) / std[:, None, None]
