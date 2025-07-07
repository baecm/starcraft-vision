from torchvision.models.detection.transform import GeneralizedRCNNTransform, ImageList

from utils.logger import Logger

class CustomRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self, *args, do_normalize=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_normalize = do_normalize

    def forward(self, images, targets=None):
        images = [img for img in images]
        if targets is not None:
            targets = [{k: v for k, v in t.items()} for t in targets]

        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")

            if self.do_normalize:
                image = self.normalize(image)
            else:
                # Logger.info("Skipping normalization of images as do_normalize is set to False.")
                pass

            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list = [(s[0], s[1]) for s in image_sizes]

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets