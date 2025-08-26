# src/model/CustomRCNNTransform.py
from torchvision.models.detection.transform import GeneralizedRCNNTransform, ImageList

class CustomRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self, *args, do_normalize=True, do_resize=True, **kwargs):
        """
        do_normalize: True면 mean/std로 normalize, False면 스킵
        do_resize   : True면 부모의 resize() 수행, False면 원본 크기 유지
        """
        super().__init__(*args, **kwargs)
        self.do_normalize = do_normalize
        self.do_resize = do_resize

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

            if self.do_resize:
                image, target_index = self.resize(image, target_index)
            # else: no-op (원본 크기 유지)

            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list = [(s[0], s[1]) for s in image_sizes]

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets
