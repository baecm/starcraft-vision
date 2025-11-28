# src/model/CustomRCNNTransform.py
from torchvision.models.detection.transform import GeneralizedRCNNTransform, ImageList
from omegaconf import ListConfig

class CustomRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self, *args, do_normalize=True, do_resize=True, **kwargs):
        """
        do_normalize: True면 mean/std로 normalize, False면 스킵
        do_resize   : True면 부모의 resize() 수행, False면 원본 크기 유지
        """
        # -------- Hydra ListConfig 방어 코드 --------
        def _to_list_if_listconfig(x):
            return list(x) if isinstance(x, ListConfig) else x

        # (1) positional args 쪽 처리
        if args:
            args = list(args)
            # GeneralizedRCNNTransform 시그니처:
            # __init__(self, min_size, max_size, image_mean=None, image_std=None, fixed_size=None)
            if len(args) >= 1:  # min_size
                args[0] = _to_list_if_listconfig(args[0])
            if len(args) >= 3:  # image_mean
                args[2] = _to_list_if_listconfig(args[2])
            if len(args) >= 4:  # image_std
                args[3] = _to_list_if_listconfig(args[3])
            args = tuple(args)

        # (2) keyword args 쪽 처리 (builder에서 min_size=..., image_mean=... 로 넘길 수도 있어서)
        for key in ("min_size", "image_mean", "image_std"):
            if key in kwargs:
                kwargs[key] = _to_list_if_listconfig(kwargs[key])

        # max_size도 혹시 ListConfig로 올 수 있으니 방어적으로 캐스팅
        if "max_size" in kwargs and isinstance(kwargs["max_size"], ListConfig):
            v = kwargs["max_size"]
            # 보통 scalar일 텐데 혹시 리스트면 첫 원소 사용
            kwargs["max_size"] = int(v[0] if isinstance(v, (list, ListConfig)) else v)
        
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
                raise ValueError(
                    f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}"
                )

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