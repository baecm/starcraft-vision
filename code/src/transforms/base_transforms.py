import torchvision.transforms as T

def get_transform(is_train: bool):
    transforms = [T.ToTensor()]
    if is_train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
