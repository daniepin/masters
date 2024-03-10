import random
from monai.transforms import (
    NormalizeIntensityd,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CropForegroundd,
    RandFlipd,
    Spacingd,
    ResizeWithPadOrCropd,
    RandSpatialCropd,
    ScaleIntensityd,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    Lambdad,
    Transform,
    MapTransform,
    GaussianSmoothd
)


def get_transforms(dataset, img_size, pixdim) -> dict:
    tranforms = {
        "ixi": {
            "train": Compose(
                [
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    CropForegroundd(keys=["image"], source_key="image"),
                    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                    Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear")),
                    ResizeWithPadOrCropd(
                        keys=["image"],
                        spatial_size=(
                            img_size[0] + 10,
                            img_size[1] + 10,
                            img_size[2] + 10,
                        ),
                        mode="constant",
                        value=0.0,
                    ),
                    RandSpatialCropd(
                        keys=["image"],
                        roi_size=img_size,
                        random_center=True,
                        random_size=False,
                    ),
                    NormalizeIntensityd(keys=["image"]),
                ]
            ),
            "val": Compose(
                [
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    CropForegroundd(keys=["image"], source_key="image"),
                    Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear")),
                    ResizeWithPadOrCropd(
                        keys=["image"],
                        spatial_size=(
                            img_size[0],
                            img_size[1],
                            img_size[2],
                        ),
                        mode="constant",
                        value=0.0,
                    ),
                    NormalizeIntensityd(keys=["image"]),
                ]
            ),
        },
        "ukb": {
            "train": Compose(
                [
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    CropForegroundd(keys=["image"], source_key="image"),
                    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                    Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear")),
                    ResizeWithPadOrCropd(
                        keys=["image"],
                        spatial_size=(
                            img_size[0] + 10,
                            img_size[1] + 10,
                            img_size[2] + 10,
                        ),
                        mode="constant",
                        value=0.0,
                    ),
                    RandSpatialCropd(
                        keys=["image"],
                        roi_size=img_size,
                        random_center=True,
                        random_size=False,
                    ),
                    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                ]
            ),
            "val": Compose(
                [
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    CropForegroundd(keys=["image"], source_key="image"),
                    Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear")),
                    ResizeWithPadOrCropd(
                        keys=["image"],
                        spatial_size=(
                            img_size[0],
                            img_size[1],
                            img_size[2],
                        ),
                        mode="constant",
                        value=0.0,
                    ),
                    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                ]
            ),
        },
    }

    return tranforms[dataset]


# Define the transformation
rand_transform_list = [
    RandRotate90d(
        keys=["image"], prob=1, max_k=3
    ),  # Rotate the image by 0, 90, 180, or 270 degrees
    RandGaussianNoised(keys=["image"], prob=1),  # Add Gaussian noise to the image
    RandGaussianSmoothd(
        keys=["image"], prob=1
    ),  # Apply Gaussian smoothing to the image
    RandAdjustContrastd(
        keys=["image"], prob=1
    ),  # Randomly adjust the contrast of the image
    #Lambdad(keys=["image"], func=lambda x: x * 0),  # Make the image all black with a probability of 0.1
]

rand_list = [
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    Lambdad,
]


class MyCustomTransform(Transform):
    def __init__(self, some_param):
        self.some_param = some_param

    def __call__(self, data):
        # Implement your custom transformation here
        transformed_data = data * self.some_param
        return transformed_data
#random_transform = Lambdad(func=lambda x: random.choice(rand_list)(x))


class RandomApplyTransform(MapTransform):
    def __init__(self, keys, transforms):
        super().__init__(keys)
        self.transforms = transforms

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            try:
                transform = random.choice(self.transforms)(keys=key, func=lambda x: x * 0)
            except:
                transform = random.choice(self.transforms)(keys=key, prob=1)
            
            d = transform(d)
        return d


def get_transforms_ood(img_size, pixdim) -> dict:
    return {
        "val": Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                CropForegroundd(keys=["image"], source_key="image"),
                Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear")),
                ResizeWithPadOrCropd(
                    keys=["image"],
                    spatial_size=(
                        img_size[0],
                        img_size[1],
                        img_size[2],
                    ),
                    mode="constant",
                    value=0.0,
                ),
                ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                # random_transform,
                # RandGaussianNoised(keys=["image"], prob=1, std=1),
                #rand_transform_list[random.randint(0, len(rand_transform_list) - 1)],
                #Lambdad(
                #    keys=["image"],
                #    func=lambda x: rand_list[random.randint(0, len(rand_list))](
                #        keys=["image"]
                #    ),
                #),
                #RandomApplyTransform(
                #    keys=["image"],
                #    transforms=[RandRotate90d, RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd]
                #),
                GaussianSmoothd(keys=["image"], sigma=1.5),
            ]
        ),
        "test": Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                CropForegroundd(keys=["image"], source_key="image"),
                Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear")),
                ResizeWithPadOrCropd(
                    keys=["image"],
                    spatial_size=(
                        img_size[0],
                        img_size[1],
                        img_size[2],
                    ),
                    mode="constant",
                    value=0.0,
                ),
                ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            ]
        ),
    }
