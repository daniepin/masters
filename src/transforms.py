from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    ScaleIntensityd,
    Spacingd,
    RandFlipd,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    RandSpatialCropd,
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
