import monai.transforms as mts


transforms = {
    "ixi": {
        "train": mts.Compose(
            [
                mts.EnsureChannelFirst(),
                mts.RandFlip(prob=0.5, spatial_axis=0),
                mts.Resize((40, 40, 40)),
                mts.NormalizeIntensity(),
            ]
        ),
        "val": mts.Compose(
            [
                mts.EnsureChannelFirst(),
                mts.Resize((40, 40, 40)),
                mts.NormalizeIntensity(),
            ]
        ),
    },
    "ukb": {
        "train": mts.Compose(
            [
                mts.EnsureChannelFirst(),
                mts.CropForeground(),
                mts.RandFlip(prob=0.5, spatial_axis=0),
                mts.Spacing(pixdim=[2, 2, 2], mode="bilinear"),
                mts.ResizeWithPadOrCrop(
                    spatial_size=(40 + 5, 40 + 5, 40 + 5), mode="constant", value=0.0
                ),
                mts.RandSpatialCrop(roi_size=40, random_center=True, random_size=False),
                mts.ScaleIntensity(minv=0.0, maxv=1.0),
            ]
        ),
        "val": mts.Compose(
            [
                mts.EnsureChannelFirst(),
                mts.CropForeground(),
                mts.Spacing(pixdim=[2, 2, 2], mode="bilinear"),
                mts.ResizeWithPadOrCrop(40, mode="constant", value=0.0),
                mts.ScaleIntensity(minv=0.0, maxv=1.0),
            ]
        ),
    },
}
