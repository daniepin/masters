import matplotlib.pyplot as plt
import os
import nibabel as nib
from PIL import Image


img1 = Image.open("/home/neutron/dev/thesis/exploration/brain/slice_69.png")
img2 = Image.open("/home/neutron/dev/thesis/exploration/brain-proc-mask/slice_69.png")
img3 = Image.open("/home/neutron/dev/thesis/exploration/brain-proc/slice_69.png")

f, axarr = plt.subplots(1, 3)
axarr[0].imshow(img1)
axarr[1].imshow(img2)
axarr[2].imshow(img3)
plt.setp(axarr, xticks=[], yticks=[])
plt.annotate(
    "",
    xy=(195, 170),
    xytext=(164, 170),
    xycoords="figure points",
    arrowprops=dict(facecolor="blue", shrink=0.5),
)
plt.annotate(
    "",
    xy=(317, 170),
    xytext=(294, 170),
    xycoords="figure points",
    arrowprops=dict(facecolor="blue", shrink=0.5),
)
plt.savefig("/home/neutron/dev/thesis/exploration/brain.png")

"""

img = nib.load("/home/neutron/dev/thesis/data/ixi/IXI-T1/IXI002-Guys-0828-T1.nii.gz")
img_data = img.get_fdata()
for i in range(img_data.shape[2]):
    plt.figure()
    plt.imshow((img_data[:, :, i]), cmap="gray")
    # plt.title(f"slice {i}")
    plt.axis("off")  # Turn off axis labels
    # Pause to display each slice (adjust the pause duration as needed)
    plt.pause(0.1)

    output_filename = os.path.join(
        "/home/neutron/dev/thesis/exploration/brain", f"slice_{i}.png"
    )
    plt.savefig(output_filename, bbox_inches="tight", pad_inches=0)
    plt.close()
"""
