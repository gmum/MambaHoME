import glob
import os
from torch.utils.data import Dataset
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    Spacingd,
    RandRotate90d,
    RandZoomd,
    RandCropByLabelClassesd,
    RandGaussianSmoothd,
    SpatialPadd,
    RandRotated,
    ToTensord,
)

class CTDataset(Dataset):
    """
    Custom dataset class for loading 3D CT scans and corresponding labels.
    Supports MONAI-style transformations for training, validation, and testing.
    """

    def __init__(
        self,
        data_path: str = None,     # Directory containing CT images and labels
        mode: str = None,          # One of ['train', 'valid', 'test']
        patch_size: list = (96, 96, 96)  # Size of 3D patches for training
    ) -> None:
        """
        Initializes the dataset with image paths, label paths, and transformation pipelines.

        Args:
            data_path: Root directory where images and labels are stored.
            mode: Dataset mode ('train', 'valid', or 'test').
            patch_size: Desired spatial size for training patches.
        """

        self.data_path = data_path
        self.patch_size = patch_size
        assert mode in ["train", "valid", "test", None], "Invalid mode specified"
        self.mode = mode

        # Collect all CT and label file paths (assumes '*ct*.nii.gz' and '*labels*.nii.gz' naming convention)
        self.scans = sorted(glob.glob(os.path.join(self.data_path, "*/*ct*.nii.gz"), recursive=True))
        self.labels = sorted(glob.glob(os.path.join(self.data_path, "*/*labels*.nii.gz"), recursive=True))

        # Ensure the number of scans matches the number of labels
        assert len(self.scans) == len(self.labels), "Mismatch between scans and labels"

        # ----------------------------
        # Define training transforms (with data augmentation)
        # ----------------------------
        self.train_transform = Compose(
            [
                # Load image and label volumes from file
                LoadImaged(keys=["image", "label"], reader="NibabelReader"),

                # Ensure channel-first format (C, H, W, D)
                EnsureChannelFirstd(keys=["image", "label"]),

                # Reorient to RAS (Right-Anterior-Superior) coordinate system
                Orientationd(keys=["image", "label"], axcodes="RAS"),

                # Normalize CT intensities to [0, 1] within a fixed window
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175, a_max=250,
                    b_min=0.0, b_max=1.0,
                    clip=True,
                ),

                # Crop around the non-zero regions of the image
                CropForegroundd(keys=["image", "label"], source_key="image"),

                # Pad images and labels to the target patch size
                SpatialPadd(keys=["image", "label"],
                            spatial_size=self.patch_size,
                            mode="constant"),

                # Random zoom for augmentation
                RandZoomd(keys=["image", "label"],
                          min_zoom=1.3, max_zoom=1.5,
                          mode=["area", "nearest"],
                          prob=0.3),

                # Randomly crop positive and negative patches based on label presence
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.patch_size,
                    pos=3,               # More positive samples
                    neg=1,               # Fewer negative samples
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),

                # Randomly rotate by 90° up to 3 times
                RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),

                # Random intensity shift (brightness variation)
                RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),

                # Apply Gaussian smoothing with random sigma
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=0.2,
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    sigma_z=(0.5, 1.15),
                ),

                # Add random Gaussian noise
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),

                # Convert numpy arrays to PyTorch tensors
                ToTensord(keys=["image", "label"]),
            ]
        )

        # ----------------------------
        # Define validation transforms
        # ----------------------------
        self.valid_transform = Compose(
            [
                LoadImaged(keys=["image", "label"], reader="NibabelReader"),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175, a_max=250,
                    b_min=0.0, b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        # ----------------------------
        # Define test transforms
        # ----------------------------
        self.test_transform = Compose(
            [
                LoadImaged(keys=["image", "label"], reader="NibabelReader"),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ToTensord(keys=["image", "label"]),
            ]
        )

    def __getitem__(self, x):
        """
        Retrieve a single image-label pair by index, apply the proper transform, and return it.

        Args:
            x: Index of the sample to load.

        Returns:
            A dictionary with keys 'image' and 'label', each as a PyTorch tensor.
        """

        # Prepare dictionary pointing to the corresponding scan and label
        data_dict = {"image": self.scans[x], "label": self.labels[x]}

        # Apply the appropriate transformation pipeline depending on the mode
        if self.mode == "train":
            data_dict = self.train_transform(data_dict)
        elif self.mode == "valid":
            data_dict = self.valid_transform(data_dict)
        elif self.mode == "test":
            data_dict = self.test_transform(data_dict)
        else:
            raise NotImplementedError("Please provide a valid mode: train, valid, or test")

        return data_dict

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.scans)
