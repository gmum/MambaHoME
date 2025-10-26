import os
import glob
import argparse
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    SaveImaged
)


class ConvertSpacing:
    def __init__(self,
                 input_path: str = None,
                 output_path: str = None,
                 spacing: list = (0.8, 0.8, 3.0)
                 ) -> None:
        """
        Initializes the ConvertSpacing class.

        Args:
            input_path: Directory containing input images and labels.
            output_path: Directory to save the transformed images and labels.
            spacing: Target voxel spacing for resampling in (x, y, z) order.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.spacing = spacing

        # Create output directory if it does not exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Collect all image and label files
        self.images = sorted(
            glob.glob(
                os.path.join(self.input_path, "images/*.nii.gz"),
                recursive=True
            )
        )
        self.labels = sorted(
            glob.glob(
                os.path.join(self.input_path, "labels/*.nii.gz"),
                recursive=True
            )
        )

    def get_output_dir(self, filepath):
        """
        Generates output directory for a given file.

        Args:
            filepath: Path of the input file.

        Returns:
            Output directory one level up from the file's original folder, inside output_path.
        """
        original_dir = os.path.dirname(filepath)
        output_dir = os.path.join(
            self.output_path,
            os.path.basename(original_dir)
        )
        return output_dir

    def __getitem__(self, x):
        """
        Loads and transforms the x-th image and label pair.

        Args:
            x: Index of the image/label pair to process.

        Returns:
            A dictionary containing the transformed 'image' and 'label'.
        """
        image_path = self.images[x]
        label_path = self.labels[x]

        # Determine output directories
        output_dir_image = self.get_output_dir(image_path)
        output_dir_label = self.get_output_dir(label_path)

        # Create directories if they do not exist
        if not os.path.exists(output_dir_image):
            os.makedirs(output_dir_image)
        if not os.path.exists(output_dir_label):
            os.makedirs(output_dir_label)

        # Define transformation pipeline
        transform = Compose([
            LoadImaged(
                keys=["image", "label"]
            ),
            EnsureChannelFirstd(
                keys=["image", "label"]
            ),
            Orientationd(
                keys=["image", "label"],
                axcodes="LPS"
            ),
            Spacingd(
                keys=["image", "label"],
                pixdim=self.spacing,
                mode=["bilinear", "nearest"]
            ),
            SaveImaged(
                keys=["image"],
                output_dir=output_dir_image,
                separate_folder=False,
                resample=False
            ),
            SaveImaged(
                keys=["label"],
                output_dir=output_dir_image,
                separate_folder=False,
                resample=False
            )
        ])

        # Apply transformations
        data_dict = {"image": image_path, "label": label_path}
        data_dict = transform(data_dict)

        # Print shapes of processed image and label
        print(data_dict["image"].shape, data_dict["label"].shape)

        # Ensure image and label shapes match
        assert data_dict['image'].shape == data_dict['label'].shape, \
            f"Image and label sizes do not match! Image size: {data_dict['image'].shape}, Label size: {data_dict['label'].shape}"

        return data_dict

    def __len__(self):
        # Return number of images
        return len(self.images)


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(
        description="Convert spacing of medical images and labels."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input dataset (with 'images' and 'labels' folders)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save preprocessed dataset."
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[0.8, 0.8, 3.0],
        help="Target voxel spacing as three floats, e.g., --spacing 0.8 0.8 3.0"
    )

    args = parser.parse_args()

    # Initialize converter with CLI arguments
    convert_spacing = ConvertSpacing(
        input_path=args.input_path,
        output_path=args.output_path,
        spacing=args.spacing
    )

    # Process all images
    for i in range(len(convert_spacing)):
        data = convert_spacing[i]
        print(f"Processed data {i+1}/{len(convert_spacing)}")
