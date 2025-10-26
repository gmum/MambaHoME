import os
import glob
import torch
import argparse
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureChannelFirstd,
    ToTensord,
    EnsureTyped,
    Invertd,
    ActivationsD,
    SaveImageD,
    AsDiscreteD
)
from models.MambaHoME import MambaHoME

# ----------------------------
# Parse arguments
# ----------------------------
parser = argparse.ArgumentParser(
    description="Inference using MambaHoME model"
)

parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    help="Path to folder with test NIfTI files"
)

parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to trained model checkpoint (.pth)"
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="inference",
    help="Directory to save output predictions"
)

parser.add_argument(
    "--roi_size",
    type=int,
    nargs=3,
    default=[192, 192, 48],
    help="ROI size for sliding window inference"
)

parser.add_argument(
    "--sw_batch_size",
    type=int,
    default=4,
    help="Sliding window batch size"
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for DataLoader"
)

parser.add_argument(
    "--overlap",
    type=float,
    default=0.5,
    help="Sliding window overlap fraction"
)

parser.add_argument(
    "--gpu",
    action="store_true",
    help="Use GPU if available"
)
args = parser.parse_args()

# ----------------------------
# Set environment and device
# ----------------------------
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Define transforms
# ----------------------------
inference_transform = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        Spacingd(keys=["image"], pixdim=(0.8, 0.8, 3.0), mode="bilinear"),
    ]
)

post_transform = Compose(
    [
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=inference_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True
        ),
        ActivationsD(keys="pred", softmax=True),
        AsDiscreteD(keys="pred", argmax=True),
        SaveImageD(
            keys="pred",
            meta_keys="pred_meta_dict",
            output_dir=args.output_dir,
            output_postfix="predictions",
            separate_folder=False,
            resample=False
        ),
    ]
)

# ----------------------------
# Load model and weights
# ----------------------------
model = MambaHoME(
    in_chans=1,
    out_chans=13,
    depths=[2, 2, 2, 2],
    feat_size=[48, 96, 192, 384],
    drop_path_rate=0,
    layer_scale_init_value=1e-6,
    hidden_size=768,
    norm_name="instance",
    conv_block=True,
    res_block=True,
    spatial_dims=3,
    expert_mult=2,
    moe_dropout=0.0,
    use_geglu=True,
    num_slots_per_expert_first=4,
    experts_list=[4, 8, 12, 16],
    experts_list_second=[8, 16, 24, 32],
    group_list=[2048, 1024, 512, 256],
)

checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

# ----------------------------
# Configure inferer
# ----------------------------
inferer = SlidingWindowInferer(
    roi_size=tuple(args.roi_size),
    sw_batch_size=args.sw_batch_size,
    sw_device="cuda" if device.type == "cuda" else "cpu",
    device="cuda" if device.type == "cuda" else "cpu",
    overlap=args.overlap,
    mode="gaussian",
    padding_mode="replicate",
)

# ----------------------------
# Prepare test dataset
# ----------------------------
test_images = sorted(glob.glob(os.path.join(args.data_path, "*.nii.gz")))
if not test_images:
    raise FileNotFoundError(f"No NIfTI files found in {args.data_path}")

test_data = [{"image": img} for img in test_images]
test_dataset = Dataset(data=test_data, transform=inference_transform)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)

# ----------------------------
# Run inference
# ----------------------------
with torch.no_grad():
    for test_data in test_loader:
        print(f"Processing {test_data['image_meta_dict']['filename_or_obj'][0]}")
        test_inputs = test_data["image"].to(device)

        test_data["pred"] = sliding_window_inference(
            inputs=test_inputs,
            roi_size=tuple(args.roi_size),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            sw_device="cuda" if device.type == "cuda" else "cpu",
            device="cuda" if device.type == "cuda" else "cpu",
            overlap=args.overlap,
            progress=True
        )

        test_data = [post_transform(i) for i in decollate_batch(test_data)]
        torch.cuda.empty_cache()
