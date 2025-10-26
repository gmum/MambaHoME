import os
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from models.mambaHoME import MambaHoME
from monai.data import DataLoader, decollate_batch
import gc

from data_loader import CTDataset
import argparse
from lightning.fabric import Fabric

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    default="../dataset",
    type=str
)

parser.add_argument(
    "--data_train",
    default="",
    type=str
)

parser.add_argument(
    "--data_val",
    default="",
    type=str
)

parser.add_argument(
    "--batch_size",
    default=2,
    type=int
)

parser.add_argument(
    "--skip_val",
    default=1,
    type=int
)

parser.add_argument(
    "--classes",
    default=3,
    type=int
)

parser.add_argument(
    "--epochs",
    default=800,
    type=int
)

parser.add_argument(
    "--lr",
    default=1e-4,
    type=float
)

parser.add_argument(
    "--weight_decay",
    default=1e-4,
    type=float
)

parser.add_argument(
    "--optimizer",
    default="AdamW",
    type=str
)

parser.add_argument(
    "--scheduler",
    default="CALR",
    type=str
)

parser.add_argument(
    "--patch_size",
    default=(96, 96, 96),
    type=tuple
)

parser.add_argument(
    "--feature_size",
    default=48,
    type=int
)

parser.add_argument(
    "--use_checkpoint",
    default=False,
    type=bool
)

parser.add_argument(
    "--num_workers",
    default=12,
    type=int
)

parser.add_argument(
    "--pin_memory",
    default=True,
    type=bool
)

parser.add_argument(
    "--use_pretrained",
    default=False,
    type=bool
)

parser.add_argument(
    "--load_checkpoint",
    default=False,
    type=bool
)

parser.add_argument(
    "--checkpoint_name",
    default="",
    type=str
)

parser.add_argument(
    "--model",
    default="MambaHoME",
    type=str
)

parser.add_argument(
    "--parallel",
    default=True,
    type=bool
)

parser.add_argument(
    "--num_devices",
    default=4,
    type=int
)

parser.add_argument(
    "--strategy",
    default="ddp",
    type=str
)
args = parser.parse_args()


def load(model, model_dict):
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    else:
        state_dict = model_dict
    current_model_dict = model.state_dict()
    for k in current_model_dict.keys():
        if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
            print(k)
    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else
        current_model_dict[k]
        for k in current_model_dict.keys()}
    model.load_state_dict(new_state_dict, strict=True)
    return model


def save_checkpoint(global_step, model, optimizer, scheduler):
    save_dict = {
        "step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    save_path = os.path.join(args.data_train, f"{args.model}_{args.epochs}.pth")
    fabric.save(save_path, save_dict)


def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    checkpoint = fabric.load(checkpoint)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    global_step = checkpoint["step"]
    return global_step


def train(global_step, train_loader, valid_loader, dice_val_best, global_step_best, fabric: Fabric):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        torch.cuda.empty_cache()
        gc.collect()
        step += 1
        x, y = (batch["image"], batch["label"])

        logit_map = model(x)
        loss = loss_function(logit_map, y)

        optimizer.zero_grad()
        fabric.backward(loss)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_iterator.set_description(
            f"Training ({global_step} / {len(train_loader) * args.epochs} Steps) (loss={loss:.5f})"
        )

        if (global_step % (args.skip_val * len(train_loader)) == 0 and global_step != 0) or \
                global_step == len(train_loader) * args.epochs:

            epoch_iterator_val = tqdm(valid_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                save_checkpoint(global_step=global_step, model=model, optimizer=optimizer, scheduler=scheduler)
                print(f"Model saved. Best Dice: {dice_val_best:.4f}, Current Dice: {dice_val:.4f}")
            else:
                print(f"Model NOT saved. Best Dice: {dice_val_best:.4f}, Current Dice: {dice_val:.4f}")
            scheduler.step()
        global_step += 1

    return global_step, dice_val_best, global_step_best


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            val_outputs = sliding_window_inference(val_inputs, args.patch_size, 1, model, sw_device="cuda",
                                                   device="cuda")

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]

            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            dice_metric1(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(f"Validate ({global_step} / {10.0} Steps)")
        mean_dice_val1 = dice_metric1.aggregate().item()
        dice_metric1.reset()
    return mean_dice_val1


fabric = Fabric(devices=args.num_devices, strategy=args.strategy)
fabric.launch()

train_ds = CTDataset(args.data_train, mode="train", patch_size=args.patch_size)
valid_ds = CTDataset(args.data_val, mode="valid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

post_label = AsDiscrete(to_onehot=args.classes)
post_pred = AsDiscrete(argmax=True, to_onehot=args.classes)
dice_metric1 = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=args.pin_memory)
valid_loader = DataLoader(valid_ds, batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_memory)

# Model selection
if args.model == "MambaHoME":
    model = MambaHoME(
        in_chans=1,
        out_chans=16,
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
else:
    raise NotImplementedError("This model does not exist!")

if args.optimizer == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-4)
elif args.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    raise NotImplementedError("Optimizer not found!")

model, optimizer = fabric.setup(model, optimizer)
train_loader, valid_loader = fabric.setup_dataloaders(train_loader, valid_loader)

if args.scheduler == "CALR":
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs // args.skip_val, verbose=True)
else:
    raise NotImplementedError("Learning rate scheduler not found!")

dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
global_step = 0

if args.load_checkpoint:
    global_step = load_checkpoint(checkpoint=args.checkpoint_name, model=model,
                                  optimizer=optimizer, scheduler=scheduler)

while global_step < len(train_loader) * args.epochs:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, valid_loader, dice_val_best, global_step_best, fabric
    )
