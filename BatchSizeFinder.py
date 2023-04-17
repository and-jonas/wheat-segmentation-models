
# modified for flash from
# https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1

import torch
from flash.image import SemanticSegmentation, SemanticSegmentationData
from flash import Trainer
import gc
import numpy as np


def find_max_batch_size(
        backbone: str,
        head: str,
        dataset_size: int,
        transform,  # the SemSegInputTransform
        init_batch_size: int,
        strategy: str,
        loss_fn: torch.nn.Module,
        metrics: list,
        optimizer: str,
        learning_rate: float,
        momentum: float,
        n_iterations: int,
        max_batch_size: int = None,
) -> int:
    batch_size = init_batch_size
    # instantiate model
    model = SemanticSegmentation(
        pretrained="imagenet",
        backbone=backbone,
        head=head,
        num_classes=2,
        metrics=metrics,
        loss_fn=loss_fn,
        optimizer=(optimizer, {"momentum": momentum}),
        learning_rate=learning_rate,
    )
    while True:
        if max_batch_size is not None and batch_size >= (max_batch_size + 2):
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = dataset_size
            break
        try:
            for _ in range(n_iterations):
                if max_batch_size is not None and batch_size >= (max_batch_size + 2):
                    break
                if batch_size >= dataset_size:
                    break
                datamodule = None  # might be required to get rid of the old datamodule
                datamodule = SemanticSegmentationData.from_folders(
                    train_folder="/projects/SegEar/data_yearmix/train/images",
                    train_target_folder="/projects/SegEar/data_yearmix/train/masks",
                    train_transform=transform,
                    val_transform=transform,
                    test_transform=transform,
                    predict_transform=transform,
                    num_classes=2,
                    batch_size=batch_size,
                )
                # instantiate the trainer
                trainer = Trainer(
                    max_steps=1,  # this underestimates memory usage (??)
                    move_metrics_to_cpu=True,
                    gpus=[3],
                    precision=16,
                    reload_dataloaders_every_n_epochs=1,
                    enable_progress_bar=True,
                    enable_checkpointing=False
                )
                # run test
                trainer.finetune(model, datamodule=datamodule, strategy=strategy)
                batch_size += 2
                del datamodule, trainer
                torch.cuda.empty_cache()
                gc.collect()
        except:
            if batch_size <= 10:
                batch_size -= 3
            elif batch_size <= 30:
                batch_size -= 4
            else:
                batch_size -= 5
            torch.cuda.empty_cache()
            gc.collect()
            break
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return batch_size


def find_max_batch_size_simple(
        backbone: str,
        head: str,
        max_batch_size: int,
        size: int,
        crop_factor: float
) -> int:
    if backbone == "resnet18" and head == "deeplabv3plus":
        batch_size = np.floor(10800000 / ((size * crop_factor) ** 2))
    elif backbone == "resnet34" and head == "deeplabv3plus":
        batch_size = np.floor(8400000 / ((size * crop_factor) ** 2))
    elif backbone == "resnet50" and head == "deeplabv3plus":
        batch_size = np.floor(4600000 / ((size * crop_factor) ** 2))
    elif backbone == "resnet101" and head == "deeplabv3plus":
        batch_size = np.floor(3200000 / ((size * crop_factor) ** 2))
    elif backbone == "resnet18" and head == "fpn":
        batch_size = np.floor(10000000 / ((size * crop_factor) ** 2))
    elif backbone == "resnet34" and head == "fpn":
        batch_size = np.floor(8400000 / ((size * crop_factor) ** 2))
    elif backbone == "resnet50" and head == "fpn":
        batch_size = np.floor(5000000 / ((size * crop_factor) ** 2))
    elif backbone == "resnet101" and head == "fpn":
        batch_size = np.floor(3600000 / ((size * crop_factor) ** 2))
    elif backbone == "resnet18" and head == "unetplusplus":
        batch_size = np.floor(3600000 / ((size * crop_factor) ** 2))
    elif backbone == "resnet34" and head == "unetplusplus":
        batch_size = np.floor(3200000 / ((size * crop_factor) ** 2))
    elif backbone == "resnet50" and head == "unetplusplus":
        batch_size = np.floor(1800000 / ((size * crop_factor) ** 2))
    elif backbone == "resnet101" and head == "unetplusplus":
        batch_size = np.floor(1400000 / ((size * crop_factor) ** 2))
    if batch_size >= max_batch_size:
        batch_size = max_batch_size

    return int(batch_size)


