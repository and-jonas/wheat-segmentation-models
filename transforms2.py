from typing import Callable, Tuple
from flash.image.segmentation.input_transform import InputTransform, prepare_target, remove_extra_dimensions
from flash.core.data.io.input import DataKeys
from torchvision import transforms as T
import kornia as K
from flash.core.data.transforms import ApplyToKeys, kornia_collate, KorniaParallelTransforms
import random
import numpy as np

from flash.core.data.properties import Properties
from flash.core.utilities.stages import RunningStage


class SemSegInputTransform(InputTransform):
    val_size: Tuple[int, int] = (700, 700)
    train_size: Tuple[int, int] = (700, 700)
    # predict_size: Tuple[int, int] = (1152, 1152)
    predict_size: Tuple[int, int] = (896, 896)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def train_per_sample_transform(self) -> Callable:
        transforms = [
            K.geometry.Resize(self.train_size, interpolation="nearest"),
            K.augmentation.PadTo((704, 704)),
            K.augmentation.RandomRotation(degrees=20),
            K.augmentation.RandomCrop((448, 448)),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomHorizontalFlip(p=0.5),
        ]

        return T.Compose([
            ApplyToKeys(
                [DataKeys.INPUT, DataKeys.TARGET],
                KorniaParallelTransforms(*transforms),
            ),
            ApplyToKeys(
                DataKeys.INPUT,
                K.augmentation.Normalize(mean=self.mean, std=self.std),
                # K.augmentation.RandomGaussianNoise(std=0.01, mean=0, p=1.0),
                K.augmentation.ColorJitter(brightness=0.5, contrast=0.25, saturation=0.25, hue=0.0, p=0.5),
            ),
        ])

    def val_per_sample_transform(self) -> Callable:
        transforms = [
            K.geometry.Resize(self.train_size, interpolation="nearest"),
            K.augmentation.PadTo((704, 704)),
        ]

        return T.Compose([
            ApplyToKeys(
                [DataKeys.INPUT, DataKeys.TARGET],
                KorniaParallelTransforms(*transforms)
            ),
            ApplyToKeys(
                DataKeys.INPUT,
                K.augmentation.Normalize(mean=self.mean, std=self.std),
            ),
        ])

    def predict_per_sample_transform(self) -> Callable:
        return ApplyToKeys(DataKeys.INPUT,
                           K.geometry.Resize(self.predict_size, interpolation="nearest"),
                           K.augmentation.Normalize(mean=self.mean, std=self.std))

    # def predict_input_per_sample_transform(self) -> Callable:
    #     # return  K.augmentation.Normalize(mean=self.mean, std=self.std)
    #     return K.geometry.Resize(self.predict_size, interpolation="nearest")
    #     # return T.Compose([
    #     #     # ApplyToKeys(
    #     #     #     [DataKeys.INPUT, DataKeys.TARGET],
    #     #     #     KorniaParallelTransforms(K.geometry.Resize(self.predict_size, interpolation="nearest"),
    #     #     #                              # K.augmentation.PadTo((704, 704)
    #     #     #                              )
    #     #     # ),
    #     #     ApplyToKeys(
    #     #         DataKeys.INPUT,
    #     #         K.augmentation.Normalize(mean=self.mean, std=self.std),
    #     #     ),
    #     # ])

    def collate(self) -> Callable:
        return kornia_collate

    def target_per_batch_transform(self) -> Callable:
        return prepare_target

    def predict_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions

    def serve_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions


# # TODO Can this work??
# class OptSemSegInputTransform(InputTransform):
#
#     def __init__(self, train_size=700, val_size=700,
#                  brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0,
#                  blur=False, kernel_size=1):
#         super().__init__()
#
#         # resolution
#         self.train_size: Tuple[int, int] = (train_size, train_size)
#         self.val_size: Tuple[int, int] = (val_size, val_size)
#         # predict_size: Tuple[int, int] = (1152, 1152)
#         # predict_size: Tuple[int, int] = (2016, 2016)
#         self.mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
#         self.std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
#
#         # color
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation
#         self.hue = hue
#
#         # blur
#         # self.blur: bool = blur
#         self.kernel_size: Tuple[int, int] = (kernel_size, kernel_size)
#
#         # padding
#         self.pad_to: Tuple[int, int] = (train_size + (16 - train_size % 16), train_size + (16 - train_size % 16))
#
#     def train_per_sample_transform(self) -> Callable:
#         transforms = [
#             K.geometry.Resize(self.train_size, interpolation="nearest"),
#             K.augmentation.PadTo(self.pad_to),
#             K.augmentation.RandomRotation(degrees=20),
#             K.augmentation.RandomCrop((448, 448)),
#             K.augmentation.RandomVerticalFlip(p=0.5),
#             K.augmentation.RandomHorizontalFlip(p=0.5),
#         ]
#
#         return T.Compose([
#             ApplyToKeys(
#                 [DataKeys.INPUT, DataKeys.TARGET],
#                 KorniaParallelTransforms(*transforms),
#             ),
#             ApplyToKeys(
#                 DataKeys.INPUT,
#                 K.augmentation.Normalize(mean=self.mean, std=self.std),
#                 K.augmentation.RandomGaussianBlur(kernel_size=self.kernel_size),
#                 K.augmentation.ColorJitter(brightness=self.brightness, contrast=self.contrast,
#                                            saturation=self.saturation, hue=self.hue, p=0.5),
#             ),
#         ])
#
#     def val_per_sample_transform(self) -> Callable:
#         transforms = [
#             K.geometry.Resize(self.val_size, interpolation="nearest"),
#             K.augmentation.PadTo(self.pad_to),
#         ]
#
#         return T.Compose([
#             ApplyToKeys(
#                 [DataKeys.INPUT, DataKeys.TARGET],
#                 KorniaParallelTransforms(*transforms)
#             ),
#             ApplyToKeys(
#                 DataKeys.INPUT,
#                 K.augmentation.Normalize(mean=self.mean, std=self.std),
#             ),
#         ])
#
#     def predict_per_sample_transform(self) -> Callable:
#         return ApplyToKeys(DataKeys.INPUT,
#                            K.geometry.Resize(self.predict_size, interpolation="nearest"),
#                            K.augmentation.Normalize(mean=self.mean, std=self.std))
#
#     def collate(self) -> Callable:
#         return kornia_collate
#
#     def target_per_batch_transform(self) -> Callable:
#         return prepare_target
#
#     def predict_per_batch_transform(self) -> Callable:
#         return remove_extra_dimensions
#
#     def serve_per_batch_transform(self) -> Callable:
#         return remove_extra_dimensions


# Workaround
def set_input_transform_options(head="deeplabv3plus",
                                size=700,
                                crop_factor=0.64,
                                blur_kernel_size=3,
                                p_color_jitter=0.0,
                                ):

    # padding is still necessary, even though 448*448 is sampled later
    divisor = 16 if head == "deeplabv3plus" else 32
    pad_to = (size + (divisor - size % divisor), size + (divisor - size % divisor))

    sigma = (1, 1)

    p_blur = 0 if blur_kernel_size == 1 else 0.5
    blur_kernel_size = (blur_kernel_size, blur_kernel_size)

    crop_size_ = int(crop_factor*size)
    crop_size = (crop_size_, crop_size_)

    # determine predict_size depending on (i) original size of patches to predict,
    # (ii) re-sizing during training and (iii) original rescaling factor for ESWW006 data segear (=1/2.25)
    # or (=1) for ESWW006 segveg
    # predict_size = 4000 * (size/1200) * (1/2.25)  # <==================================================================
    # predict_size = 4000 * (size/1200)  # <=============================================================================

    # 2022-02-15 EARS
    predict_size = 1200 * (size/1200)
    # 2022-02-15 EARS END

    multiplier = np.ceil(predict_size / 16)
    predict_size = int(multiplier * 16)

    class SemSegInputTransform(InputTransform):

        # val_size: Tuple[int, int] = (val_size, val_size)
        # train_size: Tuple[int, int] = (train_size, train_size)
        # predict_size: Tuple[int, int] = (1152, 1152)
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

        def train_per_sample_transform(self) -> Callable:

            transforms = [
                K.geometry.Resize(size, interpolation="nearest"),
                # K.augmentation.PadTo(pad_to),
                K.augmentation.RandomRotation(degrees=20),
                # K.augmentation.RandomCrop((448, 448)),
                K.augmentation.RandomCrop(crop_size),
                # K.augmentation.PadTo(pad_to_after_crop),
                K.augmentation.RandomVerticalFlip(p=0.5),
                K.augmentation.RandomHorizontalFlip(p=0.5),
            ]

            return T.Compose([
                ApplyToKeys(
                    [DataKeys.INPUT, DataKeys.TARGET],
                    KorniaParallelTransforms(*transforms),
                ),
                ApplyToKeys(
                    DataKeys.INPUT,
                    K.augmentation.Normalize(mean=self.mean, std=self.std),
                    K.augmentation.ColorJiggle(brightness=0.5, contrast=0.25, saturation=0.25,
                                               hue=0, p=p_color_jitter),
                    K.augmentation.RandomGaussianBlur(kernel_size=blur_kernel_size, sigma=sigma, p=p_blur),
                ),
            ])

        def val_per_sample_transform(self) -> Callable:
            transforms = [
                K.geometry.Resize(size, interpolation="nearest"),
                # K.augmentation.RandomCrop(crop_size),  # Required if head == "fpn"
                K.augmentation.PadTo((704, 704))
            ]

            return T.Compose([
                ApplyToKeys(
                    [DataKeys.INPUT, DataKeys.TARGET],
                    KorniaParallelTransforms(*transforms)
                ),
                ApplyToKeys(
                    DataKeys.INPUT,
                    K.augmentation.Normalize(mean=self.mean, std=self.std),
                ),
            ])

        def predict_per_sample_transform(self) -> Callable:
            return ApplyToKeys(DataKeys.INPUT,
                               # K.geometry.Resize((2336, 2336), interpolation="nearest"),
                               K.geometry.Resize(predict_size, interpolation="nearest"),
                               K.augmentation.Normalize(mean=self.mean, std=self.std))

        def collate(self) -> Callable:
            return kornia_collate

        def target_per_batch_transform(self) -> Callable:
            return prepare_target

        def predict_per_batch_transform(self) -> Callable:
            return remove_extra_dimensions

        def serve_per_batch_transform(self) -> Callable:
            return remove_extra_dimensions

    return SemSegInputTransform

