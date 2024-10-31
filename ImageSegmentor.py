import os
import torch
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import copy
import pickle
import imageio
import cv2
import flash
from flash.image import SemanticSegmentation, SemanticSegmentationData
from transforms2 import set_input_transform_options
import SegmentationFunctions
import utils

import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# define input transform for each task
transform_ear = set_input_transform_options(train_size=600,
                                            crop_factor=0.64,
                                            p_color_jitter=0,
                                            blur_kernel_size=1,
                                            predict_scale=(1/2.25))
transform_veg = set_input_transform_options(train_size=700,
                                            crop_factor=0.64,
                                            p_color_jitter=0,
                                            blur_kernel_size=1,
                                            predict_scale=1)


class Segmentor:

    def __init__(self, dirs_to_process, dir_patch_coordinates, dir_output,
                 dir_ear_model, dir_veg_model, dir_col_model,
                 img_type,
                 save_patch, save_images, save_col_masks):
        self.dirs_to_process = dirs_to_process
        self.dir_patch_coordinates = Path(dir_patch_coordinates) if dir_patch_coordinates is not None else None
        self.dir_ear_model = Path(dir_ear_model)
        self.dir_veg_model = Path(dir_veg_model)
        self.dir_col_model = Path(dir_col_model)
        self.save_patch = save_patch
        self.save_images = save_images
        self.save_col_masks = save_col_masks
        # output paths
        self.path_output = Path(dir_output)
        # -- image patch
        self.path_patch = self.path_output / "Patches"
        # -- binary masks
        self.path_ear_mask = self.path_output / "SegEar" / "Mask"
        self.path_veg_mask = self.path_output / "SegVeg" / "Mask"
        self.path_source_mask = self.path_output / "SegVeg" / "MaskSource"
        # -- overlays
        self.path_ear_overlay = self.path_output / "SegEar" / "Overlay"
        self.path_veg_overlay = self.path_output / "SegVeg" / "Overlay"
        # -- color masks
        self.path_ear_col_mask = self.path_output / "SegEar" / "ColMask"
        self.path_veg_col_mask = self.path_output / "SegVeg" / "ColMask"
        self.path_source_col_mask = self.path_output / "SegVeg" / "ColMaskSource"
        # -- masked images
        self.path_img_veg = self.path_output / "SegImg" / "VegMask"
        self.path_img_ear = self.path_output / "SegImg" / "EarMask"
        self.path_img_source = self.path_output / "SegImg" / "SourceMask"
        # -- csv
        self.path_stats = self.path_output / "Stats"
        # helpers
        self.image_type = img_type
        # load segmentation models
        self.ear_model = SemanticSegmentation.load_from_checkpoint(self.dir_ear_model)
        self.veg_model = SemanticSegmentation.load_from_checkpoint(self.dir_veg_model)
        with open(self.dir_col_model, 'rb') as model:
            self.col_model = pickle.load(model)
        # instantiate trainer
        self.trainer = flash.Trainer(max_epochs=1, accelerator='gpu', devices=[0])

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        for path in [self.path_output, self.path_patch, self.path_ear_mask, self.path_ear_overlay, self.path_veg_mask,
                     self.path_source_mask,
                     self.path_veg_overlay, self.path_ear_col_mask, self.path_veg_col_mask,
                     self.path_source_col_mask,
                     self.path_img_veg, self.path_img_ear, self.path_img_source,
                     self.path_stats]:
            path.mkdir(parents=True, exist_ok=True)

    def file_feed(self):
        """
        Creates a list of paths to images that are to be processed
        :param img_type: a character string, the file extension, e.g. "JPG"
        :return: paths
        """
        # get all files and their paths
        files = []
        for d in self.dirs_to_process:
            files.extend(glob.glob(f'{d}/*.{self.image_type}'))
        # removes all Reference images
        files = [f for f in files if "Ref" not in f]
        return files

    def segment_image(self, patch, model, transform):
        """
        Segments an image using a pre-trained semantic segmentation model.
        Creates probability maps, binary segmentation masks, and overlay
        :param image: The image to be processed as an numpy array.
        :param coordinates: A tuple of coordinates defining the ROI.
        :return: The resulting binary segmentation mask.
        """

        # image axes must be re-arranged
        patch_ = np.moveaxis(patch, 2, 0) / 255.0

        # create a datamodule from numpy array
        datamodule = SemanticSegmentationData.from_numpy(
            predict_data=[patch_],
            num_classes=2,
            train_transform=transform,
            val_transform=transform,
            test_transform=transform,
            predict_transform=transform,
            batch_size=1,  # required
        )

        # make predictions
        predictions = self.trainer.predict(
            model=model,
            datamodule=datamodule,
        )

        # extract predictions
        predictions = predictions[0][0]['preds']

        # transform predictions to probabilities and labels
        probabilities = torch.softmax(predictions, dim=0)
        probabilities_ear = probabilities[0]
        mask = torch.argmax(probabilities, dim=0)
        mask_8bit = mask*255

        # make overlay
        M = mask_8bit.ravel()
        M = np.expand_dims(M, -1)
        out_mask = np.dot(M, np.array([[1, 0, 0, 0.33]]))
        out_mask = np.reshape(out_mask, newshape=(patch.shape[0], patch.shape[1], 4))
        out_mask = out_mask.astype("uint8")
        mask = Image.fromarray(out_mask, mode="RGBA")
        img_ = Image.fromarray(patch, mode="RGB")
        img_ = img_.convert("RGBA")
        img_.paste(mask, (0, 0), mask)
        img_ = img_.convert('RGB')
        overlay = np.asarray(img_)

        return probabilities_ear, np.asarray(mask_8bit), overlay

    def process_images(self):
        """
        Wrapper, processing all images
        """
        self.prepare_workspace()
        files = self.file_feed()

        for file in files:

            print(file)

            # read image
            base_name = os.path.basename(file)
            stem_name = Path(file).stem
            png_name = base_name.replace("." + self.image_type, ".png")
            csv_name = base_name.replace("." + self.image_type, ".csv")
            img = Image.open(file)
            pix = np.array(img)

            # sample patch from image using coordinate file
            if self.dir_patch_coordinates is not None:
                c = pd.read_table(f'{self.dir_patch_coordinates}/{stem_name}.txt', sep=",").iloc[0, :].tolist()
                patch = pix[c[2]:c[3], c[0]:c[1]]
            else:
                patch = pix

            if self.save_patch:
                imageio.imwrite(self.path_patch / png_name, patch)

            # (1) segment ears in patch ================================================================================
            proba, ear_mask, ear_overlay = self.segment_image(
                patch,
                model=self.ear_model,
                transform=transform_ear
            )

            # output paths
            ear_mask_name = self.path_ear_mask / png_name
            overlay_name = self.path_ear_overlay / base_name

            # save output
            imageio.imwrite(ear_mask_name, ear_mask)
            imageio.imwrite(overlay_name, ear_overlay)

            # (2) segment vegetation in patch  =========================================================================
            proba, veg_mask, veg_overlay = self.segment_image(
                patch,
                model=self.veg_model,
                transform=transform_veg
            )

            # output paths
            veg_mask_name = self.path_veg_mask / png_name
            overlay_name = self.path_veg_overlay / base_name

            # save output
            imageio.imwrite(veg_mask_name, veg_mask)
            imageio.imwrite(overlay_name, veg_overlay)

            # get the "source" mask
            source_mask_name = self.path_source_mask / png_name
            source_mask = np.maximum(veg_mask - ear_mask, 0)
            imageio.imwrite(source_mask_name, source_mask)

            # (3) color-based segmentation =============================================================================

            # downscale
            x_new = int(patch.shape[0] * (1 / 2))
            y_new = int(patch.shape[1] * (1 / 2))
            patch_seg = cv2.resize(patch, (y_new, x_new), interpolation=cv2.INTER_LINEAR)

            # extract pixel features
            color_spaces, descriptors, descriptor_names = SegmentationFunctions.get_color_spaces(patch_seg)
            descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

            # get pixel label probabilities
            segmented_flatten_probs = self.col_model.predict(descriptors_flatten)

            # restore image
            preds = segmented_flatten_probs.reshape((descriptors.shape[0], descriptors.shape[1]))

            # convert to mask
            mask = np.zeros_like(patch_seg)
            mask[np.where(preds == "brown")] = (102, 61, 20)
            mask[np.where(preds == "yellow")] = (255, 204, 0)
            mask[np.where(preds == "green")] = (0, 100, 0)

            # upscale
            x_new = int(patch_seg.shape[0] * (2))
            y_new = int(patch_seg.shape[1] * (2))
            mask = cv2.resize(mask, (y_new, x_new), interpolation=cv2.INTER_NEAREST)

            # (4) generate outputs =====================================================================================

            # veg col mask
            veg_col_mask_name = self.path_veg_col_mask / png_name
            veg_col_mask = copy.copy(mask)
            veg_col_mask[np.where(veg_mask == 0)] = (0, 0, 0)

            # ear col mask
            ear_col_mask_name = self.path_ear_col_mask / png_name
            ear_col_mask = copy.copy(mask)
            ear_col_mask[np.where(ear_mask == 0)] = (0, 0, 0)

            # source col mask
            source_col_mask_name = self.path_source_col_mask / png_name
            source_col_mask = copy.copy(veg_col_mask)
            source_col_mask[np.where(ear_mask == 255)] = (0, 0, 0)

            if self.save_col_masks:
                imageio.imwrite(veg_col_mask_name, veg_col_mask)
                imageio.imwrite(ear_col_mask_name, ear_col_mask)
                imageio.imwrite(source_col_mask_name, source_col_mask)

            # remove background and/or objects - original patches
            veg_image = copy.copy(patch)
            veg_image[np.where(veg_mask == 0)] = (0, 0, 0)

            ear_image = copy.copy(patch)
            ear_image[np.where(ear_mask == 0)] = (0, 0, 0)

            source_image = copy.copy(veg_image)
            source_image[np.where(ear_mask == 255)] = (0, 0, 0)

            # save masked images
            if self.save_images:
                imageio.imwrite(self.path_img_veg / png_name, veg_image)
                imageio.imwrite(self.path_img_ear / png_name, ear_image)
                imageio.imwrite(self.path_img_source / png_name, source_image)

            # get color properties
            desc, desc_names = utils.color_index_transformation(patch)

            # specify levels at which to extract the data
            levels = [veg_mask, ear_mask, source_mask]
            level_names = ["veg", "ear", "source"]

            # loop over the different levels of interest (veg, ear, source)
            stats = []
            stat_names = []
            for l, ln in zip(levels, level_names):
                # loop over the different color features
                for d, dn in zip(desc, desc_names):
                    s, sn = utils.index_distribution(image=d, feature_name=dn, level_id=ln, level_mask=l)
                    stats.extend(s)
                    stat_names.extend(sn)
            # write to data frame
            dfa = pd.DataFrame([stats])
            dfa.columns = stat_names
            dfa.insert(loc=0, column='image_id', value=stem_name)

            # get pixel fractions
            # total cover per fraction
            veg_cover = len(np.where(veg_mask == 255)[0]) / (4000 * 4000)
            ear_cover = len(np.where(ear_mask == 255)[0]) / (4000 * 4000)
            source_cover = len(np.where(source_mask == 255)[0]) / (4000 * 4000)
            cover_stat_names = ["veg_cover", "ear_cover", "source_cover"]
            dfa[cover_stat_names] = [[veg_cover, ear_cover, source_cover]]

            # cover within fraction per color
            ear_green = len(np.where(ear_col_mask[:, :, 1] == 100)[0]) / len(np.where(ear_mask == 255)[0])
            ear_chlr = len(np.where(ear_col_mask[:, :, 1] == 204)[0]) / len(np.where(ear_mask == 255)[0])
            ear_necr = len(np.where(ear_col_mask[:, :, 1] == 61)[0]) / len(np.where(ear_mask == 255)[0])
            veg_green = len(np.where(veg_col_mask[:, :, 1] == 100)[0]) / len(np.where(veg_mask == 255)[0])
            veg_chlr = len(np.where(veg_col_mask[:, :, 1] == 204)[0]) / len(np.where(veg_mask == 255)[0])
            veg_necr = len(np.where(veg_col_mask[:, :, 1] == 61)[0]) / len(np.where(veg_mask == 255)[0])
            source_green = len(np.where(source_col_mask[:, :, 1] == 100)[0]) / len(np.where(veg_mask == 255)[0])
            source_chlr = len(np.where(source_col_mask[:, :, 1] == 204)[0]) / len(np.where(veg_mask == 255)[0])
            source_necr = len(np.where(source_col_mask[:, :, 1] == 61)[0]) / len(np.where(veg_mask == 255)[0])
            status_stat_names = ["ear_green", "ear_chlr", "ear_necr", "veg_green", "veg_chlr", "veg_necr",
                                 "source_green", "source_chlr", "source_necr"]
            dfa[status_stat_names] = [[ear_green, ear_chlr, ear_necr, veg_green, veg_chlr, veg_necr,
                                       source_green, source_chlr, source_necr]]
            dfa.to_csv(self.path_stats / csv_name, index=False)






