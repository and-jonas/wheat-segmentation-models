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
                 save_patch, save_images, save_col_masks,
                 skip_processed,
                 scale_f, crop_size):
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
        # additional args
        self.skip_processed = skip_processed
        self.scale_f = scale_f
        self.crop_size = crop_size

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

        # removes all processed files
        if self.skip_processed:
            img_ids = [os.path.basename(x) for x in files]
            processed = [os.path.basename(x) for x in glob.glob(f'{self.path_stem_ear_overlay}/*.JPG')]
            proc_idx = [idx for idx, img in enumerate(img_ids) if img not in processed]
            files = [files[i] for i in proc_idx]

        return files[2:]

    @staticmethod
    def make_overlay(patch, mask, colors=[(1, 0, 0, 0.25)]):
        img_ = Image.fromarray(patch, mode="RGB")
        img_ = img_.convert("RGBA")
        class_labels = np.unique(mask)
        for i, v in enumerate(class_labels[1:]):
            r, g, b, a = colors[i]
            M = np.where(mask == v, 255, 0)
            M = M.ravel()
            M = np.expand_dims(M, -1)
            out_mask = np.dot(M, np.array([[r, g, b, a]]))
            out_mask = np.reshape(out_mask, newshape=(patch.shape[0], patch.shape[1], 4))
            out_mask = out_mask.astype("uint8")
            M = Image.fromarray(out_mask, mode="RGBA")
            img_.paste(M, (0, 0), M)
        img_ = img_.convert('RGB')
        overlay = np.asarray(img_)

        return overlay

    @staticmethod
    def tile_image(patch, split):
        h, w = split
        height, width, _ = patch.shape
        new_h = int(height/h + height/h % 32)
        new_w = int(width/w + width/w % 32)
        X_points = utils.start_points(size=width, split_size=new_w, overlap=width/w % 32)
        Y_points = utils.start_points(size=height, split_size=new_h, overlap=height/h % 32)
        splitted = []
        count = 0
        for i in Y_points:
            for j in X_points:
                splitted.append(patch[i:i + new_h, j:j + new_w])
                count += 1

        return splitted

    @staticmethod
    def extract_predictions(output):
        patch_masks = []
        for i in range(len(output)):
            # get predictions
            predictions = output[i][0]['preds']
            # transform predictions to probabilities and labels
            probabilities = torch.softmax(predictions, dim=0)
            # probabilities_ear = probabilities[0]
            mask = torch.argmax(probabilities, dim=0)
            mask_8bit = np.asarray(np.uint8((mask*255) / (np.max(np.uint8(mask)))))
            patch_masks.append(mask_8bit)
        return patch_masks

    @staticmethod
    def merge_predictions(patch, masks, split):
        # existing and new width and height of patches
        h, w = split
        height, width, _ = patch.shape
        new_h = int(height/h + height/h % 32)
        new_w = int(width/w + width/w % 32)
        m_h = int(height/h % 32)
        m_w = int(width/w % 32)
        p_h = int(height/h)
        p_w = int(width/w)
        int_h = new_h - (new_h - p_h) / 2
        int_w = new_w - (new_w - p_w) / 2

        # get the cropping coordinates for each mask
        seq_h1 = ([0] + [int(m_h/2)] * (h - 2) + [m_h])*h
        seq_h2 = ([p_h] + [int(int_h)] * (h - 2) + [new_h])*h
        seq_w1 = ([0] + [int(m_w/2)] * (w - 2) + [m_w])*w
        seq_w2 = ([p_w] + [int(int_w)] * (w - 2) + [new_w])*w

        # crop masks so that their merged product will match the original image
        index = range(h*w)
        masks_ = [m[seq_h1[i]:seq_h2[i], seq_w1[i]:seq_w2[i]] for m, i in zip(masks, index)]

        rows = []
        for i in range(0, h*w, w):
            row = np.concatenate(masks_[i:i + w], axis=1)  # Concatenate 4 patches horizontally
            rows.append(row)
        merged_image = np.concatenate(rows, axis=0)
        return merged_image

    def segment_image(self, patch, model, transform, colors, split):
        """
        Segments an image using a pre-trained semantic segmentation model.
        Creates probability maps, binary segmentation masks, and overlay
        :param image: The image to be processed as a numpy array.
        :param coordinates: A tuple of coordinates defining the ROI.
        :return: The resulting binary segmentation mask.
        """

        # ADJUST SIZE  <================================================================================================
        x_new = int((patch.shape[0]/split[0] - patch.shape[0]/split[0] % 16)*split[0])
        y_new = int((patch.shape[1]/split[1] - patch.shape[1]/split[1] % 16)*split[1])
        patch = cv2.resize(patch, (y_new, x_new), interpolation=cv2.INTER_LINEAR)

        # tile image into overlapping patches, if needed
        patches = self.tile_image(patch, split=split)

        # image axes must be re-arranged
        patches_ = [np.moveaxis(p, 2, 0) / 255.0 for p in patches]

        # create a datamodule from numpy array
        datamodule = SemanticSegmentationData.from_numpy(
            predict_data=patches_,
            num_classes=2,
            train_transform=transform,
            val_transform=transform,
            test_transform=transform,
            predict_transform=transform,
            batch_size=1,  # required
            num_workers=48
        )

        # make predictions
        predictions = self.trainer.predict(
            model=model,
            datamodule=datamodule,
        )

        # extract the predictions for each patch of the image
        masks_8bit = self.extract_predictions(output=predictions)

        # merge the predictions into a single mask
        mask_8bit = self.merge_predictions(patch, masks=masks_8bit, split=split)

        # create the overlay
        overlay = self.make_overlay(patch, mask=mask_8bit, colors=colors)

        return np.asarray(mask_8bit), overlay

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

            # scale image to match image resolution from ESWW006 - ESWW010
            pix = np.asarray(img)
            pix = cv2.resize(pix, dsize=None, fx=self.scale_f, fy=self.scale_f)

            # crop
            pix = utils.center_crop(pix, crop_size=self.crop_size)

            # sample patch from image using coordinate file
            if self.dir_patch_coordinates is not None:
                c = pd.read_table(f'{self.dir_patch_coordinates}/{stem_name}.txt', sep=",").iloc[0, :].tolist()
                patch = pix[c[2]:c[3], c[0]:c[1]]
            else:
                patch = pix

            if self.save_patch:
                imageio.imwrite(self.path_patch / png_name, patch)

            # (1) segment ears in patch ================================================================================
            ear_mask, ear_overlay = self.segment_image(
                patch,
                model=self.ear_model,
                transform=transform_ear,
                colors=[(1, 0, 0, 0.25)],
                split=(2, 2)
            )

            # output paths
            ear_mask_name = self.path_ear_mask / png_name
            overlay_name = self.path_ear_overlay / base_name

            # save output
            imageio.imwrite(ear_mask_name, ear_mask)
            imageio.imwrite(overlay_name, ear_overlay)

            # (2) segment vegetation in patch  =========================================================================
            veg_mask, veg_overlay = self.segment_image(
                patch,
                model=self.veg_model,
                transform=transform_veg,
                colors=[(1, 0, 0, 0.25)],
                split=(2, 2)
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






