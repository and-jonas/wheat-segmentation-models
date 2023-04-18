
import os
import re
from EarSegmentor import EarSegmentor

workdir = '/home/anjonas/public/Public/Jonas/Data/FPWW002/wheat_head_annotations/year_mix_80_20/validation/images'
# workdir = '/home/anjonas/public/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation/all_annotations/images'


def run():
    dirs_to_process = [workdir]  # must be a list
    task = "SegEar"
    # task = "SegVeg"
    dir_output = f'{workdir}/Output'
    dir_model = "/projects/SegEar/segear_ff.pt"
    # dir_model = "/projects/SegVeg2/segveg_ff.pt"
    dir_patch_coordinates = None
    image_pre_segmentor = EarSegmentor(dirs_to_process=dirs_to_process,
                                       task=task,
                                       dir_model=dir_model,
                                       dir_patch_coordinates=dir_patch_coordinates,
                                       dir_output=dir_output,
                                       img_type="png")
    image_pre_segmentor.process_images()


if __name__ == "__main__":
    run()
