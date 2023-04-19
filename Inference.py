
import os
from ImageSegmentor import EarSegmentor

abspath = os.path.abspath(__file__)
d_name = os.path.dirname(abspath)
os.chdir(d_name)

workdir = d_name + "/data/Inference/images"


def run():
    dirs_to_process = [workdir]  # must be a list
    dir_output = "output"
    dir_ear_model = "segear_ff.pt"
    dir_veg_model = "segveg_ff.pt"
    dir_col_model = "segcol_rf.pkl"
    dir_patch_coordinates = "data/Inference/coordinates"
    image_pre_segmentor = EarSegmentor(dirs_to_process=dirs_to_process,
                                       dir_ear_model=dir_ear_model,
                                       dir_veg_model=dir_veg_model,
                                       dir_col_model=dir_col_model,
                                       dir_patch_coordinates=dir_patch_coordinates,
                                       dir_output=dir_output,
                                       img_type="JPG")
    image_pre_segmentor.process_images()


if __name__ == "__main__":
    run()
