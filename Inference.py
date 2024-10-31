
import os
import argparse
from ImageSegmentor import Segmentor

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run segmentation.")
parser.add_argument("--experiment", type=str, required=True, help="Experiment ID")
parser.add_argument("--output_dir", type=str, default="output2", help="Name of output directory")
parser.add_argument("--model_veg", type=str, default="segveg_v2.pt", help="Vegetation segmentation model")
parser.add_argument("--model_ear", type=str, default="segear_v2.pt", help="Ear segmentation model")
parser.add_argument("--model_col", type=str, default="segcol_rf.pkl", help="Color segmentation model")
parser.add_argument("--save_patch", type=bool, default=False, help="Save the analyzed patch")
parser.add_argument("--save_images", type=bool, default=False, help="Save masked images")
parser.add_argument("--save_col_masks", type=bool, default=False, help="Save color masks")

args = parser.parse_args()
experiment = args.experiment
output_dir = args.output_dir
model_veg = args.model_veg
model_ear = args.model_ear
model_col = args.model_col
save_patch = args.save_patch
save_images = args.save_images
save_col_masks = args.save_col_masks

# set up directory
abspath = os.path.abspath(__file__)
d_name = os.path.dirname(abspath)
os.chdir(d_name)

# determine year depending on experiment id
if experiment == "ESWW006":
    year = "2022"
elif experiment == "ESWW007" or experiment == "ESWW008":
    year = "2023"
elif experiment == "ESWW009" or experiment == "ESWW010":
    year = "2024"

# set working directory
workdir = f'/home/anjonas/public/Public/Jonas/Data/{experiment}/ImagesNadir'

# get a list of directories to process
dirs = [f for f in os.listdir(workdir) if year in f]
dirs = [os.path.join(workdir, d) for d in dirs]
dirs = [d + "/JPEG_cam" for d in dirs]


# function to process directories
def run():
    dirs_to_process = dirs  # must be a list
    dir_output = f"{workdir}/{output_dir}"
    dir_ear_model = model_ear
    dir_veg_model = model_veg
    dir_col_model = model_col
    dir_patch_coordinates = f"{workdir}/Meta/patch_coordinates"
    image_pre_segmentor = Segmentor(dirs_to_process=dirs_to_process,
                                    dir_ear_model=dir_ear_model,
                                    dir_veg_model=dir_veg_model,
                                    dir_col_model=dir_col_model,
                                    dir_patch_coordinates=dir_patch_coordinates,
                                    dir_output=dir_output,
                                    img_type="JPG",
                                    save_patch=save_patch,
                                    save_images=save_images,
                                    save_col_masks=save_col_masks)
    image_pre_segmentor.process_images()


# process all
if __name__ == "__main__":
    run()
