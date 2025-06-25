
from ImageSegmentor import Segmentor
import os

workdir = '/home/anjonas/public/Public/Jonas/Data/EXT/5'

dirs = [f for f in os.listdir(workdir) if '2024' in f]
dirs = [os.path.join(workdir, d) for d in dirs]
dirs = [d + "/JPEG" for d in dirs]

# function to process directories
def run():
    dirs_to_process = dirs  # must be a list
    image_pre_segmentor = Segmentor(dirs_to_process=dirs_to_process,
                                    dir_ear_model="segear_v2.pt",
                                    dir_veg_model="segveg_v2.pt",
                                    dir_col_model="segcol_rf.pkl",
                                    dir_output = f'{workdir}/output_test',
                                    img_type="png",
                                    save_patch=True,
                                    save_images=True,
                                    save_col_masks=True,
                                    dir_patch_coordinates = None,
                                    skip_processed=False,
                                    scale_f=2.5, crop_size=4480)
    image_pre_segmentor.process_images()


# process all
if __name__ == "__main__":
    run()
