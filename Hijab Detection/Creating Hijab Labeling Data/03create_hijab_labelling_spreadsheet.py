import pandas as pd
from glob import glob
from tqdm import tqdm
import shutil

data = pd.read_csv("/home/gridsan/akapl/playground/hijab_labelling.csv")
# For each image copy it and the corresponding face to a separate folder
for iname in tqdm(data.image_name):
    try:
        shutil.copyfile(f"/home/gridsan/groups/irancovid/data/iran_instagram_p1and2_images_s0/{iname}.jpg",
                f"/home/gridsan/groups/irancovid/data/iran_instagram_p1and2_images_s0/hijab-images/{iname}.jpg")
        shutil.copyfile(f"/home/gridsan/groups/irancovid/data/iran_instagram_p1and2_images_s0/aligned-faces/{iname}_0.png",
                f"/home/gridsan/groups/irancovid/data/iran_instagram_p1and2_images_s0/hijab-faces/{iname}.png")
    except Exception as e:
        print(f"ERROR: {iname} -- {e}")
