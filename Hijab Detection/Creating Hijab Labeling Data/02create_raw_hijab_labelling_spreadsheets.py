# Ease of life
import sys
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
# Working with images
from PIL import Image
import matplotlib.pyplot as plt
# Face recognition
import dlib
import face_recognition_models
import face_recognition
import openface # Used for face-alignment
# For gender prediction
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

# Length chosen based on experimentation
start = int(sys.argv[1]) * 34000
end = start + 34000

irancovid_path = "/home/gridsan/groups/irancovid"
images_path = f"{irancovid_path}/data/iran_instagram_p1and2_images_s0"

# Detect the gender for ALL aligned-face images
gender_detection = load_model(f"{irancovid_path}/models/gender_detection",
        compile=True)
image_paths = glob(f"{images_path}/aligned-faces/*")[start:end]
# Create data frame with image path and face
data = pd.DataFrame({"image_path": image_paths,
                     "image_name": ["_".join(image_path.split("/")[-1].split("_")[:-1]) for image_path in image_paths],
                     "face_number": [image_path.split("/")[-1].split("_")[-1].split(".")[0] for image_path in image_paths]})

images = np.asarray(data.image_path\
    .apply(lambda x: img_to_array(load_img(x, target_size=(200,200)))).to_list())

# Add gender prediction label to it
predictions = gender_detection.predict(images)
data["gender_prediction"] = [{0: "Male", 1: "Female"}[g] for g in np.argmax(predictions,axis=1)]
data.to_csv(f"/home/gridsan/akapl/playground/data/raw_hijab_labelling_{start}-{end}.csv", index=False)
