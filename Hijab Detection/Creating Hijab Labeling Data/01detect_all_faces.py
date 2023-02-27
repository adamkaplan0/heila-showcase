# Ease of life
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

irancovid_path = "/home/gridsan/groups/irancovid"
images_path = f"{irancovid_path}/data/iran_instagram_p1and2_images_s0"

# Extract all of the faces from the downloaded images
face_detector = dlib.get_frontal_face_detector()
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
face_aligner = openface.AlignDlib(predictor_68_point_model)

def process_image(image_source_path, image_destination_folder, size=200):
    try:
        image = face_recognition.load_image_file(image_source_path)
        for i, face_rectangle in enumerate(face_detector(image, 1)):
            face_aligned = face_aligner\
                .align(size, image, face_rectangle,
                        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            Image.fromarray(face_aligned)\
                .save(f"{image_destination_folder}/{image_source_path.split('/')[-1].split('.')[-2]}_{i}.png")
    except:
        print(f"Error with image: {image_source_path}")

for image_path in tqdm(glob(f"{images_path}/*.jpg")):
    process_image(image_path, f"{images_path}/aligned-faces")

