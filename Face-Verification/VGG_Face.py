# common dependencies
import os
from os import path
import warnings
import time
import logging

# 3rd party dependencies
import tensorflow as tf

from deepface.basemodels import VGGFace, Facenet512
from deepface.commons import distance as dst

# -----------------------------------
# configurations for dependencies

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_version = int(tf.__version__.split(".", maxsplit=1)[0])
if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
# -----------------------------------

def build_model(model_name):

    """
    This function builds a deepface model
    Parameters:
            model_name (string): face recognition or facial attribute model
                    VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
                    Age, Gender, Emotion, Race for facial attributes

    Returns:
            built deepface model
    """

    # singleton design pattern
    global model_obj

    models = {"VGG-Face": VGGFace.loadModel,
              "Facenet512": Facenet512.loadModel}

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj:
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
        else:
            raise ValueError(f"Invalid model_name passed - {model_name}")

    return model_obj[model_name]

# def represent(
#     img_objs,
#     model_name="VGG-Face",
#     normalization="base",
# ):
#     resp_objs = []

#     model = build_model(model_name)

#     # ---------------------------------

#     for img, region, confidence in img_objs:
#         # custom normalization
#         img = functions.normalize_input(img=img, normalization=normalization)

#         # represent
#         if "keras" in str(type(model)):
#             # new tf versions show progress bar and it is annoying
#             embedding = model.predict(img, verbose=0)[0].tolist()
#         else:
#             # SFace and Dlib are not keras models and no verbose arguments
#             embedding = model.predict(img)[0].tolist()

#         resp_obj = {}
#         resp_obj["embedding"] = embedding
#         resp_obj["facial_area"] = region
#         resp_obj["face_confidence"] = confidence
#         resp_objs.append(resp_obj)

#     return resp_objs

def represent(
    img,
    model_name="VGG-Face",
):

    model = build_model(model_name)

    # ---------------------------------
    
    # # custom normalization
    # img = functions.normalize_input(img=img, normalization=normalization)

    # represent
    if "keras" in str(type(model)):
        # new tf versions show progress bar and it is annoying
        embedding = model.predict(img, verbose=0)[0].tolist()
    else:
        # SFace and Dlib are not keras models and no verbose arguments
        embedding = model.predict(img)[0].tolist()

    resp_obj = {}
    resp_obj["embedding"] = embedding

    return resp_obj


def verify(
    img1,
    img2,
    model_name="VGG-Face",
    distance_metric="cosine",
):

    tic = time.time()

    # --------------------------------
    # now we will find the face pair with minimum distance
    img1_embedding_obj = represent(
        img=img1,
        model_name=model_name,
    )

    img2_embedding_obj = represent(
        img=img2,
        model_name=model_name,
    )

    img1_representation = img1_embedding_obj["embedding"]
    img2_representation = img2_embedding_obj["embedding"]

    if distance_metric == "cosine":
        distance = dst.findCosineDistance(img1_representation, img2_representation)
    elif distance_metric == "euclidean":
        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
    elif distance_metric == "euclidean_l2":
        distance = dst.findEuclideanDistance(
            dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation)
        )
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    # -------------------------------
    threshold = dst.findThreshold(model_name, distance_metric)

    toc = time.time()

    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "similarity_metric": distance_metric,
        "time": round(toc - tic, 2),
    }

    return resp_obj