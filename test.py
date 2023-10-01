import sys

import cv2
import numpy as np
from tensorflow import keras

from nn import classify_nd, load_model

model = load_model("./model/model.keras")

path = sys.argv[1]


def run_keras():
    keras_img = keras.preprocessing.image.load_img("7.png", target_size=(224, 224))
    keras_image = keras.preprocessing.image.img_to_array(keras_img)
    img = cv2.cvtColor(keras_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("keras.png", img)
    keras_image /= 255.0

    result = classify_nd(
        model,
        np.asarray([keras_image]),
    )

    return result


def run_cv2():
    img = cv2.imread("7.png", cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite("cv2.png", img)
    img = img.astype("float") / 255.0

    result = classify_nd(
        model,
        np.asarray([img]),
    )
    return result


print(run_cv2())
print(run_keras())
