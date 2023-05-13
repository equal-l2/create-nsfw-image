# この辺をモデル使って自動化してみたい
# https://gamingchahan.com/ecchi/
# https://kennycason.com/posts/2017-10-01-genetic-algorithm-draw-images-japanese.html

from io import BytesIO
from random import randint, random

import cv2
import numpy as np
import tensorflow as tf
from nsfw_detector import predict

print("Loading model")
model = predict.load_model("./model/mobilenet_v2_140_224")
print("Ended: Loading model")

IMG_SHAPE = (250, 200, 3)

max_value = 0
while True:
    # create image
    # TODO: 遺伝要素(現時点ではランダムだけ)
    # TODO: 楕円も追加する?
    image = np.zeros(IMG_SHAPE, np.uint8)
    for _ in range(1000):
        x = randint(0, IMG_SHAPE[1])
        w = randint(x, IMG_SHAPE[1]) - x
        y = randint(0, IMG_SHAPE[0])
        h = randint(y, IMG_SHAPE[0]) - y
        color = tuple(randint(0, 255) for _ in range(3))
        # print(y,h,x,w)
        rect = np.full((h, w, 3), color, np.uint8)

        # 重なった部分だけを上手いこと透過させたいのでクロップしてからweighted sum
        # https://stackoverflow.com/a/56472613
        opacity = random()
        sub_img = image[y : y + h, x : x + w]
        # print(rect.shape, sub_img.shape)
        cv2.addWeighted(sub_img, opacity, rect, 1 - opacity, 0, dst=sub_img)
        image[y : y + h, x : x + w] = sub_img

    # なんか一旦ファイルに落とさないとまともに動かないのでごまかす
    # TODO: Issueが嘘を言ってないか、npのみとBytesIOで比較（既存ファイルで）
    # https://github.com/GantMan/nsfw_model/issues/89#issuecomment-770139533
    # https://stackoverflow.com/a/52865864
    success, cv2_buf = cv2.imencode(".png", image)
    bytes_buf = BytesIO(cv2_buf)

    # モデルが食える形にしてやる
    # TODO: BytesIO使わなくてもいけるのでは……(リサイズと正規化できればいい気がする)
    tf_img = tf.keras.preprocessing.image.load_img(bytes_buf, target_size=(224, 224))
    tf_image = tf.keras.preprocessing.image.img_to_array(tf_img) / 255.0

    result = predict.classify_nd(
        model,
        np.asarray([tf_image]),
        # プログレスバーを消す
        predict_args={
            "verbose": 0,
        },
    )

    value = result[0]["hentai"]

    if value > max_value:
        print("new value:", value)
        max_value = value
        cv2.imshow("Image", image)
        cv2.waitKey(0)
