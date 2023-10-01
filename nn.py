from io import BytesIO
from typing import TypeAlias, cast

from cv2 import imencode
from numpy import uint8
from numpy.typing import NDArray
from tensorflow import keras
from tensorflow_hub import KerasLayer

Model: TypeAlias = keras.Sequential


def to_pill_image(image: NDArray[uint8]):
    # なんか一旦ファイルに落とさないとまともに動かないのでごまかす
    # TODO: Issueが嘘を言ってないか、npのみとBytesIOで比較（既存ファイルで）
    # https://github.com/GantMan/nsfw_model/issues/89#issuecomment-770139533
    # https://stackoverflow.com/a/52865864
    success, cv2_buf = imencode(".png", image)
    if not success:
        raise Exception("imencode failed")
    bytes_buf = BytesIO(cv2_buf)

    # モデルが食える形にしてやる
    # TODO: BytesIO使わなくてもいけるのでは……(リサイズと正規化できればいい気がする)
    keras_img = keras.preprocessing.image.load_img(bytes_buf, target_size=(224, 224))
    pill_image = keras.preprocessing.image.img_to_array(keras_img) / 255.0

    # TODO: typing
    return pill_image


def load_model(path: str) -> Model:
    return cast(
        Model,
        keras.models.load_model(path, custom_objects={"KerasLayer": KerasLayer}),
    )


def classify_nd(model: Model, nd_images) -> list[dict[str, float]]:
    model_preds = model.predict(
        nd_images,
        verbose=0,  # プログレスバーを消す #type: ignore
    )

    categories = ["drawings", "hentai", "neutral", "porn", "sexy"]

    def map_preds(single_preds) -> dict[str, float]:
        single_probs: dict[str, float] = {}
        for j, pred in enumerate(single_preds):
            single_probs[categories[j]] = float(pred)
        return single_probs

    probs: list[dict[str, float]] = [
        map_preds(single_preds) for single_preds in model_preds
    ]
    return probs
