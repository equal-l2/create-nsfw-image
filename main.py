# この辺をモデル使って自動化してみたい
# https://gamingchahan.com/ecchi/
# https://kennycason.com/posts/2017-10-01-genetic-algorithm-draw-images-japanese.html

import heapq
from dataclasses import dataclass
from io import BytesIO
from random import choice, randint, random, sample, uniform
from typing import NewType, Self, TypeVar, cast

import cv2
import numpy as np
import tensorflow as tf
from nsfw_detector import predict
from numpy.typing import NDArray

print("Loading model")
model = predict.load_model("./model/mobilenet_v2_140_224")
print("Ended: Loading model")

ImgShape = NewType("ImgShape", tuple[int, int, int])
Color = NewType("Color", tuple[int, int, int])


@dataclass(kw_only=True)
class Gene:
    x: int
    y: int
    w: int
    h: int
    color: Color
    opacity: float

    @classmethod
    def random(cls, shape: ImgShape) -> Self:
        # TODO: 楕円も追加する?
        x = randint(0, shape[1])
        w = randint(x, shape[1]) - x
        y = randint(0, shape[0])
        h = randint(y, shape[0]) - y
        color = cast(Color, tuple(randint(0, 255) for _ in range(3)))
        opacity = random()

        assert x + w <= shape[1]
        assert y + h <= shape[0]

        return Gene(x=x, w=w, y=y, h=h, color=color, opacity=opacity)

    def crossbreed(self, other: Self, shape: ImgShape) -> Self:
        _T = TypeVar("_T", int, float)

        def minmax_delta(a: _T, b: _T, delta: _T) -> tuple[_T, _T]:
            e_min = min(a, b) - delta
            e_max = max(a, b) + delta
            return e_min, e_max

        def clamp(x: _T, n_min: _T, n_max: _T) -> _T:
            if x < n_min:
                return n_min
            if x > n_max:
                return n_max
            return x

        def gen_new_color(a: Color, b: Color, delta: int) -> Color:
            return cast(
                Color,
                tuple(
                    clamp(randint(*minmax_delta(p[0], p[1], delta)), 0, 255)
                    for p in zip(a, b)
                ),
            )

        delta_int = 10
        delta_float = 0.1

        new_x = clamp(randint(*minmax_delta(self.x, other.x, delta_int)), 0, shape[1])
        new_w = clamp(
            randint(*minmax_delta(self.w, other.w, delta_int)), 0, shape[1] - new_x
        )
        new_y = clamp(randint(*minmax_delta(self.y, other.y, delta_int)), 0, shape[0])
        new_h = clamp(
            randint(*minmax_delta(self.h, other.h, delta_int)), 0, shape[0] - new_y
        )
        new_color = gen_new_color(self.color, other.color, delta_int)
        new_opacity = clamp(
            uniform(*minmax_delta(self.x, other.x, delta_float)), 0.0, 1.0
        )

        return Gene(
            x=new_x, y=new_y, w=new_w, h=new_h, color=new_color, opacity=new_opacity
        )


class Picture:
    genes: list[Gene]
    shape: ImgShape
    image: NDArray[np.uint8]

    def __init__(self, genes: list[Gene], shape: ImgShape) -> None:
        self.genes = genes
        self.shape = shape
        self.image = np.zeros(shape, np.uint8)

    @classmethod
    def random(cls, gene_count: int, shape: ImgShape) -> Self:
        genes = [Gene.random(shape) for _ in range(gene_count)]
        return Picture(genes, shape)

    def render(self) -> NDArray[np.uint8]:
        # clear image
        self.image.fill(0)

        for gene in self.genes:
            x = gene.x
            y = gene.y
            h = gene.h
            w = gene.w
            rect = np.full((h, w, 3), gene.color, np.uint8)

            # 重なった部分だけを上手いこと透過させたいのでクロップしてからweighted sum
            # https://stackoverflow.com/a/56472613
            sub_img = self.image[y : y + h, x : x + w]
            opacity = gene.opacity
            cv2.addWeighted(sub_img, opacity, rect, 1 - opacity, 0, dst=sub_img)
            self.image[y : y + h, x : x + w] = sub_img

        return self.image

    def predict(self) -> float:
        # なんか一旦ファイルに落とさないとまともに動かないのでごまかす
        # TODO: Issueが嘘を言ってないか、npのみとBytesIOで比較（既存ファイルで）
        # https://github.com/GantMan/nsfw_model/issues/89#issuecomment-770139533
        # https://stackoverflow.com/a/52865864
        success, cv2_buf = cv2.imencode(".png", self.render())
        bytes_buf = BytesIO(cv2_buf)

        # モデルが食える形にしてやる
        # TODO: BytesIO使わなくてもいけるのでは……(リサイズと正規化できればいい気がする)
        tf_img = tf.keras.preprocessing.image.load_img(
            bytes_buf, target_size=(224, 224)
        )
        tf_image = tf.keras.preprocessing.image.img_to_array(tf_img) / 255.0

        result = predict.classify_nd(
            model,
            np.asarray([tf_image]),
            # プログレスバーを消す
            predict_args={
                "verbose": 0,
            },
        )

        return result[0]["hentai"]

    def crossbreed(self, other: Self, mut_rate: float) -> Self:
        def breed(gs: tuple[Gene, Gene], mut_rate: float) -> Gene:
            if random() < mut_rate:
                # mutation
                return gs[0].crossbreed(gs[1], shape=self.shape)
            else:
                # selection
                return choice(gs)

        return Picture(
            [breed(gs, mut_rate) for gs in zip(self.genes, other.genes)], self.shape
        )


IMG_SHAPE: ImgShape = ImgShape((250, 200, 3))
GENES = 1000
POOL = 50
TOP_N = 5
TO_NEXT = 5
MUT_RATE = 0.5


def breed(l: list[Picture], mut_rate: float) -> Picture:
    return l[0].crossbreed(l[1], mut_rate)


# init
pool = [Picture.random(GENES, IMG_SHAPE) for _ in range(POOL)]

for n in range(1000):
    top = heapq.nlargest(TOP_N, pool, key=lambda x: x.predict())

    print(f"value: {top[0].predict()}")
    if n % 100 == 0:
        cv2.imshow("Image", top[0].render())
        cv2.waitKey(0)

    pool = [
        breed(
            sample(population=top, k=2),
            MUT_RATE,
        )
        for _ in range(POOL - TO_NEXT)
    ] + top[0:TO_NEXT]
