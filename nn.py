from typing import Any
from PIL import Image
from transformers import pipeline

classifier = pipeline(
    "image-classification", model="Falconsai/nsfw_image_detection", use_fast=True
)


def classify(images: list[Image.Image]) -> list[float]:
    # [
    #   [{'label': 'nsfw', 'score': 0.9998204112052917}, {'label': 'normal', 'score': 0.00017962830315809697}],
    #   [...]
    # ]
    raw_scores: list[list[dict[str, Any]]] = classifier(images)

    scores: list[float] = [
        next(v2 for v2 in v if v2["label"] == "nsfw")["score"] for v in raw_scores
    ]

    return scores
