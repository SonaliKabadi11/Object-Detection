from typing import Any, TypedDict

import gradio as gr
from PIL import Image, ImageDraw
from transformers import Pipeline, pipeline


class BoundingBox(TypedDict):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class Detection(TypedDict):
    score: float
    label: str
    box: BoundingBox


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    annotated_image = image.convert("RGB")
    draw = ImageDraw.Draw(annotated_image)

    for detection in detections:
        box = detection["box"]
        draw.rectangle(
            (box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
            outline="red",
            width=3,
        )

    return annotated_image


def create_interface(detector: Pipeline, text_to_speech: Pipeline) -> gr.Interface:
    def detect_objects(
        image: Image.Image | None,
    ) -> tuple[Image.Image, list[list[str | float | int]], tuple[int, Any]]:
        if image is None:
            raise gr.Error("Please upload an image.")

        detections = detector(image.convert("RGB"))
        categories = sorted({detection["label"] for detection in detections})
        summary = (
            f"I detected {len(detections)} objects. The categories are {', '.join(categories)}. "
            "See the table for confidence scores and bounding-box coordinates."
            if detections
            else "I did not detect any objects."
        )
        audio_result = text_to_speech(summary)
        audio = (audio_result["sampling_rate"], audio_result["audio"].squeeze(0))

        rows = [
            [
                detection["label"],
                round(detection["score"], 4),
                detection["box"]["xmin"],
                detection["box"]["ymin"],
                detection["box"]["xmax"],
                detection["box"]["ymax"],
            ]
            for detection in detections
        ]

        return draw_detections(image, detections), rows, audio

    return gr.Interface(
        fn=detect_objects,
        inputs=gr.Image(type="pil", sources=["upload"], label="Upload an image"),
        outputs=[
            gr.Image(type="pil", label="Detected image"),
            gr.Dataframe(
                headers=["Object", "Confidence", "xmin", "ymin", "xmax", "ymax"],
                datatype=["str", "number", "number", "number", "number", "number"],
                label="Detected objects",
                interactive=False,
            ),
            gr.Audio(label="Summary Audio"),
        ],
        title="Object Detection",
        description="Upload an image to see automatically detected objects and hear a spoken summary.",
    )


def main() -> None:
    detector = pipeline("object-detection", model="PekingU/rtdetr_r50vd_coco_o365")
    text_to_speech = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    app = create_interface(detector, text_to_speech)

    app.launch(server_name="127.0.0.1", share=False, inbrowser=False)


if __name__ == "__main__":
    main()
