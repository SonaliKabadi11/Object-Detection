from typing import TypedDict

import gradio as gr
from PIL import Image, ImageDraw
from huggingface_hub import snapshot_download
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


def create_interface(detector: Pipeline) -> gr.Interface:
    def detect_objects(
        image: Image.Image | None,
    ) -> tuple[Image.Image, list[list[str | float | int]]]:
        if image is None:
            raise gr.Error("Please upload an image.")

        detections = detector(image.convert("RGB"))
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
        
        return draw_detections(image, detections), rows

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
        ],
        title="Object Detection",
        description="Upload an image to view detected objects, bounding boxes, and confidence scores.",
    )


def main() -> None:
    model_path = snapshot_download(
        "facebook/detr-resnet-50",
        local_files_only=True,
    )
    detector = pipeline("object-detection", model=model_path)
    app = create_interface(detector)
    app.launch(server_name="127.0.0.1", share=False, inbrowser=False)


if __name__ == "__main__":
    main()
