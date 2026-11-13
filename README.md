# Object Detection

A local Gradio app that automatically detects objects in uploaded images, draws bounding boxes, displays a results table, and plays an English spoken summary.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

Install the project dependencies:

```powershell
uv sync
```

The object-detection and English text-to-speech models are downloaded from Hugging Face on first launch if they are not already cached.

## Run

```powershell
uv run object-detection
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860), upload an image, and click **Submit**. The app automatically detects objects, displays the annotated image and a table with each detection's label, confidence, and bounding-box coordinates, and reads a summary aloud.

## Detection model

The app uses [`PekingU/rtdetr_r50vd_coco_o365`](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365), an RT-DETR model pretrained on Objects365 and fine-tuned on COCO. It detects objects automatically; users do not need to provide labels. Like any trained detector, it may miss or misclassify objects, especially objects outside its trained categories. It is not a guarantee of detecting every possible object.

## Models and licensing

- Object detection: [`PekingU/rtdetr_r50vd_coco_o365`](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365) (Apache 2.0)
- English text-to-speech: [`facebook/mms-tts-eng`](https://huggingface.co/facebook/mms-tts-eng)

The text-to-speech model is licensed under CC BY-NC 4.0, which restricts commercial use. Check each model's license before using or distributing the app.
