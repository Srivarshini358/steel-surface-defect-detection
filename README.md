# Steel Surface Defect Detection and Localization Using YOLOv8

An interactive computer vision application that detects and localizes six types of steel surface defects using a fine-tuned YOLOv8n model trained on the NEU-DET dataset.

## Defect classes
- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

## Tech stack
- Python
- YOLOv8n
- Ultralytics
- Gradio
- NEU-DET dataset

## How it works
1. Upload a steel surface image.
2. The YOLOv8n model predicts defect classes and confidence scores.
3. Bounding boxes localize detected defects.
